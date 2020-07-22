from queue import Queue

import hashlib
import logging
import random
from typing import Optional

import carla
import numpy as np

from carla_real_traffic_scenarios import DT
from carla_real_traffic_scenarios.early_stop import EarlyStopMonitor, EarlyStop
from carla_real_traffic_scenarios.ngsim import FRAMES_BEFORE_MANUVEUR, FRAMES_AFTER_MANUVEUR, NGSimDataset, DatasetMode
from carla_real_traffic_scenarios.ngsim.ngsim_recording import NGSimRecording, LaneChangeInstant, PIXELS_TO_METERS
from carla_real_traffic_scenarios.reward import RewardType
from carla_real_traffic_scenarios.scenario import ScenarioStepResult, Scenario, ChauffeurCommand
from carla_real_traffic_scenarios.utils.carla import RealTrafficVehiclesInCarla, setup_carla_settings
from carla_real_traffic_scenarios.utils.collections import find_first_matching
from carla_real_traffic_scenarios.utils.geometry import normalize_angle
from carla_real_traffic_scenarios.utils.topology import get_lane_id
from carla_real_traffic_scenarios.utils.transforms import distance_between_on_plane
from sim2real.runner import DONE_CAUSE_KEY

CROSSTRACK_ERROR_TOLERANCE = 0.3
YAW_DEG_ERRORS_TOLERANCE = 10
TARGET_LANE_ALIGNMENT_FRAMES = 10

LOGGER = logging.getLogger(__name__)


def _unroll_waypoint(wp, max_distance, step, backward=True):
    ref_wp = wp
    q = Queue()
    q.put(wp)
    waypoints = [wp]
    while not q.empty():
        wp = q.get()
        tmp = wp.previous(step) if backward else wp.next(step)
        for w in tmp:
            if w.transform.location.distance(ref_wp.transform.location) < max_distance:
                q.put(w)
        waypoints.extend(tmp)
    return waypoints


def _wp2str(wp, ref=None):
    d = wp.transform.location.distance(ref.transform.location) if ref else 0
    return f'id={get_lane_id(wp)} junct={wp.is_junction} ' \
           f'loc=({wp.transform.location.x:0.2f},{wp.transform.location.y:0.2f}) ' \
           f'yaw={wp.transform.rotation.yaw:0.2f}rad ' \
           f'd={d:0.2f}'


def _get_lane_ids(lane_wp):
    return sorted(set(
        [get_lane_id(wp) for wp in _unroll_waypoint(lane_wp, 300, 2)] + \
        [get_lane_id(wp) for wp in _unroll_waypoint(lane_wp, 300, 2, backward=False)]
    ))


class NGSimLaneChangeScenario(Scenario):
    """
    Possible improvements:
    - include bikes in CARLA to model NGSim motorcycles
    """

    def __init__(self, ngsim_dataset: NGSimDataset, *, dataset_mode: DatasetMode, data_dir, reward_type: RewardType,
                 client: carla.Client):
        super().__init__(client)

        setup_carla_settings(client, synchronous=True, time_delta_s=DT)

        self._ngsim_recording = NGSimRecording(
            data_dir=data_dir, ngsim_dataset=ngsim_dataset,
        )
        self._ngsim_dataset = ngsim_dataset
        self._ngsim_vehicles_in_carla = None
        self._target_alignment_counter: int
        self._dataset_mode = dataset_mode
        self._early_stop_monitor: Optional[EarlyStopMonitor] = None

        def determine_split(lane_change_instant: LaneChangeInstant) -> DatasetMode:
            split_frac = 0.8
            hash_num = int(hashlib.sha1(str(lane_change_instant).encode('utf-8')).hexdigest(), 16)
            if (hash_num % 100) / 100 < split_frac:
                return DatasetMode.TRAIN
            else:
                return DatasetMode.VALIDATION

        self._lane_change_instants = [
            lci for lci in self._ngsim_recording.lane_change_instants if determine_split(lci) == dataset_mode
        ]
        LOGGER.info(
            f"Got {len(self._lane_change_instants)} lane change subscenarios "
            f"in {ngsim_dataset.name}_{dataset_mode.name}")
        self._reward_type = reward_type

    def reset(self, vehicle: carla.Vehicle):
        if self._ngsim_vehicles_in_carla:
            self._ngsim_vehicles_in_carla.close()

        self._ngsim_vehicles_in_carla = RealTrafficVehiclesInCarla(self._client, self._world)

        if self._early_stop_monitor:
            self._early_stop_monitor.close()

        timeout_s = (FRAMES_BEFORE_MANUVEUR + FRAMES_AFTER_MANUVEUR) * DT
        self._early_stop_monitor = EarlyStopMonitor(vehicle, timeout_s=timeout_s)

        while True:
            self._lane_change: LaneChangeInstant = random.choice(self._lane_change_instants)
            frame_manuveur_start = max(self._lane_change.frame_start - FRAMES_BEFORE_MANUVEUR, 0)
            self._ngsim_recording.reset(timeslot=self._lane_change.timeslot, frame=frame_manuveur_start - 1)
            ngsim_vehicles = self._ngsim_recording.step()
            agent_ngsim_vehicle = find_first_matching(ngsim_vehicles, lambda v: v.id == self._lane_change.vehicle_id)
            t = agent_ngsim_vehicle.transform
            self._start_lane_waypoint = self._world_map.get_waypoint(t.as_carla_transform().location)
            self._target_lane_waypoint = {
                ChauffeurCommand.CHANGE_LANE_LEFT: self._start_lane_waypoint.get_left_lane,
                ChauffeurCommand.CHANGE_LANE_RIGHT: self._start_lane_waypoint.get_right_lane,
            }[self._lane_change.chauffeur_command]()

            if self._start_lane_waypoint and self._target_lane_waypoint:
                self._start_lane_ids = _get_lane_ids(self._start_lane_waypoint)
                self._target_lane_ids = _get_lane_ids(self._target_lane_waypoint)
                assert not (set(self._start_lane_ids) & set(self._target_lane_ids))  # ensure disjoint sets of ids
                break

        self._target_alignment_counter = 0
        self._previous_progress = 0
        self._total_distance_m = None
        self._checkpoints_distance_m = None

        vehicle.set_transform(t.as_carla_transform())
        v = t.orientation * agent_ngsim_vehicle.speed * PIXELS_TO_METERS
        vehicle.set_velocity(v.to_vector3(0).as_carla_vector3d())  # meters per second,

        other_ngsim_vehicles = [v for v in ngsim_vehicles if v.id != self._lane_change.vehicle_id]
        self._ngsim_vehicles_in_carla.step(other_ngsim_vehicles)

    def step(self, ego_vehicle: carla.Vehicle) -> ScenarioStepResult:
        ngsim_vehicles = self._ngsim_recording.step()
        other_ngsim_vehicles = [v for v in ngsim_vehicles if v.id != self._lane_change.vehicle_id]
        self._ngsim_vehicles_in_carla.step(other_ngsim_vehicles)

        ego_transform = ego_vehicle.get_transform()
        waypoint = self._world_map.get_waypoint(ego_transform.location)

        on_start_lane = False
        on_target_lane = False

        if waypoint:
            lane_id = get_lane_id(waypoint)
            on_start_lane = lane_id in self._start_lane_ids
            on_target_lane = lane_id in self._target_lane_ids

        not_on_expected_lanes = not (on_start_lane or on_target_lane)
        chauffeur_command = self._lane_change.chauffeur_command if on_start_lane else ChauffeurCommand.LANE_FOLLOW

        scenario_finished_with_success = False
        alignment_errors = None
        if on_target_lane:
            aligned, alignment_errors = self._is_lane_aligned(ego_transform, waypoint)
            scenario_finished_with_success |= aligned

        early_stop = EarlyStop.NONE
        if not scenario_finished_with_success:
            early_stop = self._early_stop_monitor(ego_transform)
            if not_on_expected_lanes:
                early_stop |= EarlyStop.MOVED_TOO_FAR

        done = scenario_finished_with_success | bool(early_stop)
        reward = int(self._reward_type == RewardType.DENSE) * self._get_progress_change(ego_transform)
        reward += int(scenario_finished_with_success)
        reward += int(bool(early_stop)) * -1

        done_info = {}
        if done and scenario_finished_with_success:
            done_info[DONE_CAUSE_KEY] = 'success'
        elif done and early_stop:
            done_info[DONE_CAUSE_KEY] = EarlyStop(early_stop).decomposed_name('_').lower()
        info = {
            'ngsim_dataset': {
                'road': self._ngsim_dataset.name,
                'timeslice': self._lane_change.timeslot.file_suffix,
                'frame': self._ngsim_recording.frame,
                'dataset_mode': self._dataset_mode.name,
            },
            'reward_type': self._reward_type.name,
            'on_start_lane': on_start_lane,
            'on_target_lane': on_target_lane,
            'is_junction': waypoint.is_junction,
            'alignment_errors': alignment_errors,
            'target_alignment_counter': self._target_alignment_counter,
            **done_info
        }
        return ScenarioStepResult(chauffeur_command, reward, done, info)

    def _is_lane_aligned(self, ego_transform, waypoint):
        crosstrack_error = distance_between_on_plane(ego_transform.location, waypoint.transform.location)
        yaw_error = normalize_angle(np.deg2rad(ego_transform.rotation.yaw - waypoint.transform.rotation.yaw))
        aligned_with_target_lane = crosstrack_error < CROSSTRACK_ERROR_TOLERANCE and \
                                   yaw_error < np.deg2rad(YAW_DEG_ERRORS_TOLERANCE)
        if aligned_with_target_lane:
            self._target_alignment_counter += 1
        else:
            self._target_alignment_counter = 0
        return self._target_alignment_counter == TARGET_LANE_ALIGNMENT_FRAMES, \
               str(f'track={crosstrack_error:0.2f}m yaw={yaw_error:0.2f}rad')

    def close(self):
        if self._early_stop_monitor:
            self._early_stop_monitor.close()
            self._early_stop_monitor = None

        if self._ngsim_vehicles_in_carla:
            self._ngsim_vehicles_in_carla.close()
            self._ngsim_vehicles_in_carla = None

        self._lane_change_instants = []
        self._lane_change = None

        del self._ngsim_recording
        self._ngsim_recording = None

    def _get_progress_change(self, ego_transform: carla.Transform):

        current_location = ego_transform.location
        current_waypoint = self._world_map.get_waypoint(current_location)
        lane_id = get_lane_id(current_waypoint)
        on_start_lane = lane_id in self._start_lane_ids
        on_target_lane = lane_id in self._target_lane_ids

        checkpoints_number = 10
        if self._total_distance_m is None:
            target_lane_location = self._target_lane_waypoint.transform.location
            self._total_distance_m = current_location.distance(target_lane_location)
            self._checkpoints_distance_m = self._total_distance_m / checkpoints_number

        def _calc_progress_change(target_waypoint, current_location):
            distance_from_target = current_location.distance(target_waypoint.transform.location)
            distance_traveled_m = self._total_distance_m - distance_from_target
            checkpoints_passed_number = int(distance_traveled_m / self._checkpoints_distance_m)
            progress = checkpoints_passed_number / checkpoints_number
            progress_change = progress - self._previous_progress
            self._previous_progress = progress
            return progress_change

        progress_change = 0
        if on_start_lane:
            target_waypoint = {
                ChauffeurCommand.CHANGE_LANE_LEFT: current_waypoint.get_left_lane,
                ChauffeurCommand.CHANGE_LANE_RIGHT: current_waypoint.get_right_lane,
            }[self._lane_change.chauffeur_command]()
            if target_waypoint:
                progress_change = _calc_progress_change(target_waypoint, current_location)
            else:
                LOGGER.info('Have no lane to perform maneuver')
        elif on_target_lane:
            target_waypoint = current_waypoint
            progress_change = _calc_progress_change(target_waypoint, current_location)

        return progress_change
