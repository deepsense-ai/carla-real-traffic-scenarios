import hashlib
import logging
import os
import random
from typing import Optional

import carla

from carla_real_traffic_scenarios import DT, DONE_CAUSE_KEY
from carla_real_traffic_scenarios.early_stop import EarlyStopMonitor, EarlyStop
from carla_real_traffic_scenarios.ngsim import FRAMES_BEFORE_MANUVEUR, FRAMES_AFTER_MANUVEUR, NGSimDataset, DatasetMode
from carla_real_traffic_scenarios.ngsim.ngsim_recording import NGSimRecording, LaneChangeInstant, PIXELS_TO_METERS
from carla_real_traffic_scenarios.reward import RewardType
from carla_real_traffic_scenarios.scenario import ScenarioStepResult, Scenario, ChauffeurCommand
from carla_real_traffic_scenarios.trajectory import LaneAlignmentMonitor, TARGET_LANE_ALIGNMENT_FRAMES, \
    CROSSTRACK_ERROR_TOLERANCE, YAW_DEG_ERRORS_TOLERANCE, LaneChangeProgressMonitor
from carla_real_traffic_scenarios.utils.carla import RealTrafficVehiclesInCarla, setup_carla_settings
from carla_real_traffic_scenarios.utils.collections import find_first_matching
from carla_real_traffic_scenarios.utils.topology import get_lane_id, get_lane_ids


LOGGER = logging.getLogger(__name__)


def _wp2str(wp, ref=None):
    d = wp.transform.location.distance(ref.transform.location) if ref else 0
    return f'id={get_lane_id(wp)} junct={wp.is_junction} ' \
           f'loc=({wp.transform.location.x:0.2f},{wp.transform.location.y:0.2f}) ' \
           f'yaw={wp.transform.rotation.yaw:0.2f}rad ' \
           f'd={d:0.2f}'


class NGSimLaneChangeScenario(Scenario):
    """
    Possible improvements:
    - include bikes in CARLA to model NGSim motorcycles
    """

    def __init__(
        self,
        ngsim_dataset: NGSimDataset,
        *,
        dataset_mode: DatasetMode,
        data_dir,
        reward_type: RewardType,
        client: carla.Client,
        seed=None
    ):
        super().__init__(client)
        setup_carla_settings(client, synchronous=True, time_delta_s=DT)

        self._ngsim_recording = NGSimRecording(
            data_dir=data_dir, ngsim_dataset=ngsim_dataset,
        )
        self._ngsim_dataset = ngsim_dataset
        self._ngsim_vehicles_in_carla = None
        self._dataset_mode = dataset_mode
        self._early_stop_monitor: Optional[EarlyStopMonitor] = None
        self._progress_monitor: Optional[LaneChangeProgressMonitor] = None
        self._lane_alignment_monitor = LaneAlignmentMonitor(lane_alignment_frames=TARGET_LANE_ALIGNMENT_FRAMES,
                                                            cross_track_error_tolerance=CROSSTRACK_ERROR_TOLERANCE,
                                                            yaw_deg_error_tolerance=YAW_DEG_ERRORS_TOLERANCE)

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
            epseed = os.environ.get("epseed")
            if epseed:
                random.seed(int(epseed))
            self._lane_change: LaneChangeInstant = random.choice(self._lane_change_instants)
            self._sampled_dataset_excerpt_info = dict(
                episode_seed=epseed,
                file_suffix=self._lane_change.timeslot.file_suffix,
                frame_start=self._lane_change.frame_start,
                original_veh_id=self._lane_change.vehicle_id
            )
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
                self._start_lane_ids = get_lane_ids(self._start_lane_waypoint)
                self._target_lane_ids = get_lane_ids(self._target_lane_waypoint)
                assert not (set(self._start_lane_ids) & set(self._target_lane_ids))  # ensure disjoint sets of ids
                break

        self._lane_alignment_monitor.reset()
        self._progress_monitor = LaneChangeProgressMonitor(self._world_map,
                                                           start_lane_ids=self._start_lane_ids,
                                                           target_lane_ids=self._target_lane_ids,
                                                           lane_change_command=self._lane_change.chauffeur_command)

        vehicle.set_transform(t.as_carla_transform())
        v = t.orientation * agent_ngsim_vehicle.speed * PIXELS_TO_METERS
        vehicle.set_velocity(v.to_vector3(0).as_carla_vector3d())  # meters per second,

        other_ngsim_vehicles = [v for v in ngsim_vehicles if v.id != self._lane_change.vehicle_id]
        self._ngsim_vehicles_in_carla.step(other_ngsim_vehicles)

    def step(self, ego_vehicle: carla.Vehicle) -> ScenarioStepResult:
        ngsim_vehicles = self._ngsim_recording.step()
        other_ngsim_vehicles = []
        original_veh_transform = None
        for veh in ngsim_vehicles:
            if veh.id == self._lane_change.vehicle_id:
                original_veh_transform = veh.transform.as_carla_transform()
            else:
                other_ngsim_vehicles.append(veh)

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

        scenario_finished_with_success = on_target_lane and \
                                         self._lane_alignment_monitor.is_lane_aligned(ego_transform, waypoint.transform)

        early_stop = EarlyStop.NONE
        if not scenario_finished_with_success:
            early_stop = self._early_stop_monitor(ego_transform)
            if not_on_expected_lanes:
                early_stop |= EarlyStop.MOVED_TOO_FAR

        done = scenario_finished_with_success | bool(early_stop)
        reward = int(self._reward_type == RewardType.DENSE) * self._progress_monitor.get_progress_change(ego_transform)
        reward += int(scenario_finished_with_success)
        reward += int(bool(early_stop)) * -1

        done_info = {}
        if done and scenario_finished_with_success:
            done_info[DONE_CAUSE_KEY] = 'success'
        elif done and early_stop:
            done_info[DONE_CAUSE_KEY] = early_stop.decomposed_name('_').lower()
        info = {
            'ngsim_dataset': {
                'road': self._ngsim_dataset.name,
                'timeslice': self._lane_change.timeslot.file_suffix,
                'frame': self._ngsim_recording.frame,
                'dataset_mode': self._dataset_mode.name,
            },
            'scenario_data': {
                'ego_veh': ego_vehicle,
                'original_veh_transform': original_veh_transform,
                'original_to_ego_distance': original_veh_transform.location.distance(ego_transform.location)
            },
            'reward_type': self._reward_type.name,
            'on_start_lane': on_start_lane,
            'on_target_lane': on_target_lane,
            'is_junction': waypoint.is_junction if waypoint else False,
            **self._lane_alignment_monitor.info(),
            **done_info
        }
        return ScenarioStepResult(chauffeur_command, reward, done, info)

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
