import os
import logging
from pathlib import Path
import random
from typing import Union, Optional, List, Tuple

import carla
import numpy as np
import scipy.spatial

from carla_real_traffic_scenarios import DT, DONE_CAUSE_KEY
from carla_real_traffic_scenarios.early_stop import EarlyStop, EarlyStopMonitor
from carla_real_traffic_scenarios.ngsim import DatasetMode
from carla_real_traffic_scenarios.opendd.dataset import OpenDDDataset
from carla_real_traffic_scenarios.opendd.recording import OpenDDVehicle, OpenDDRecording
from carla_real_traffic_scenarios.reward import RewardType
from carla_real_traffic_scenarios.scenario import Scenario, ScenarioStepResult, ChauffeurCommand
from carla_real_traffic_scenarios.trajectory import Trajectory
from carla_real_traffic_scenarios.utils.carla import setup_carla_settings, RealTrafficVehiclesInCarla
from carla_real_traffic_scenarios.utils.transforms import Vector2

LOGGER = logging.getLogger()
MAX_DISTANCE_FROM_REF_TRAJECTORY_M = 3
NUM_CHECKPOINTS = 10


def _quantify_progress(progress, num_checkpoints=NUM_CHECKPOINTS):
    return np.floor(progress * num_checkpoints) / num_checkpoints


class Chauffeur:

    def __init__(self, vehicle: OpenDDVehicle, roads_utm: List[Tuple[Vector2, Vector2]]) -> None:
        self.vehicle = vehicle
        self._roads_utm = roads_utm
        self._trajectory_carla = [t.position.as_numpy()[:2] for t in self.vehicle.trajectory_carla]
        self._cmds = self._precalculate_commands()

    def _precalculate_commands(self) -> List[ChauffeurCommand]:
        exits_utm = np.array([exit.as_numpy() if exit else np.zeros(2) for entry, exit in self._roads_utm])
        trajectory_utm = [t.position.as_numpy()[:2] for t in self.vehicle.trajectory_utm]

        # find roundabout exit with which vehicle will drive
        dm_exits = scipy.spatial.distance_matrix(exits_utm, trajectory_utm)
        exit_distances_m = np.min(dm_exits, axis=1)
        exit_idx = np.argmin(exit_distances_m)

        # find roundabout entry just before selected roundabout exit
        before_exit_idx = exit_idx - 1
        dm_entries = scipy.spatial.distance_matrix([self._roads_utm[before_exit_idx][0].as_numpy()], trajectory_utm)

        # find trajectory part with
        lane_change_cmd_idx = np.argmin(dm_entries[0])
        lane_follow_cmd_idx = np.argmin(dm_exits[exit_idx])

        cmds = [ChauffeurCommand.LANE_FOLLOW] * len(trajectory_utm)
        cmds[lane_change_cmd_idx:lane_follow_cmd_idx] = [ChauffeurCommand.TURN_RIGHT] * \
                                                        (lane_follow_cmd_idx - lane_change_cmd_idx)
        return cmds

    def get_cmd(self, transform_carla: carla.Transform):
        location_carla = np.array([transform_carla.location.x, transform_carla.location.y])
        idx = np.argmin(scipy.spatial.distance_matrix([location_carla], self._trajectory_carla), axis=1)[0]
        return self._cmds[idx]


class OpenDDScenario(Scenario):
    def __init__(
        self,
        client: carla.Client,
        *,
        dataset_dir: Union[str, Path],
        reward_type: RewardType,
        dataset_mode: DatasetMode,
        place_name: Optional[str] = None,
    ):
        super().__init__(client)

        setup_carla_settings(client, synchronous=True, time_delta_s=DT)

        self._dataset = OpenDDDataset(dataset_dir)
        self._recording = OpenDDRecording(dataset=self._dataset, dataset_mode=dataset_mode)
        self._dataset_mode = dataset_mode
        self._reward_type = reward_type
        self._place_name = place_name

        self._chauffeur: Optional[Chauffeur] = None
        self._early_stop_monitor: Optional[EarlyStopMonitor] = None
        self._carla_sync: Optional[RealTrafficVehiclesInCarla] = None
        self._current_progress = 0

    def reset(self, ego_vehicle: carla.Vehicle):
        if self._carla_sync:
            self._carla_sync.close()
        self._carla_sync = RealTrafficVehiclesInCarla(self._client, self._world)

        if self._early_stop_monitor:
            self._early_stop_monitor.close()

        session_names = self._dataset.session_names
        if self._place_name:
            session_names = [n for n in session_names if self._place_name.lower() in n]
        epseed = os.environ.get("epseed")
        if epseed:
            epseed = int(epseed)
            random.seed(epseed)
        session_name = random.choice(session_names)
        # Another random is used inside
        ego_id, timestamp_start_s, timestamp_end_s = self._recording.reset(session_name=session_name, seed=epseed)
        self._sampled_dataset_excerpt_info = dict(
            episode_seed=epseed,
            session_name=session_name,
            timestamp_start_s=timestamp_start_s,
            timestamp_end_s=timestamp_end_s,
            original_veh_id=ego_id,
        )
        env_vehicles = self._recording.step()
        other_vehicles = [v for v in env_vehicles if v.id != ego_id]
        self._carla_sync.step(other_vehicles)

        opendd_ego_vehicle = self._recording._env_vehicles[ego_id]
        opendd_ego_vehicle.set_end_of_trajectory_timestamp(timestamp_end_s)
        self._chauffeur = Chauffeur(opendd_ego_vehicle, self._recording.place.roads_utm)

        ego_vehicle.set_transform(opendd_ego_vehicle.transform_carla.as_carla_transform())
        ego_vehicle.set_velocity(opendd_ego_vehicle.velocity.as_carla_vector3d())

        self._current_progress = 0
        trajectory_carla = [t.as_carla_transform() for t in opendd_ego_vehicle.trajectory_carla]
        self._trajectory = Trajectory(trajectory_carla=trajectory_carla)
        timeout_s = (timestamp_end_s - timestamp_start_s) * 1.5
        timeout_s = min(timeout_s, self._recording._timestamps[-1] - timestamp_start_s)
        self._early_stop_monitor = EarlyStopMonitor(ego_vehicle, timeout_s=timeout_s, trajectory=self._trajectory,
                                                    max_trajectory_distance_m=MAX_DISTANCE_FROM_REF_TRAJECTORY_M)

    def step(self, ego_vehicle: carla.Vehicle) -> ScenarioStepResult:
        ego_transform = ego_vehicle.get_transform()
        original_veh_transform = self._chauffeur.vehicle.transform_carla.as_carla_transform()

        progress = self._get_progress(ego_transform)
        progress_change = max(0, _quantify_progress(progress) - _quantify_progress(self._current_progress))
        self._current_progress = progress

        scenario_finished_with_success = progress >= 1.0
        early_stop = EarlyStop.NONE
        if not scenario_finished_with_success:
            early_stop = self._early_stop_monitor(ego_transform)
            if self._recording.has_finished:
                early_stop |= EarlyStop.TIMEOUT
        done = scenario_finished_with_success | bool(early_stop)
        reward = int(self._reward_type == RewardType.DENSE) * progress_change
        reward += int(scenario_finished_with_success)
        reward += int(bool(early_stop)) * -1

        cmd = self._chauffeur.get_cmd(ego_transform)
        done_info = {}
        if done and scenario_finished_with_success:
            done_info[DONE_CAUSE_KEY] = 'success'
        elif done and early_stop:
            done_info[DONE_CAUSE_KEY] = early_stop.decomposed_name('_').lower()

        info = {
            'opendd_dataset': {
                'session': self._recording.session_name,
                'timestamp_s': f'{self._recording.timestamp_s:0.3f}',
                'objid': self._chauffeur.vehicle.id,
                'dataset_mode': self._dataset_mode.name,
            },
            'scenario_data': {
                'ego_veh': ego_vehicle,
                'original_veh_transform': original_veh_transform,
                'original_to_ego_distance': original_veh_transform.location.distance(ego_transform.location)
            },
            'reward_type': self._reward_type.name,
            **done_info
        }

        env_vehicles = self._recording.step()
        other_vehicles = [v for v in env_vehicles if v.id != self._chauffeur.vehicle.id]
        self._carla_sync.step(other_vehicles)

        return ScenarioStepResult(cmd, reward, done, info)

    def close(self):
        if self._early_stop_monitor:
            self._early_stop_monitor.close()
            self._early_stop_monitor = None

        if self._carla_sync:
            self._carla_sync.close()

        if self._recording:
            self._recording.close()
            del self._recording
            self._recording = None

    def _get_progress(self, ego_transform: carla.Transform):
        s, *_ = self._trajectory.find_nearest_trajectory_point(ego_transform)
        return s / self._trajectory.total_length_m
