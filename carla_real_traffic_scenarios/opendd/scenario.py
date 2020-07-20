import logging
from pathlib import Path
import random
from typing import Union, Optional, List, Tuple

import carla
import numpy as np
import scipy.spatial

from carla_real_traffic_scenarios import DT
from carla_real_traffic_scenarios.early_stop import EarlyStop, EarlyStopMonitor
from carla_real_traffic_scenarios.ngsim import DatasetMode
from carla_real_traffic_scenarios.opendd.dataset import OpenDDDataset
from carla_real_traffic_scenarios.opendd.recording import OpenDDVehicle, OpenDDRecording
from carla_real_traffic_scenarios.reward import RewardCalculator, DenseRewardCalculator, SparseRewardCalculator, \
    RewardType
from carla_real_traffic_scenarios.scenario import Scenario, ScenarioStepResult, ChauffeurCommand
from carla_real_traffic_scenarios.utils.carla import setup_carla_settings, RealTrafficVehiclesInCarla
from carla_real_traffic_scenarios.utils.transforms import Vector2
from sim2real.runner import DONE_CAUSE_KEY

LOGGER = logging.getLogger()


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

    def __init__(self, client: carla.Client, *, dataset_dir: Union[str, Path], reward_type: RewardType,
                 dataset_mode: DatasetMode, place_name: Optional[str] = None):
        super().__init__(client)

        setup_carla_settings(client, synchronous=True, time_delta_s=DT)

        self._dataset = OpenDDDataset(dataset_dir)
        self._recording = OpenDDRecording(dataset=self._dataset, dataset_mode=dataset_mode)
        self._dataset_mode = dataset_mode
        self._reward_type = reward_type
        self._place_name = place_name

        self._chauffeur: Optional[Chauffeur] = None
        self._early_stop_monitor: Optional[EarlyStopMonitor] = None
        self._reward_calculator: Optional[RewardCalculator] = None
        self._carla_sync: Optional[RealTrafficVehiclesInCarla] = None

        self._timeout_s = None

    def reset(self, ego_vehicle: carla.Vehicle):
        if self._carla_sync:
            self._carla_sync.close()
        self._carla_sync = RealTrafficVehiclesInCarla(self._client, self._world)

        if self._early_stop_monitor:
            self._early_stop_monitor.close()

        session_names = self._dataset.session_names
        if self._place_name:
            session_names = [n for n in session_names if self._place_name.lower() in n]
        session_name = random.choice(session_names)
        ego_id, timestamp_start_s, timestamp_end_s = self._recording.reset(session_name=session_name)

        timeout_s = (timestamp_end_s - timestamp_start_s) * 1.5
        timeout_s = min(timeout_s, self._recording._timestamps[-1] - timestamp_start_s)
        self._early_stop_monitor = EarlyStopMonitor(ego_vehicle, timeout_s=timeout_s)

        env_vehicles = self._recording.step()
        other_vehicles = [v for v in env_vehicles if v.id != ego_id]
        self._carla_sync.step(other_vehicles)

        opendd_ego_vehicle = self._recording._env_vehicles[ego_id]
        opendd_ego_vehicle.set_end_of_trajectory_timestamp(timestamp_end_s)
        self._chauffeur = Chauffeur(opendd_ego_vehicle, self._recording.place.roads_utm)
        trajectory_carla = [t.as_carla_transform() for t in opendd_ego_vehicle.trajectory_carla]
        self._reward_calculator = {
            RewardType.SPARSE: SparseRewardCalculator,
            RewardType.DENSE: DenseRewardCalculator
        }[self._reward_type](trajectory_carla)

        ego_vehicle.set_transform(opendd_ego_vehicle.transform_carla.as_carla_transform())
        ego_vehicle.set_velocity(opendd_ego_vehicle.velocity.as_carla_vector3d())

        self._timeout_s = timestamp_start_s + (timestamp_end_s - timestamp_start_s) * 1.5

    def step(self, ego_vehicle: carla.Vehicle) -> ScenarioStepResult:
        ego_transform = ego_vehicle.get_transform()

        early_stop = EarlyStop.NONE
        reward, done, early_stop_ = self._reward_calculator(ego_transform)
        scenario_finished_with_success = done and not early_stop_
        if not scenario_finished_with_success:
            early_stop = self._early_stop_monitor(ego_transform)
            if early_stop_:
                early_stop |= EarlyStop.MOVED_TOO_FAR
            if self._recording.has_finished:
                early_stop |= EarlyStop.TIMEOUT

        done = scenario_finished_with_success | bool(early_stop)
        reward += int(bool(early_stop)) * -1
        reward += int(scenario_finished_with_success)

        cmd = self._chauffeur.get_cmd(ego_transform)
        done_info = {}
        if done and scenario_finished_with_success:
            done_info[DONE_CAUSE_KEY] = 'success'
        elif done and early_stop:
            done_info[DONE_CAUSE_KEY] = EarlyStop(early_stop).decomposed_name('_').lower()

        info = {
            'opendd_dataset': {
                'session': self._recording.session_name,
                'timestamp_s': f'{self._recording.timestamp_s:0.3f}',
                'objid': self._chauffeur.vehicle.id,
                'dataset_mode': self._dataset_mode.name,
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
