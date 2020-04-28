import enum
import hashlib
import logging
import random
from enum import auto
from typing import List

import carla
import numpy as np

from sim2real.carla import Vector3, ChauffeurCommand
from sim2real.carla.scenarios import ScenarioStepResult, Scenario
from sim2real.carla.scenarios.lane_change.lane_change_scenario import CROSSTRACK_ERROR_TOLERANCE, \
    YAW_DEG_ERRORS_TOLERANCE, TARGET_LANE_ALIGNMENT_FRAMES
from sim2real.carla.server import CarlaServerController
from sim2real.carla.transforms import Transform, Vector2, distance_between_on_plane
from sim2real.carla.vehicle import Vehicle
from sim2real.ngsim_dataset import FRAMES_BEFORE_MANUVEUR, FRAMES_AFTER_MANUVEUR, NGSimDataset
from carla_real_traffic_scenarios.ngsim.ngsim_carla_sync import NGSimVehiclesInCarla, MAPPER_BY_NGSIM_DATASET
from carla_real_traffic_scenarios.ngsim.ngsim_recording import NGSimRecording, LaneChangeInstant, PIXELS_TO_METERS
from sim2real.utils.collections import find_first_matching

CARLA_TOWN6_SECTION_METERS = 200

LOGGER = logging.getLogger(__name__)


class DatasetMode(enum.Enum):
    TRAIN = auto()
    VALIDATION = auto()


class NGSimLaneChangeScenario(Scenario):
    """
    Possible improvements:
    - include bikes in CARLA to model PPUU motorcycles
    - take different 200-meter subsections of 300-meter long ppuu road instead of taking first 200 meters
    - randomize step-ahead from lane-change moment
    """

    def __init__(self, ngsim_dataset: NGSimDataset, dataset_mode: DatasetMode):
        super().__init__(ngsim_dataset.carla_map,
                         f"NGSIM_LANE_CHANGE_{ngsim_dataset.name.upper()}_{dataset_mode.name.upper()}")
        LARGE_ENOUGH_SO_ALL_DATA_TAKEN = 10000
        self._ngsim_recording = NGSimRecording(
            ngsim_dataset=ngsim_dataset, use_display=False, x_max_meters=LARGE_ENOUGH_SO_ALL_DATA_TAKEN
        )
        self._ngsim_dataset = ngsim_dataset
        self._ngsim_vehicles_in_carla = None
        self._target_alignment_counter: int
        self._dataset_mode = dataset_mode

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
        LOGGER.info(f"Got {len(self._lane_change_instants)} lane change subscenarios in {self.name}")

    def reset(self, carla_server_controller: CarlaServerController, vehicle: Vehicle):
        if self._ngsim_vehicles_in_carla:
            self._ngsim_vehicles_in_carla.close()

        self._lane_change: LaneChangeInstant = random.choice(self._lane_change_instants)

        self._ngsim_vehicles_in_carla = NGSimVehiclesInCarla(carla_server_controller, self._ngsim_dataset)
        self._target_alignment_counter = 0

        self._previous_chauffeur_command = self._lane_change.chauffeur_command
        self._remaining_steps = FRAMES_BEFORE_MANUVEUR + FRAMES_AFTER_MANUVEUR

        frame_manuveur_start = max(self._lane_change.frame_start - FRAMES_BEFORE_MANUVEUR, 0)
        self._ngsim_recording.reset(timeslot=self._lane_change.timeslot, frame=frame_manuveur_start - 1)
        ngsim_vehicles = self._ngsim_recording.step()

        agent_ngsim_vehicle = find_first_matching(ngsim_vehicles, lambda v: v.id == self._lane_change.vehicle_id)
        other_ngsim_vehicles = [v for v in ngsim_vehicles if v.id != self._lane_change.vehicle_id]

        mapper = MAPPER_BY_NGSIM_DATASET[self._ngsim_dataset]
        t = mapper.ngsim_to_carla(agent_ngsim_vehicle.get_transform(), vehicle.vehicle_model.z_offset,
                                  vehicle.vehicle_model.rear_axle_offset)
        vehicle.set_transform(t)
        v = t.orientation * agent_ngsim_vehicle._speed * PIXELS_TO_METERS
        vehicle.set_velocity(v.to_vector3(0))  # meters per second,

        self._ngsim_vehicles_in_carla.step(other_ngsim_vehicles)

    def step(self, vehicle: Vehicle) -> ScenarioStepResult:
        ngsim_vehicles = self._ngsim_recording.step()
        other_ngsim_vehicles = [v for v in ngsim_vehicles if v.id != self._lane_change.vehicle_id]
        self._ngsim_vehicles_in_carla.step(other_ngsim_vehicles)

        waypoint = vehicle.get_current_waypoint()

        done = False
        reward = 0

        on_start_lane = waypoint.lane_id == self._ngsim_dataset.carla_lane_by_ngsim_lane(self._lane_change.lane_from)
        on_target_lane = waypoint.lane_id == self._ngsim_dataset.carla_lane_by_ngsim_lane(self._lane_change.lane_to)

        if on_start_lane:
            chauffeur_command = self._lane_change.chauffeur_command
        elif on_target_lane:
            chauffeur_command = ChauffeurCommand.LANE_FOLLOW

            lane_t = Transform.from_carla_transform(waypoint.transform)
            crosstrack_error = distance_between_on_plane(vehicle.transform, lane_t)
            yaw_error = abs(vehicle.transform.orientation.yaw_radians - lane_t.orientation.yaw_radians)

            aligned_with_target_lane = \
                crosstrack_error < CROSSTRACK_ERROR_TOLERANCE and yaw_error < np.deg2rad(YAW_DEG_ERRORS_TOLERANCE)

            if aligned_with_target_lane:
                self._target_alignment_counter += 1
            else:
                self._target_alignment_counter = 0

            if self._target_alignment_counter == TARGET_LANE_ALIGNMENT_FRAMES:
                done = True
                reward = 1
        else:  # off road
            done = True
            chauffeur_command = ChauffeurCommand.LANE_FOLLOW

        self._remaining_steps -= 1

        if self._remaining_steps == 0:
            done = True

        self._previous_chauffeur_command = chauffeur_command
        info = {
            'ngsim_dataset': {
                'road': self._ngsim_dataset.name,
                'timeslice': self._lane_change.timeslot.file_suffix,
                'frame': self._ngsim_recording.frame,
                'dataset_mode': self._dataset_mode.name
            },
            'target_alignment_counter': self._target_alignment_counter,
        }
        return ScenarioStepResult(chauffeur_command, reward, done, info)

    def get_spawn_points(self, carla_map: carla.Map) -> List[carla.Transform]:
        """Dummy spawn point. Initial vehicle conditions are determined in env reset"""
        # TODO merge reset and get_spawn_point into one thing. In reset return spawn position
        DUMMY_SPAWN_POINT = [Transform(
            Vector3(0, 0, 50),
            Vector2(1, 0)
        ).as_carla_transform()]
        return DUMMY_SPAWN_POINT
