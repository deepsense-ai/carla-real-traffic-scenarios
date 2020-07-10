import hashlib
import logging
import random
from typing import Optional

import carla
import numpy as np

from carla_real_traffic_scenarios import DT
from carla_real_traffic_scenarios.ngsim import FRAMES_BEFORE_MANUVEUR, FRAMES_AFTER_MANUVEUR, NGSimDataset, DatasetMode
from carla_real_traffic_scenarios.ngsim.ngsim_recording import NGSimRecording, LaneChangeInstant, PIXELS_TO_METERS
from carla_real_traffic_scenarios.scenario import ScenarioStepResult, Scenario, ChauffeurCommand
from carla_real_traffic_scenarios.utils.carla import RealTrafficVehiclesInCarla
from carla_real_traffic_scenarios.utils.collections import find_first_matching
from carla_real_traffic_scenarios.utils.geometry import normalize_angle
from carla_real_traffic_scenarios.utils.transforms import distance_between_on_plane

CROSSTRACK_ERROR_TOLERANCE = 0.3
YAW_DEG_ERRORS_TOLERANCE = 10
TARGET_LANE_ALIGNMENT_FRAMES = 10

LOGGER = logging.getLogger(__name__)


class NGSimLaneChangeScenario(Scenario):
    """
    Possible improvements:
    - include bikes in CARLA to model NGSim motorcycles
    """

    def __init__(
        self,
        ngsim_dataset: NGSimDataset,
        dataset_mode: DatasetMode,
        data_dir,
        client: carla.Client,
        evaluation_scenario_seed: Optional[int] = None,
    ):
        super().__init__(client)

        settings = self._world.get_settings()
        if not settings.synchronous_mode:
            LOGGER.warning("Only synchronous_mode==True supported. Will change to synchronous_mode==True")
            settings.synchronous_mode = True
        if settings.fixed_delta_seconds != DT:
            LOGGER.warning(f"Only fixed_delta_seconds=={DT} supported. Will change to fixed_delta_seconds=={DT}")
            settings.fixed_delta_seconds = DT
        self._world.apply_settings(settings)

        self._ngsim_recording = NGSimRecording(
            data_dir=data_dir, ngsim_dataset=ngsim_dataset,
        )
        self._ngsim_dataset = ngsim_dataset
        self._ngsim_vehicles_in_carla = None
        self._collision_sensor = None
        self._target_alignment_counter: int
        self._dataset_mode = dataset_mode
        self._evaluation_scenario_seed = evaluation_scenario_seed

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
            f"Got {len(self._lane_change_instants)} lane change subscenarios in {ngsim_dataset.name}_{dataset_mode.name}")

    def reset(self, vehicle: carla.Vehicle):
        if self._collision_sensor:
            self._collision_sensor.destroy()
            self._collision_sensor = None
        self._collided = False

        if self._ngsim_vehicles_in_carla:
            self._ngsim_vehicles_in_carla.close()

        # attack collision sensor
        blueprint_library = self._world.get_blueprint_library()
        blueprint = blueprint_library.find('sensor.other.collision')
        self._collision_sensor = self._world.spawn_actor(blueprint, vehicle.get_transform(), attach_to=vehicle)

        def on_collided(e):
            self._collided = True

        self._collision_sensor.listen(on_collided)

        if self._evaluation_scenario_seed is not None:
            random.seed(self._evaluation_scenario_seed)
        self._lane_change: LaneChangeInstant = random.choice(self._lane_change_instants)
        print(self._lane_change)

        self._ngsim_vehicles_in_carla = RealTrafficVehiclesInCarla(self._client, self._world)
        self._target_alignment_counter = 0

        self._previous_chauffeur_command = self._lane_change.chauffeur_command
        self._remaining_steps = FRAMES_BEFORE_MANUVEUR + FRAMES_AFTER_MANUVEUR

        frame_manuveur_start = max(self._lane_change.frame_start - FRAMES_BEFORE_MANUVEUR, 0)
        self._ngsim_recording.reset(timeslot=self._lane_change.timeslot, frame=frame_manuveur_start - 1)
        ngsim_vehicles = self._ngsim_recording.step()

        agent_ngsim_vehicle = find_first_matching(ngsim_vehicles, lambda v: v.id == self._lane_change.vehicle_id)
        other_ngsim_vehicles = [v for v in ngsim_vehicles if v.id != self._lane_change.vehicle_id]

        t = agent_ngsim_vehicle.transform
        vehicle.set_transform(t.as_carla_transform())
        v = t.orientation * agent_ngsim_vehicle.speed * PIXELS_TO_METERS
        vehicle.set_velocity(v.to_vector3(0).as_carla_vector3d())  # meters per second,

        self._ngsim_vehicles_in_carla.step(other_ngsim_vehicles)

    def step(self, ego_vehicle: carla.Vehicle) -> ScenarioStepResult:
        ngsim_vehicles = self._ngsim_recording.step()
        other_ngsim_vehicles = [v for v in ngsim_vehicles if v.id != self._lane_change.vehicle_id]
        self._ngsim_vehicles_in_carla.step(other_ngsim_vehicles)

        waypoint = self._world_map.get_waypoint(ego_vehicle.get_transform().location)

        done = False
        reward = 0

        on_start_lane = waypoint.lane_id == self._ngsim_dataset.carla_lane_by_ngsim_lane(self._lane_change.lane_from)
        on_target_lane = waypoint.lane_id == self._ngsim_dataset.carla_lane_by_ngsim_lane(self._lane_change.lane_to)

        if on_start_lane:
            chauffeur_command = self._lane_change.chauffeur_command
        elif on_target_lane:
            chauffeur_command = ChauffeurCommand.LANE_FOLLOW

            crosstrack_error = distance_between_on_plane(
                ego_vehicle.get_transform().location, waypoint.transform.location
            )
            yaw_error = normalize_angle(
                np.deg2rad(ego_vehicle.get_transform().rotation.yaw - waypoint.transform.rotation.yaw)
            )

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

        if self._collided:
            done = True

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

    def close(self):
        if self._ngsim_vehicles_in_carla:
            self._ngsim_vehicles_in_carla.close()
            self._ngsim_vehicles_in_carla = None

        if self._collision_sensor and self._collision_sensor.is_alive:
            self._collision_sensor.destroy()

        self._lane_change_instants = []
        self._lane_change = None

        del self._ngsim_recording
        self._ngsim_recording = None
