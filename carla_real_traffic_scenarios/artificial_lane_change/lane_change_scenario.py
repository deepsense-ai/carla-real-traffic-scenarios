import logging
from typing import List

import numpy as np
import random
from more_itertools import windowed
from operator import eq

import carla
from carla_real_traffic_scenarios.artificial_lane_change.controller import TeleportCommandsController
from carla_real_traffic_scenarios.scenario import ChauffeurCommand, Scenario, ScenarioStepResult
from carla_real_traffic_scenarios.utils.collections import remove_succesive_duplicates
from carla_real_traffic_scenarios.utils.topology import get_lane_id, Topology
from carla_real_traffic_scenarios.utils.transforms import Transform, Vector3, Vector2, distance_between_on_plane

LOGGER = logging.getLogger(__name__)

TARGET_LANE_ALIGNMENT_FRAMES = 10
CROSSTRACK_ERROR_TOLERANCE = 0.3
YAW_DEG_ERRORS_TOLERANCE = 10
VEHICLE_SLOT_LENGTH_M = 8
ADVERSARY_VEHICLES_NUMBER = 8
MIN_SPEED_MPS = 3.0  # ~11 km/h
JAM_SPEED_MPS = 5.56  # ~20 km/h
CITY_SPEED_MPS = 16.67  # ~60 km/h
HIGHWAY_SPEED_MPS = 38.89  # ~140 km/h

SPEED_RANGE_NAMES = {
    'CONST': (MIN_SPEED_MPS, MIN_SPEED_MPS),
    'SLOW': (MIN_SPEED_MPS, JAM_SPEED_MPS),
    'CITY': (JAM_SPEED_MPS, CITY_SPEED_MPS),
    'HIGHWAY': (CITY_SPEED_MPS, HIGHWAY_SPEED_MPS),
    'WIDE': (MIN_SPEED_MPS, CITY_SPEED_MPS),
}

# it is maximum distance to make a lane change maneuver
# note: (~42m is visible before ego vehicle on bird view; same for distance behind vehicle)
FINAL_POINT_DISTANCE_M = 100


class ArtificialLaneChangeScenario(Scenario):

    def __init__(self, *, client: carla.Client, cmd_for_changing_lane=ChauffeurCommand.CHANGE_LANE_LEFT,
                 env_vehicles_route_length_m=300, speed_range_token: str, no_columns=True):
        super().__init__(client=client)
        start_point = Transform(Vector3(-144.4, -22.41, 0), Vector2(-1.0, 0.0))
        self._find_lane_waypoints(cmd_for_changing_lane, start_point.position.as_carla_location())
        self._cmd_for_changing_lane = cmd_for_changing_lane
        self._done_counter: int = TARGET_LANE_ALIGNMENT_FRAMES

        vehicles_number = 0 if no_columns else ADVERSARY_VEHICLES_NUMBER
        env_vehicles = self._spawn_env_vehicles(vehicles_number)
        self._controllers = self._wrap_with_controllers(env_vehicles)

        env_vehicles_speed_range_mps = SPEED_RANGE_NAMES[speed_range_token]
        self._speed_range_mps = env_vehicles_speed_range_mps
        self._env_vehicle_column_ahead_range_m = (5, 30)

        self._topology = Topology(self._world_map)
        self._routes = self._obtain_routes(self._target_lane_waypoint, env_vehicles_route_length_m)

    def _obtain_routes(self, pass_through_waypoint: carla.Waypoint, total_length: float) -> List[List[carla.Transform]]:
        part_length = total_length / 2
        forward_routes = self._topology.get_forward_routes(pass_through_waypoint, part_length)
        backward_routes = self._topology.get_backward_routes(pass_through_waypoint, part_length)
        routes = [
            backward_route + forward_route
            for backward_route in backward_routes for forward_route in forward_routes
        ]
        return [remove_succesive_duplicates(r, equal_fn=eq) for r in routes]

    def _find_lane_waypoints(self, cmd_for_changing_lane, start_location: carla.Location):
        self._start_lane_waypoint = self._world_map.get_waypoint(start_location)
        if self._start_lane_waypoint is None:
            raise RuntimeError(f'Could not find matching lane for starting location {start_location}')

        self._target_lane_waypoint = {
            ChauffeurCommand.CHANGE_LANE_LEFT: self._start_lane_waypoint.get_left_lane,
            ChauffeurCommand.CHANGE_LANE_RIGHT: self._start_lane_waypoint.get_right_lane,
        }[cmd_for_changing_lane]()
        if self._target_lane_waypoint is None:
            raise RuntimeError(f'Could not find {cmd_for_changing_lane} lane for {start_location} location')

    def _spawn_env_vehicles(self, n: int):
        blueprints = self._world.get_blueprint_library().filter('vehicle.*')
        blueprints = [b for b in blueprints if int(b.get_attribute('number_of_wheels')) == 4]
        spawn_points = self._world_map.get_spawn_points()

        def _get_env_vehicles():
            return [
                v for v in self._world.get_actors().filter('vehicle.*')
                if v.attributes.get('role_name') != 'hero' and v.is_alive
            ]

        env_vehicles = _get_env_vehicles()
        missing_vehicles = n - len(env_vehicles)
        while missing_vehicles:
            cmds = [
                carla.command.SpawnActor(blueprint, spawn_point)
                for blueprint, spawn_point in zip(random.choices(blueprints, k=missing_vehicles),
                                                  random.sample(spawn_points, k=missing_vehicles))
            ]
            self._client.apply_batch_sync(cmds, do_tick=True)
            env_vehicles = _get_env_vehicles()
            missing_vehicles = n - len(env_vehicles)
        return env_vehicles

    def _wrap_with_controllers(self, env_vehicles):
        return {
            str(v.id): TeleportCommandsController(v)
            for v in env_vehicles
        }

    def reset(self, ego_vehicle: carla.Vehicle):
        self._done_counter = TARGET_LANE_ALIGNMENT_FRAMES
        self._route = random.choice(self._routes)

        min_speed, max_speed = self._speed_range_mps
        range_size = max_speed - min_speed
        speed_mps = min_speed + np.random.random() * range_size

        column_ahead_of_ego_m = np.random.randint(*self._env_vehicle_column_ahead_range_m)
        cmds = self._setup_controllers(self._start_lane_waypoint.transform.location, speed_mps,
                                       self._route, column_ahead_of_ego_m)
        cmds.append(carla.command.ApplyTransform(ego_vehicle.id, transform=self._start_lane_waypoint.transform))
        if cmds:
            self._client.apply_batch_sync(cmds, do_tick=False)

    def step(self, ego_vehicle: carla.Vehicle) -> ScenarioStepResult:

        self._move_env_vehicles()

        ego_vehicle_transform = Transform.from_carla_transform(ego_vehicle.get_transform())
        current_lane_waypoint = self._world_map.get_waypoint(ego_vehicle_transform.position.as_carla_location())
        current_lane = get_lane_id(current_lane_waypoint)

        # TODO: in fact there are many different lanes which are allowed to go
        allowed_lanes = list(map(get_lane_id, [self._start_lane_waypoint, self._target_lane_waypoint]))

        on_target_lane = current_lane == get_lane_id(self._target_lane_waypoint)
        offroad = current_lane not in allowed_lanes

        info = {
            "target_lane_aligmnent_counter": self._done_counter
        }

        chauffeur_cmd = self._cmd_for_changing_lane
        done = offroad
        if on_target_lane:  # not finished yet
            current_lane_transform = Transform.from_carla_transform(current_lane_waypoint.transform)

            crosstrack_error, yaw_error = self._calculate_errors(current_lane_transform, ego_vehicle_transform)
            aligned_with_target_lane = crosstrack_error < CROSSTRACK_ERROR_TOLERANCE and \
                                       yaw_error < np.deg2rad(YAW_DEG_ERRORS_TOLERANCE)
            if aligned_with_target_lane:
                self._done_counter -= 1
            else:
                self._done_counter = TARGET_LANE_ALIGNMENT_FRAMES

            chauffeur_cmd = ChauffeurCommand.LANE_FOLLOW
            done = self._done_counter == 0

        reward = self._calculate_reward(done)
        return ScenarioStepResult(chauffeur_cmd, reward, done, info)

    def _calculate_reward(self, done):
        return int(done)

    def _calculate_errors(self, current_lane_transform, ego_vehicle_transform):
        crosstrack_error = distance_between_on_plane(ego_vehicle_transform, current_lane_transform)
        yaw_error = abs(ego_vehicle_transform.orientation.yaw_radians - current_lane_transform.orientation.yaw_radians)
        return crosstrack_error, yaw_error

    def _setup_controllers(self, ego_vehicle_location, speed_mps, route, column_ahead_of_ego_m):
        resolution_m = np.median([t1.location.distance(t2.location) for t1, t2 in windowed(route, 2)])
        m2idx = 1 / resolution_m

        ego_vehicle_idx = int(np.argmin([t.location.distance(ego_vehicle_location) for t in route]))
        column_start_idx = ego_vehicle_idx + int(column_ahead_of_ego_m * m2idx)

        # env vehicles spacing on route
        def _calc_offset(current_idx, controllers_left):
            slot_length = int(np.ceil(VEHICLE_SLOT_LENGTH_M * m2idx))

            remain_route_for_random_space = max(current_idx - controllers_left * slot_length, 0)
            remain_route_for_single_random_space = remain_route_for_random_space // controllers_left * 1.5

            random_space = 0
            if remain_route_for_single_random_space:
                random_space = np.random.randint(0, remain_route_for_single_random_space)

            return slot_length + random_space

        ncontrollers = len(self._controllers)
        current_idx = column_start_idx
        cmds = []
        for idx, controller in enumerate(self._controllers.values()):
            initial_location = route[current_idx].location
            cmds.extend(controller.reset(speed_mps=speed_mps, route=route, initial_location=initial_location))
            controllers_left = ncontrollers - idx
            offset = _calc_offset(current_idx, controllers_left)
            current_idx = max(current_idx - offset, 0)

        return cmds

    def _move_env_vehicles(self):
        cmds = []
        for controller in self._controllers.values():
            finished, cmds_ = controller.step()
            cmds.extend(cmds)
            if finished:
                cmds.extend(controller.reset())  # reset its position to beginning of the route
        if cmds:
            self._client.apply_batch_sync(cmds, do_tick=False)
