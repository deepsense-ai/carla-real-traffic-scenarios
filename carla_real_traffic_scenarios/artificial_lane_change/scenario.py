import logging
from typing import List, Optional

import carla
import numpy as np
import random
from more_itertools import windowed, unique_justseen

from carla_real_traffic_scenarios import DT
from carla_real_traffic_scenarios.artificial_lane_change.controller import TeleportCommandsController
from carla_real_traffic_scenarios.early_stop import EarlyStop, EarlyStopMonitor
from carla_real_traffic_scenarios.reward import RewardType
from carla_real_traffic_scenarios.scenario import ChauffeurCommand, Scenario, ScenarioStepResult
from carla_real_traffic_scenarios.trajectory import LaneAlignmentMonitor, LaneChangeProgressMonitor, \
    TARGET_LANE_ALIGNMENT_FRAMES, CROSSTRACK_ERROR_TOLERANCE, YAW_DEG_ERRORS_TOLERANCE
from carla_real_traffic_scenarios.utils.topology import get_lane_id, Topology, get_lane_ids
from carla_real_traffic_scenarios.utils.transforms import Transform, Vector3, Vector2
from sim2real.runner import DONE_CAUSE_KEY

LOGGER = logging.getLogger(__name__)

FRAMES_BEFORE_MANUVEUR = 50
FRAMES_AFTER_MANUVEUR = 50

VEHICLE_SLOT_LENGTH_M = 8
MAX_VEHICLE_RANDOM_SPACE_M = 15   # with current parameters - quant 5
MAX_MANEUVER_LENGTH_M = 200
BIRD_VIEW_HEIGHT_M = 100

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


# env vehicles spacing on route
def _calc_offset(current_idx, controllers_left, resolution_m):
    m2idx = 1 / resolution_m
    slot_length = int(np.ceil(VEHICLE_SLOT_LENGTH_M * m2idx))

    max_random_space = int(MAX_VEHICLE_RANDOM_SPACE_M * m2idx)
    random_space = np.random.randint(0, max_random_space) if max_random_space else 0

    return slot_length + random_space


def _is_behind_ego_or_inside_birdview(c, ego_vehicle_location):
    env_vehicle_direction = Vector3.from_carla_location(c.forward_vector).as_numpy()
    relative_vector = (Vector3.from_carla_location(c.location) -
                       Vector3.from_carla_location(ego_vehicle_location)).as_numpy()
    deviation = np.arccos(np.clip(np.dot(env_vehicle_direction, relative_vector) /
                                  (np.linalg.norm(env_vehicle_direction) * np.linalg.norm(relative_vector)), -1.0, 1.0))
    is_inside = True
    is_behind_ego = np.abs(deviation) > np.pi / 1.5
    if not is_behind_ego:
        x, y = ego_vehicle_location.x, ego_vehicle_location.y
        half_range = BIRD_VIEW_HEIGHT_M / 2 * 1.4
        xmin, xmax = x - half_range, x + half_range
        ymin, ymax = y - half_range, y + half_range
        is_inside = xmin <= c.location.x <= xmax and ymin <= c.location.y <= ymax
    return is_inside


class ArtificialLaneChangeScenario(Scenario):

    def __init__(self, *, client: carla.Client, cmd_for_changing_lane=ChauffeurCommand.CHANGE_LANE_LEFT,
                 speed_range_token: str, no_columns=True, reward_type: RewardType = RewardType.DENSE):
        super().__init__(client=client)
        start_point = Transform(Vector3(-144.4, -22.41, 0), Vector2(-1.0, 0.0))
        self._find_lane_waypoints(cmd_for_changing_lane, start_point.position.as_carla_location())
        self._cmd_for_changing_lane = cmd_for_changing_lane
        self._reward_type = reward_type

        self._progress_monitor: Optional[LaneChangeProgressMonitor] = None
        self._lane_alignment_monitor = LaneAlignmentMonitor(lane_alignment_frames=TARGET_LANE_ALIGNMENT_FRAMES,
                                                            cross_track_error_tolerance=CROSSTRACK_ERROR_TOLERANCE,
                                                            yaw_deg_error_tolerance=YAW_DEG_ERRORS_TOLERANCE)
        self._early_stop_monitor: Optional[EarlyStopMonitor] = None

        # vehicles shall fill bird view + first vehicle shall reach end of route forward part
        # till the last one reaches bottom of the bird view; assume just VEHICLE_SLOT_LENGTH_M spacing
        max_env_vehicles_number = int(BIRD_VIEW_HEIGHT_M * 1.2 // VEHICLE_SLOT_LENGTH_M)
        env_vehicles = [] if no_columns else self._spawn_env_vehicles(max_env_vehicles_number)
        self._controllers = self._wrap_with_controllers(env_vehicles)

        env_vehicles_speed_range_mps = SPEED_RANGE_NAMES[speed_range_token]
        self._speed_range_mps = env_vehicles_speed_range_mps
        self._env_vehicle_column_ahead_range_m = (30, 50)

        route_length_m = max(MAX_MANEUVER_LENGTH_M + BIRD_VIEW_HEIGHT_M,
                             max_env_vehicles_number * (MAX_VEHICLE_RANDOM_SPACE_M + VEHICLE_SLOT_LENGTH_M)) * 3
        self._topology = Topology(self._world_map)
        self._routes: List[List[carla.Transform]] = self._obtain_routes(self._target_lane_waypoint, route_length_m)

    def _obtain_routes(self, pass_through_waypoint: carla.Waypoint, total_length_m: float) \
            -> List[List[carla.Transform]]:
        part_length_m = total_length_m / 2
        forward_routes = self._topology.get_forward_routes(pass_through_waypoint, part_length_m)
        backward_routes = self._topology.get_backward_routes(pass_through_waypoint, part_length_m)
        routes = [
            backward_route + forward_route
            for backward_route in backward_routes for forward_route in forward_routes
        ]
        return [list(unique_justseen(r)) for r in routes]

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

        self._start_lane_ids = get_lane_ids(self._start_lane_waypoint, max_distances=(300, 10))
        self._target_lane_ids = get_lane_ids(self._target_lane_waypoint, max_distances=(300, 10))
        common_lanes = set(self._start_lane_ids) & set(self._target_lane_ids)
        assert not (common_lanes)  # ensure disjoint sets of ids

    def _spawn_env_vehicles(self, n: int):
        blueprints = self._world.get_blueprint_library().filter('vehicle.*')
        blueprints = [b for b in blueprints if int(b.get_attribute('number_of_wheels')) == 4]
        spawn_points = self._world_map.get_spawn_points()
        max_ticks = 4

        def _get_env_vehicles():
            return [
                v for v in self._world.get_actors().filter('vehicle.*')
                if v.attributes.get('role_name') != 'hero' and v.is_alive
            ]

        def _wait_for_env_vehicles(total_number, max_ticks):
            env_vehicles = _get_env_vehicles()
            while len(env_vehicles) < total_number and max_ticks > 0:
                self._world.tick()
                max_ticks -= 1
                env_vehicles = _get_env_vehicles()
            return env_vehicles

        env_vehicles = _get_env_vehicles()
        missing_vehicles = n - len(env_vehicles)
        while missing_vehicles:
            cmds = [
                carla.command.SpawnActor(blueprint, spawn_point)
                for blueprint, spawn_point in zip(random.choices(blueprints, k=missing_vehicles),
                                                  random.sample(spawn_points, k=missing_vehicles))
            ]
            self._client.apply_batch_sync(cmds, do_tick=True)
            env_vehicles = _wait_for_env_vehicles(n, max_ticks)
            missing_vehicles = n - len(env_vehicles)
        return env_vehicles

    def _wrap_with_controllers(self, env_vehicles):
        return {
            str(v.id): TeleportCommandsController(v)
            for v in env_vehicles
        }

    def reset(self, ego_vehicle: carla.Vehicle):
        if self._early_stop_monitor:
            self._early_stop_monitor.close()

        timeout_s = (FRAMES_BEFORE_MANUVEUR + FRAMES_AFTER_MANUVEUR) * DT
        self._early_stop_monitor = EarlyStopMonitor(ego_vehicle, timeout_s=timeout_s)

        self._route = random.choice(self._routes)

        min_speed, max_speed = self._speed_range_mps
        range_size = max_speed - min_speed
        speed_mps = min_speed + np.random.random() * range_size

        column_ahead_of_ego_m = np.random.randint(*self._env_vehicle_column_ahead_range_m)
        cmds = self._setup_controllers(self._start_lane_waypoint.transform.location, speed_mps,
                                       self._route, column_ahead_of_ego_m)

        ego_speed_mps = min_speed + np.random.random() * range_size
        t = Transform.from_carla_transform(self._start_lane_waypoint.transform)
        velocity = (t.orientation * ego_speed_mps).to_vector3(0).as_carla_vector3d()
        cmds.append(carla.command.ApplyTransform(ego_vehicle.id, transform=self._start_lane_waypoint.transform))
        cmds.append(carla.command.ApplyVelocity(ego_vehicle.id, velocity))
        if cmds:
            self._client.apply_batch_sync(cmds, do_tick=False)

        self._lane_alignment_monitor.reset()
        self._progress_monitor = LaneChangeProgressMonitor(self._world_map,
                                                           start_lane_ids=self._start_lane_ids,
                                                           target_lane_ids=self._target_lane_ids,
                                                           lane_change_command=self._cmd_for_changing_lane)

    def step(self, ego_vehicle: carla.Vehicle) -> ScenarioStepResult:
        ego_transform = ego_vehicle.get_transform()
        self._move_env_vehicles(ego_transform.location)

        waypoint = self._world_map.get_waypoint(ego_transform.location)

        on_start_lane = False
        on_target_lane = False

        if waypoint:
            lane_id = get_lane_id(waypoint)
            on_start_lane = lane_id in self._start_lane_ids
            on_target_lane = lane_id in self._target_lane_ids

        not_on_expected_lanes = not (on_start_lane or on_target_lane)
        chauffeur_command = self._cmd_for_changing_lane if on_start_lane else ChauffeurCommand.LANE_FOLLOW

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
            done_info[DONE_CAUSE_KEY] = EarlyStop(early_stop).decomposed_name('_').lower()

        info = {
            'reward_type': self._reward_type.name,
            'on_start_lane': on_start_lane,
            'on_target_lane': on_target_lane,
            'is_junction': waypoint.is_junction if waypoint else False,
            **self._lane_alignment_monitor.info(),
            **done_info
        }

        return ScenarioStepResult(chauffeur_command, reward, done, info)

    def _setup_controllers(self, ego_vehicle_location, speed_mps, route, column_ahead_of_ego_m):
        resolution_m = np.median([t1.location.distance(t2.location) for t1, t2 in windowed(route, 2)])
        m2idx = 1 / resolution_m

        ego_vehicle_idx = int(np.argmin([t.location.distance(ego_vehicle_location) for t in route]))
        column_start_idx = ego_vehicle_idx + int(column_ahead_of_ego_m * m2idx)

        ncontrollers = len(self._controllers)
        current_idx = column_start_idx
        cmds = []
        for idx, controller in enumerate(self._controllers.values()):
            initial_location = route[current_idx].location
            cmds.extend(controller.reset(speed_mps=speed_mps, route=route, initial_location=initial_location))
            controllers_left = ncontrollers - idx
            offset = _calc_offset(current_idx, controllers_left, resolution_m)
            current_idx = current_idx - offset
            assert current_idx >= 0

        return cmds

    def _move_env_vehicles(self, ego_vehicle_location: carla.Location):
        column_end_location = None
        column_end_idx = None
        route_resolution_m = None

        cmds = []
        for controller_idx, controller in enumerate(self._controllers.values()):
            finished, cmds_ = controller.step()
            if not finished and _is_behind_ego_or_inside_birdview(controller, ego_vehicle_location):
                cmds.extend(cmds_)
            else:
                if column_end_location is None:
                    # obtain location of last vehicle in column
                    idxes = [c.idx for c in self._controllers.values()]
                    locations = [c.location for c in self._controllers.values()]
                    column_end_location = locations[int(np.argmin(idxes))]

                    # resolution of not resampled route
                    route_resolution_m = \
                        np.median([t1.location.distance(t2.location) for t1, t2 in windowed(self._route, 2)])

                    # idx of nearest point on not resamples route
                    column_end_idx = np.argmin([t.location.distance(column_end_location) for t in self._route])

                # obtain reset location
                offset = _calc_offset(column_end_idx, 1, route_resolution_m)
                column_end_idx = column_end_idx - offset
                assert column_end_idx >= 0
                column_end_location = self._route[column_end_idx].location

                cmds.extend(controller.reset(initial_location=column_end_location))
        if cmds:
            self._client.apply_batch_sync(cmds, do_tick=False)

    def close(self):
        if self._early_stop_monitor:
            self._early_stop_monitor.close()
            self._early_stop_monitor = None
