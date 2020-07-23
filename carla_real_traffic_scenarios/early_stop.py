import logging
from typing import Optional

import carla
from enum import IntFlag, auto, _decompose

from carla_real_traffic_scenarios.trajectory import Trajectory
from carla_real_traffic_scenarios.utils.carla import CollisionSensor

LOGGER = logging.getLogger(__name__)


class EarlyStop(IntFlag):
    NONE = 0
    MOVED_TOO_FAR = auto()
    TIMEOUT = auto()
    OFFROAD = auto()
    COLLISION = auto()
    ERROR = auto()

    def decomposed_name(self, sep='|'):
        return sep.join([e.name for e in _decompose(self.__class__, self)[0]])


class EarlyStopMonitor:

    def __init__(self, ego_vehicle: carla.Vehicle, *, trajectory=None, max_trajectory_distance_m=None, timeout_s=None):
        self._world = ego_vehicle.get_world()
        self._world_map = self._world.get_map()
        self._collision_sensor: CollisionSensor = CollisionSensor(self._world, ego_vehicle)

        self._trajectory: Optional[Trajectory] = trajectory
        self._max_trajectory_distance_m = max_trajectory_distance_m

        self._start_timestamp_s = self._get_timestamp_s(self._world)
        self._timeout_s = timeout_s

    def _get_timestamp_s(self, world):
        snapshot = world.get_snapshot()
        return snapshot.timestamp.elapsed_seconds

    def __call__(self, ego_transform: carla.Transform):
        early_stop = EarlyStop.NONE
        for check_fn in [self._check_collision, self._check_offroad, self._check_timeout, self._check_move_away]:
            early_stop |= check_fn(ego_transform)

        return early_stop

    def _check_move_away(self, ego_transform):
        move_away = EarlyStop.NONE
        if self._trajectory:
            *_, distance_m = self._trajectory.find_nearest_trajectory_point(ego_transform)
            if distance_m > self._max_trajectory_distance_m:
                move_away = EarlyStop.MOVED_TOO_FAR
                LOGGER.debug(f'Vehicle moved too far: {distance_m:0.3f}m/{self._max_trajectory_distance_m:0.3f}m')
        return move_away

    def _check_timeout(self, *_):
        timeout = EarlyStop.NONE
        elapsed_s = self._get_timestamp_s(self._world) - self._start_timestamp_s
        if self._timeout_s is not None and elapsed_s > self._timeout_s:  # timeout
            timeout = EarlyStop.TIMEOUT
            LOGGER.debug(f'Timeout elapsed: {elapsed_s:0.3f}s while timeout is set to {self._timeout_s:0.3f}s')
        return timeout

    def _check_offroad(self, ego_transform):
        offroad = EarlyStop.NONE
        if self._world_map.get_waypoint(ego_transform.location, project_to_road=False) is None:
            waypoint = self._world_map.get_waypoint(ego_transform.location)
            distance_to_lane = waypoint.transform.location.distance(ego_transform.location)
            # if distance to nearest lane ceneter not exceeds 30% of half of lane width
            probably_is_on_lane = (waypoint.lane_width / 2) * 1.3 > distance_to_lane
            if not probably_is_on_lane:
                offroad = EarlyStop.OFFROAD
                LOGGER.debug(f'Vehicle off-road')
        return offroad

    def _check_collision(self, *_):
        collided = EarlyStop.NONE
        if self._collision_sensor.has_collided:
            collided = EarlyStop.COLLISION
            LOGGER.debug(f'Vehicle collision')
        return collided

    def close(self):
        if self._collision_sensor:
            self._collision_sensor.destroy()
            self._collision_sensor = None
