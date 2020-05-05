import logging
from typing import List, Optional

import numpy as np

import carla
from carla_real_traffic_scenarios import FPS
from carla_real_traffic_scenarios.utils.collections import remove_succesive_duplicates
from carla_real_traffic_scenarios.utils.transforms import Transform, resample_points, positions_to_transforms

LOGGER = logging.getLogger(__name__)


class TeleportCommandsController:

    def __init__(self, vehicle: carla.Vehicle, step_length_s: float = 1 / FPS):
        self._actor_id = vehicle.id
        self._step_length_s = step_length_s

        self._idx = 0
        self._speed_mps = 6
        self._resampled_route: List[Transform] = []

    def reset(self, *, speed_mps: Optional[float] = None,
              route: Optional[List[carla.Transform]] = None,
              initial_location: Optional[carla.Location] = None):

        if speed_mps is not None:
            self._speed_mps = speed_mps

        if route is not None:
            step_m = self._speed_mps * self._step_length_s
            self._resampled_route: List[Transform] = self._resample_route(route, step_m=step_m)

        initial_idx = 0
        if initial_location:
            distances = [t.as_carla_transform().location.distance(initial_location) for t in self._resampled_route]
            initial_idx = np.argmin(distances)
        transform = self._resampled_route[initial_idx]
        cmds = self._get_commands(transform)
        self._idx = initial_idx + 1
        return cmds

    def step(self):
        transform = self._resampled_route[self._idx]
        cmds = self._get_commands(transform)
        self._idx += 1
        not_finished = self._idx < len(self._resampled_route)
        return not not_finished, cmds

    def _get_commands(self, transform: Transform):
        velocity = (transform.orientation * self._speed_mps).to_vector3(0).as_carla_vector3d()
        cmds = [
            carla.command.ApplyVelocity(self._actor_id, velocity),
            carla.command.ApplyTransform(self._actor_id, transform=transform.as_carla_transform()),
        ]
        return cmds

    def _resample_route(self, route: List[carla.Transform], step_m: float) -> List[carla.Transform]:
        assert len(route) > 1
        try:
            positions = [Transform.from_carla_transform(transform).position for transform in route]
            positions = remove_succesive_duplicates(positions)
            positions = resample_points(positions, step_m=step_m)
            route = positions_to_transforms(positions)
            return route
        except Exception:
            LOGGER.error(f'#route={len(route)} route={route}')
            raise

    @property
    def actor_id(self):
        return self._actor_id