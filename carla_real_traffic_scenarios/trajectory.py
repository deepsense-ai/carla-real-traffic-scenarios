from typing import List, Tuple

import carla
import more_itertools
import numpy as np
import scipy.spatial
from functools import lru_cache


@lru_cache()
def _get_nearest_location(locations_carla: np.ndarray, location_ego: np.ndarray):
    dm = scipy.spatial.distance_matrix([location_ego], locations_carla)
    idx = int(np.argmin(dm, axis=1)[0])
    return idx, dm[0][idx]


class Trajectory:

    def __init__(self, trajectory_carla: List[carla.Transform]):
        self._trajectory_carla = trajectory_carla
        self._locations_carla = np.array([[t.location.x, t.location.y] for t in trajectory_carla])

        self._segments_length_m = [
            t1.location.distance(t2.location) for t1, t2 in more_itertools.windowed(trajectory_carla, 2)
        ]
        self._length_m = sum(self._segments_length_m)

    def find_nearest_trajectory_point(self, transform_carla: carla.Transform) -> Tuple[int, carla.Transform, float]:
        location_ego = np.round([transform_carla.location.x, transform_carla.location.y], decimals=1)
        idx, distance_from_trajectory = _get_nearest_location(self._locations_carla, location_ego)
        return idx, self._trajectory_carla[idx], distance_from_trajectory

    def get_distance_from_start(self, transform_carla: carla.Transform):
        idx, *_, distance_from_trajectory = self.find_nearest_trajectory_point(transform_carla)

    @property
    def total_length_m(self):
        return self._length_m

    def __call__(self, transform_carla: carla.Transform) -> Tuple[bool, float]:
        pass
