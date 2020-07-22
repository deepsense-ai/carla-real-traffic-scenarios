from typing import List, Tuple

import carla
import more_itertools
import numpy as np
import scipy.spatial
from functools import lru_cache

from carla_real_traffic_scenarios.utils.geometry import normalize_angle
from carla_real_traffic_scenarios.utils.transforms import distance_between_on_plane


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

CROSSTRACK_ERROR_TOLERANCE = 0.3
YAW_DEG_ERRORS_TOLERANCE = 10
TARGET_LANE_ALIGNMENT_FRAMES = 10


class LaneAlignmentMonitor:

    def __init__(self, lane_alignment_frames, cross_track_error_tolerance, yaw_deg_error_tolerance):
        self._cross_track_error_tolerance = cross_track_error_tolerance
        self._yaw_rad_error_tolerance = np.deg2rad(yaw_deg_error_tolerance)
        self._lane_alignment_frames = lane_alignment_frames
        self._lane_alignment_counter = 0
        self._last_cross_track_error = 0
        self._last_yaw_rad_error = 0

    def reset(self):
        self._lane_alignment_counter = 0
        self._last_cross_track_error = 0
        self._last_yaw_rad_error = 0

    def is_lane_aligned(self, ego_transform: carla.Transform, lane_transform: carla.Transform):
        cross_track_error = distance_between_on_plane(ego_transform.location, lane_transform.location)
        yaw_rad_error = normalize_angle(np.deg2rad(ego_transform.rotation.yaw - lane_transform.rotation.yaw))
        aligned_with_target_lane = cross_track_error < self._cross_track_error_tolerance and \
                                   yaw_rad_error < self._yaw_rad_error_tolerance
        if aligned_with_target_lane:
            self._lane_alignment_counter += 1
        else:
            self._lane_alignment_counter = 0
        self._last_cross_track_error = cross_track_error
        self._last_yaw_rad_error = yaw_rad_error
        return self._lane_alignment_counter == self._lane_alignment_frames

    def info(self):
        alignment_errors = str(f'track={self._last_cross_track_error:0.2f}m yaw={self._last_yaw_rad_error:0.2f}rad')
        return {
            'alignment_errors': alignment_errors,
            'target_alignment_counter': self._lane_alignment_counter
        }
