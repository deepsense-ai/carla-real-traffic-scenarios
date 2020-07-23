import logging
from typing import List, Tuple

import carla
import more_itertools
import numpy as np
import scipy.spatial

from carla_real_traffic_scenarios.scenario import ChauffeurCommand
from carla_real_traffic_scenarios.utils.geometry import normalize_angle
from carla_real_traffic_scenarios.utils.topology import get_lane_id
from carla_real_traffic_scenarios.utils.transforms import distance_between_on_plane

LOGGER = logging.getLogger(__name__)


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

        self._s = np.cumsum(np.pad(self._segments_length_m, 1, mode='constant'))
        last_idx = len(trajectory_carla) - 1
        self._length_m = self._s[last_idx]

    def find_nearest_trajectory_point(self, transform_carla: carla.Transform) -> Tuple[int, carla.Transform, float]:
        location_ego = np.array([transform_carla.location.x, transform_carla.location.y])
        idx, distance_from_trajectory = _get_nearest_location(self._locations_carla, location_ego)

        return self._s[idx], self._trajectory_carla[idx], distance_from_trajectory

    @property
    def total_length_m(self):
        return self._length_m


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
                                   np.abs(yaw_rad_error) < self._yaw_rad_error_tolerance
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


class LaneChangeProgressMonitor:

    def __init__(self, world_map: carla.Map, *,
                 start_lane_ids: List[Tuple[int, int, int]], target_lane_ids: List[Tuple[int, int, int]],
                 lane_change_command: ChauffeurCommand, checkpoints_number: int = 10):
        self._world_map = world_map
        self._start_lane_ids = start_lane_ids
        self._target_lane_ids = target_lane_ids
        self._lane_change_command = lane_change_command
        self._checkpoints_number = checkpoints_number

        self._total_distance_m = None
        self._checkpoints_distance_m = None
        self._previous_progress = 0

    def get_progress_change(self, ego_transform: carla.Transform):
        current_location = ego_transform.location
        current_waypoint = self._world_map.get_waypoint(current_location)
        lane_id = get_lane_id(current_waypoint)
        on_start_lane = lane_id in self._start_lane_ids
        on_target_lane = lane_id in self._target_lane_ids

        # on init calculate "lane width"
        if self._total_distance_m is None:
            target_waypoint = self._get_target_lane_waypoint(current_waypoint)
            self._total_distance_m = current_location.distance(target_waypoint.transform.location)
            self._checkpoints_distance_m = self._total_distance_m / self._checkpoints_number

        progress_change = 0
        if on_start_lane:
            target_waypoint = self._get_target_lane_waypoint(current_waypoint)
            if target_waypoint:
                progress_change = self._calc_progress_change(target_waypoint, current_location)
            else:
                LOGGER.info('Have no lane to perform maneuver')
        elif on_target_lane:
            target_waypoint = current_waypoint
            progress_change = self._calc_progress_change(target_waypoint, current_location)

        return progress_change

    def _get_target_lane_waypoint(self, current_waypoint):
        return {
            ChauffeurCommand.CHANGE_LANE_LEFT: current_waypoint.get_left_lane,
            ChauffeurCommand.CHANGE_LANE_RIGHT: current_waypoint.get_right_lane,
        }[self._lane_change_command]()

    def _calc_progress_change(self, target_waypoint, current_location):
        distance_from_target = current_location.distance(target_waypoint.transform.location)
        distance_traveled_m = self._total_distance_m - distance_from_target
        checkpoints_passed_number = int(distance_traveled_m / self._checkpoints_distance_m)
        progress = checkpoints_passed_number / self._checkpoints_number
        progress_change = progress - self._previous_progress
        self._previous_progress = progress
        return progress_change
