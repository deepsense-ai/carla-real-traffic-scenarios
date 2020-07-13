from typing import Tuple, List

import carla
import more_itertools
import numpy as np
import scipy.spatial


class RewardCalculator:

    def __call__(self, transform_carla: carla.Transform):
        raise NotImplementedError()


class DenseRewardCalculator(RewardCalculator):

    def __init__(self, trajectory_carla: List[carla.Transform], *,
                 num_waypoints: int = 10, max_distance_trajectory_m: float = 3) -> None:
        trajectory_steps_length_m = [
            t1.location.distance(t2.location) for t1, t2 in more_itertools.windowed(trajectory_carla, 2)
        ]
        trajectory_length_m = sum(trajectory_steps_length_m)
        waypoint_distances_m = trajectory_length_m / num_waypoints

        cum_distance_m = 0
        self._waypoint_idxes = []
        for idx, step_length_m in enumerate(trajectory_steps_length_m):
            cum_distance_m += step_length_m
            if cum_distance_m >= waypoint_distances_m * (len(self._waypoint_idxes) + 1):
                self._waypoint_idxes.append(idx)

            # lets last checkpoint be on last trajectory waypoint
            if len(self._waypoint_idxes) >= num_waypoints - 1:
                self._waypoint_idxes.append(len(trajectory_carla) - 1)
                break

        self._waypoint_idxes = np.array(self._waypoint_idxes, dtype='int')
        self._finish_at_idx = len(trajectory_carla) - 1
        self._completed_waypoints = -1
        self._reward_quant = 1 / num_waypoints
        self._max_distance_trajectory_m = max_distance_trajectory_m

        # for faster nearest trajectory point calculation
        self._trajectory_carla = np.array([[t.location.x, t.location.y] for t in trajectory_carla])

    def __call__(self, transform_carla: carla.Transform):
        idx, min_distance_from_trajectory_m = self._find_nearest_trajectory_point(transform_carla)
        trajectory_finished = idx >= self._finish_at_idx
        moved_away_too_far_from_trajectory = min_distance_from_trajectory_m > self._max_distance_trajectory_m
        done = trajectory_finished or moved_away_too_far_from_trajectory

        completed_waypoints = np.where(self._waypoint_idxes <= idx)[0]
        current_completed_waypoint = completed_waypoints[-1] if len(completed_waypoints) else -1
        reward = self._reward_quant * max(current_completed_waypoint - self._completed_waypoints, 0)
        early_stop = moved_away_too_far_from_trajectory

        self._completed_waypoints = max(current_completed_waypoint, self._completed_waypoints)

        return reward, done, early_stop

    def _find_nearest_trajectory_point(self, transform_carla: carla.Transform) -> Tuple[int, float]:
        transform_carla = np.array([transform_carla.location.x, transform_carla.location.y])
        dm = scipy.spatial.distance_matrix([transform_carla], self._trajectory_carla)
        idx = int(np.argmin(dm, axis=1)[0])
        return idx, dm[0][idx]


class SparseRewardCalculator(DenseRewardCalculator):

    def __call__(self, transform_carla: carla.Transform):
        idx, min_distance_from_trajectory_m = self._find_nearest_trajectory_point(transform_carla)
        trajectory_finished = idx >= self._finish_at_idx
        moved_away_too_far_from_trajectory = min_distance_from_trajectory_m > self._max_distance_trajectory_m
        reward = float(trajectory_finished)
        early_stop = moved_away_too_far_from_trajectory
        return reward, trajectory_finished or moved_away_too_far_from_trajectory, early_stop
