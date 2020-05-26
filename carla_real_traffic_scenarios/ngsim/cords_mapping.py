import cv2.cv2 as cv2
import numpy as np

from carla_real_traffic_scenarios.ngsim import NGSimDatasets
from carla_real_traffic_scenarios.utils.transforms import Vector2, Transform

LANE_WIDTH_METERS = 3.7
LANE_WIDTH_PIXELS = 24  # pixels / 3.7 m, lane width
METER_TO_PIXELS = LANE_WIDTH_PIXELS / LANE_WIDTH_METERS
PIXELS_TO_METERS = 1. / METER_TO_PIXELS
FOOT_TO_METERS = 0.3048  # metres per foot


class NGSimToCarlaMapper:

    def __init__(self, ngsim_origin: np.ndarray, carla_origin: np.ndarray):
        assert ngsim_origin.shape == (2,), "Expected 2d point"
        assert carla_origin.shape == (2,), "Expected 2d point"

        ngsim_base_vectors = np.array([[1, 0], [0, 1]])
        scale = PIXELS_TO_METERS
        carla_base_vectors = np.array([[1, 0], [0, 1]]) * scale

        self._transformation_matrix = cv2.getAffineTransform(
            src=np.float32([ngsim_origin, *(ngsim_origin + ngsim_base_vectors)]),
            dst=np.float32([carla_origin, *(carla_origin + carla_base_vectors)]),
        )
        NO_TRANSLATION = 0
        self._rotation = Vector2.from_numpy(
            self._transformation_matrix @ np.array([1, 0, NO_TRANSLATION])
        ).yaw_radians

    def ngsim_to_carla(self, ngsim_transform: Transform, z: float, rear_axle_offset: float) -> Transform:
        p = ngsim_transform.position.as_vector2()
        p = self._transformation_matrix @ np.array([p.x, p.y, 1])
        p = Vector2.from_numpy(p)
        direction = Vector2.from_yaw_radian(ngsim_transform.orientation.yaw_radians - self._rotation)
        p -= direction * rear_axle_offset  # in ngsim origin point is in the center of rear axle
        return Transform(p.to_vector3(z), direction)


# TDO put inside declarative structure?
MAPPER_BY_NGSIM_DATASET = {
    NGSimDatasets.I80: NGSimToCarlaMapper(
        ngsim_origin=np.array([0, 0]),
        carla_origin=np.array([30, -25.7]),
    ),
    NGSimDatasets.US101: NGSimToCarlaMapper(
        ngsim_origin=np.array([0, 0]),
        carla_origin=np.array([-300, -17]),
    )
}
