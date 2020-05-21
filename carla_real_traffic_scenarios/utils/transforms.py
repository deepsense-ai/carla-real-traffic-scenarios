#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from typing import NamedTuple, Union, List

import carla
import numpy as np
from more_itertools import windowed, unique_justseen
from scipy import interpolate


class Vector3(NamedTuple):
    x: float
    y: float
    z: float

    def as_tuple(self):
        return self.x, self.y, self.z

    def as_tuple_2d(self):
        return self.x, self.y

    def __add__(self, other) -> 'Vector3':
        if isinstance(other, Vector3):
            return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
        elif isinstance(other, Vector2):
            return Vector3(self.x + other.x, self.y + other.y, self.z)
        else:
            raise ValueError(f"Unsupported {other}")

    def __sub__(self, other) -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: float) -> 'Vector3':
        return Vector3(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other):
        return Vector3(self.x / other, self.y / other, self.z / other)

    def as_carla_location(self):
        import carla
        return carla.Location(float(self.x), float(self.y), float(self.z))

    @staticmethod
    def from_carla_location(location):
        return Vector3(location.x, location.y, location.z)

    @staticmethod
    def from_dict(d):
        return Vector3(d['x'], d['y'], d['z'])

    @staticmethod
    def from_numpy(ar):
        return Vector3(*ar)

    def as_numpy(self):
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def with_z(self, z: float):
        return Vector3(self.x, self.y, z)

    def zero_z(self):
        return Vector3(self.x, self.y, 0)

    def as_vector2(self):
        return Vector2(self.x, self.y)

    def as_carla_vector3d(self):
        import carla
        return carla.Vector3D(self.x, self.y, self.z)


class Vector2(NamedTuple):
    x: float
    y: float

    @staticmethod
    def convert_from(e: Union[Vector3, 'Vector2', 'Transform']):
        if isinstance(e, Vector2):
            return e
        elif isinstance(e, Vector3):
            return e.as_vector2()
        elif isinstance(e, Transform):
            return e.position.as_vector2()
        elif isinstance(e, carla.Location):
            return Vector2(e.x, e.y)
        else:
            raise ValueError(f"Invalid object type {type(e)}")

    def to_vector3(self, z):
        return Vector3(self.x, self.y, z)

    def __mul__(self, other: float) -> 'Vector2':
        return Vector2(self.x * other, self.y * other)

    def __truediv__(self, other) -> 'Vector2':
        return Vector2(self.x / other, self.y / other)

    def __add__(self, other: 'Vector2') -> 'Vector2':
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Vector2') -> 'Vector2':
        return Vector2(self.x - other.x, self.y - other.y)

    def as_tuple(self):
        return self.x, self.y

    def as_carla_rotation(self):
        import carla
        yaw = math.atan2(self.y, self.x) * 180 / math.pi
        return carla.Rotation(yaw=float(yaw), roll=float(0), pitch=float(0))

    @staticmethod
    def from_carla_rotation(rotation):
        yaw_rad = rotation.yaw * math.pi / 180
        x, y = math.cos(yaw_rad), math.sin(yaw_rad)
        assert math.isclose((math.atan2(y, x) * 180 / math.pi) % 360, rotation.yaw % 360, rel_tol=0.00001), \
            f'{math.atan2(y, x) * 180 / math.pi} is not eq {rotation.yaw}'
        return Vector2(x, y)

    @property
    def yaw_radians(self):
        return math.atan2(self.y, self.x)

    @staticmethod
    def from_yaw_radian(radian):
        return Vector2(
            x=np.cos(radian),
            y=np.sin(radian),
        )

    def normalized(self) -> 'Vector2':
        v = self.as_numpy()
        norm = np.linalg.norm(v, ord=1)
        if norm == 0:
            norm = np.finfo(v.dtype).eps
        normalized_v = v / norm
        return Vector2.from_numpy(normalized_v)

    def as_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @staticmethod
    def from_numpy(arr: np.ndarray) -> 'Vector2':
        assert arr.shape == (2,)
        return Vector2(x=arr[0], y=arr[1])

    @staticmethod
    def distace_between(p1: 'Vector2', p2: 'Vector2') -> float:
        return np.linalg.norm(p1.as_numpy() - p2.as_numpy())


class Transform(NamedTuple):
    position: Vector3
    orientation: Vector2

    @staticmethod
    def from_carla_transform(transform) -> 'Transform':
        return Transform(
            Vector3.from_carla_location(transform.location),
            Vector2.from_carla_rotation(transform.rotation)
        )

    def as_carla_transform(self) -> 'carla.Transform':
        import carla
        return carla.Transform(
            location=self.position.as_carla_location(),
            rotation=self.orientation.as_carla_rotation()
        )


def distance_between(p1: Vector3, p2: Vector3):
    return np.linalg.norm(p1.as_numpy() - p2.as_numpy())


def distance_between_on_plane(p1: Union[Transform, Vector2, Vector3], p2: Union[Transform, Vector2, Vector3]):
    p1 = Vector2.convert_from(p1)
    p2 = Vector2.convert_from(p2)
    return np.linalg.norm(p1.as_numpy() - p2.as_numpy())


def convert_to_vector2(p: Union[Vector3, Vector2, Transform, np.ndarray]) -> Vector2:
    if isinstance(p, Transform):
        return p.position.as_vector2()
    elif isinstance(p, Vector3):
        return p.as_vector2()
    elif isinstance(p, np.ndarray):
        return Vector2.from_numpy(p)
    else:
        return p


def resample_points(positions: List[Union[Vector2, Vector3]], step_m=1) -> List[Vector2]:
    """Interpolates points so they are evenly spaced (1 meter between each point)"""
    positions = [convert_to_vector2(p) for p in positions]
    points = np.array([p.as_numpy() for p in positions])

    assert len(list(unique_justseen(positions))) == len(positions)  # breaks resampling code

    x, y = zip(*points)

    DEGREE_ONE_SO_WE_HAVE_POLYLINE = 1
    f, u = interpolate.splprep([x, y], s=0, k=DEGREE_ONE_SO_WE_HAVE_POLYLINE, per=0)

    distance = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)).sum()
    linspace = np.linspace(0, 1, int(distance / step_m))
    x, y = interpolate.splev(linspace, f)

    points_fitted = np.stack([x, y], axis=1)

    return [Vector2.from_numpy(p) for p in points_fitted]


def positions_to_transforms(positions: List[Union[Vector3, Vector2]]) -> List[Transform]:
    """Add orientation data to positions using normals to determine angles"""
    positions = [convert_to_vector2(w) for w in positions]

    assert len(positions) > 1

    guarded_positions = [
        positions[0] - (positions[1] - positions[0]),
        *positions,
        positions[-1] + (positions[-1] - positions[-2]),
    ]
    smooth_orientations = []
    for p1, p2, p3 in windowed(guarded_positions, 3):
        dir_1 = (p2 - p1).normalized()
        dir_2 = (p3 - p2).normalized()
        tangent_dir = (dir_1 + dir_2).normalized()
        smooth_orientations.append(tangent_dir)

    assert len(positions) == len(smooth_orientations), \
        f"Got {len(positions)} and {len(smooth_orientations)}"

    return [Transform(p.to_vector3(0.), o) for (p, o) in zip(positions, smooth_orientations)]
