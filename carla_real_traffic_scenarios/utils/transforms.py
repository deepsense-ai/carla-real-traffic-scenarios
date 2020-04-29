#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import carla
import math
from typing import NamedTuple, Union

import numpy as np


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
    def from_carla_orientation(orientation):
        return Vector2(orientation.x, orientation.y)

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
