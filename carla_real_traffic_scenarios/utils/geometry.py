# -*- coding: utf-8 -*-
from typing import Tuple

import carla
import numpy as np


def jaccard_rectangles(r1_width, r1_height, r2_width, r2_height) -> float:
    r1_area = r1_width * r1_height
    r2_area = r2_width * r2_height
    intersection_area = min(r1_width, r2_width) * min(r1_height, r2_height)
    return intersection_area / (r1_area + r2_area - intersection_area)


def normalize_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi]."""
    return normalize_angle_npy(np.array(angle, dtype=np.float32)).item()


def normalize_angle_npy(angles: np.ndarray) -> np.ndarray:
    """Normalize angles to [-pi, pi]."""
    TWO_PI = np.array([2 * np.pi], dtype=np.float32)
    angles = angles % TWO_PI
    angles[angles > np.pi] -= TWO_PI
    angles[angles < -np.pi] += TWO_PI
    return angles


def points_on_ring(radius: float, num_points: int) -> Tuple[np.array, np.array]:
    """Generates `n` coordinates lying on a ring with radius `r` and center at (0, 0)."""
    t = np.linspace(0, 2 * np.pi, num_points)
    xs = radius * np.cos(t)
    ys = radius * np.sin(t)
    return xs, ys
