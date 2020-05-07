import json
import carla
from pathlib import Path

from typing import Any, List

import logging

log = logging.getLogger(__name__)


def import_json(path: Path) -> Any:
    with path.open() as in_file:
        data = json.load(in_file)
    log.debug(f"Loaded JSON file: {path}")
    return data


def export_json(data: Any, path: Path):
    with path.open("w") as out_file:
        json.dump(data, out_file, indent=4)
    log.debug(f"Saved JSON file: {path}")


def clone_rotation(rot: carla.Rotation) -> carla.Rotation:
    return carla.Rotation(rot.pitch, rot.yaw, rot.roll)


def clone_location(loc: carla.Location) -> carla.Location:
    return carla.Location(loc.x, loc.y, loc.z)


def clone_transform(trans: carla.Transform) -> carla.Transform:
    """Transform, Location, Rotation from carla module does not support
    serialization out-of-the-box yet, hence deepcopy does not work.
    """
    return carla.Transform(
        clone_location(trans.location), clone_rotation(trans.rotation)
    )
