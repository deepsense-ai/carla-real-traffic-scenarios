import math
from typing import NamedTuple

from dataclasses import dataclass

import carla

def distance(a: carla.Location, b: carla.Location):
    dx = a.x - b.x
    dy = a.y - b.y
    return math.sqrt(dx * dx + dy * dy)

class CircleArea(NamedTuple):
    location: carla.Location
    radius: float

    def __contains__(self, loc: carla.Location) -> bool:
        dist = distance(loc, self.location)
        # print("Distance to next area:", dist)
        return dist <= self.radius


@dataclass
class RoundaboutNode:
    name: str
    spawn_point: carla.Location
    entrance_area: CircleArea
    next_exit: CircleArea
    final_area_for_next_exit: CircleArea
    next_node: "RoundaboutNode"
