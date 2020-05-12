import carla
from dataclasses import dataclass
from typing import NamedTuple
from carla_real_traffic_scenarios.scenario import ChauffeurCommand


class CircleArea(NamedTuple):
    center: carla.Location
    radius: float

    def __contains__(self, loc: carla.Location) -> bool:
        return loc.distance(self.center) <= self.radius


@dataclass
class RoundaboutNode:
    name: str
    spawn_point: carla.Location
    entrance_area: CircleArea
    next_exit: CircleArea
    final_area_for_next_exit: CircleArea
    next_node: "RoundaboutNode"


class RouteCheckpoint(NamedTuple):
    name: str
    area: CircleArea
    command: ChauffeurCommand
