import carla
from dataclasses import dataclass
from typing import NamedTuple

from libs.carla_real_traffic_scenarios.carla_real_traffic_scenarios.scenario import ChauffeurCommand
from libs.carla_real_traffic_scenarios.carla_real_traffic_scenarios.utils import (
    geometry,
)


class CircleArea(NamedTuple):
    location: carla.Location
    radius: float

    def __contains__(self, loc: carla.Location) -> bool:
        dist = geometry.distance(loc, self.location)
        return dist <= self.radius


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

    # FIXME Could be a method of CircleArea instead...
    def draw(self, world: carla.World, **kwargs):
        world.debug.draw_point(self.area.location, **kwargs)
        xs, ys = geometry.points_on_ring(radius=self.area.radius, num_points=10)
        for x, y in zip(xs, ys):
            center = carla.Location(
                x=self.area.location.x + x, y=self.area.location.y + y, z=0.1
            )
            world.debug.draw_point(center, **kwargs)
