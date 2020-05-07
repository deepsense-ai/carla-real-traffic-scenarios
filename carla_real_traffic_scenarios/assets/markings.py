import carla
from typing import List, NamedTuple
from pathlib import Path
from carla_real_traffic_scenarios.assets import utils
import logging

Regex = str

log = logging.getLogger(__name__)


class Marking(NamedTuple):
    """Represents spawn markings for static assets from Carla's blueprint library.
    All blueprints: https://carla.readthedocs.io/en/latest/bp_library/
    """

    id: int
    transform: carla.Transform
    blueprint_patterns: List[Regex]
    yaw_agnostic: bool

    def draw(self, world: carla.World):
        if self.yaw_agnostic:
            end = utils.clone_location(self.transform.location)
            end.z += 1.5
            world.debug.draw_arrow(
                begin=self.transform.location,
                end=end,
                thickness=0.2,
                arrow_size=2,
                life_time=0,
                color=carla.Color(r=20, g=60, b=160),
            )
        else:
            world.debug.draw_arrow(
                begin=self.transform.location,
                end=self.transform.location + self.transform.get_forward_vector(),
                thickness=0.1,
                arrow_size=0.1,
                life_time=0,
                color=carla.Color(r=20, g=200, b=100),
            )
        world.debug.draw_string(
            self.transform.location, text=str(self.id), color=carla.Color(255, 255, 255), life_time=0
        )

    def serialize(self) -> dict:
        """JSON representation of self."""
        loc = self.transform.location
        rot = self.transform.rotation
        return {
            "id": self.id,
            "location": {"x": loc.x, "y": loc.y, "z": loc.z},
            "rotation": {"pitch": rot.pitch, "yaw": rot.yaw, "roll": rot.roll},
            "yaw_agnostic": self.yaw_agnostic,
            "blueprint_patterns": self.blueprint_patterns,
        }

    @staticmethod
    def from_serialized(data: dict) -> "Marking":
        """Make an instance of `Marking` class from serialized dict."""
        return Marking(
            id=data["id"],
            transform=carla.Transform(
                carla.Location(x=data["location"]["x"], y=data["location"]["y"], z=data["location"]["z"]),
                carla.Rotation(
                    pitch=data["rotation"]["pitch"], yaw=data["rotation"]["yaw"], roll=data["rotation"]["roll"]
                ),
            ),
            blueprint_patterns=data["blueprint_patterns"],
            yaw_agnostic=data["yaw_agnostic"],
        )


def serialize_to_json_file(markings: List[Marking], path: Path):
    """Serialize markings into JSON file."""
    markings = [m.serialize() for m in markings]
    utils.export_json(data=markings, path=path)
    log.info("Markings have been successfully exported.")


def deserialize_json_file(path: Path) -> List[Marking]:
    """Deserialize markings from JSON file."""
    data = utils.import_json(path)
    if not type(data) is list:
        raise TypeError("Current format requires a list of markings! See `Marking.from_serialized`")
    log.info("Markings have been successfully loaded.")
    return [Marking.from_serialized(marking_data) for marking_data in data]
