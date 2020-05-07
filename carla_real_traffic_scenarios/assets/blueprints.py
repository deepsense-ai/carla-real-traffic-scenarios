import carla
import numpy as np


def randomize_attributes(bp: carla.ActorBlueprint) -> None:
    """Works in-place.

    Attributes for all blueprints: https://github.com/carla-simulator/carla/blob/7f91986f6957260b33b939e1e066a9094e272087/Docs/bp_library.md
    There are more attributes, but most of them is read-only... (number_of_wheels, age, gender, size)
    """

    # Applies to vehicles only
    if bp.has_attribute("color"):
        rgb = np.random.randint(0, 255, size=3, dtype=int)
        value = ",".join(map(str, rgb))
        bp.set_attribute("color", value)

    # Taken from carla docs
    # for attr in bp:
    #     if attr.is_modifiable and attr:
    #         bp.set_attribute(attr.id, random.choice(attr.recommended_values))
