from enum import Enum
from typing import NamedTuple, Dict

import carla


class ChauffeurCommand(Enum):
    LANE_FOLLOW = 0
    GO_STRAIGHT = 1
    TURN_RIGHT = 2
    TURN_LEFT = 3
    CHANGE_LANE_LEFT = 4
    CHANGE_LANE_RIGHT = 5


class ScenarioStepResult(NamedTuple):
    chauffeur_cmd: ChauffeurCommand
    reward: float
    done: bool
    info: Dict


class Scenario:

    def __init__(self, client: carla.Client):
        self._client = client
        self._world = client.get_world()
        self._world_map = self._world.get_map()

    def reset(self, ego_vehicle: carla.Vehicle):
        """Set up scenarios. Move ego_vehicle to required positions. Spawn other actors if necessary"""
        raise Exception("Not implemented")

    def step(self, ego_vehicle: carla.Vehicle) -> ScenarioStepResult:
        """Manage other actors. Calculate rewards. Check against done condition"""
        raise Exception("Not implemented")
