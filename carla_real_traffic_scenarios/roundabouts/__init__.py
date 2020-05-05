import itertools
import math
import random
from enum import Enum, auto
from typing import NamedTuple

from dataclasses import dataclass

import carla
from libs.carla_real_traffic_scenarios.carla_real_traffic_scenarios.roundabouts.controller import (
    VehiclePIDController,
)
from libs.carla_real_traffic_scenarios.carla_real_traffic_scenarios.scenario import (
    ScenarioStepResult,
)
from sim2real.carla import DT, ChauffeurCommand
import numpy as np


def distance(a: carla.Location, b: carla.Location):
    dx = a.x - b.x
    dy = a.y - b.y
    return math.sqrt(dx * dx + dy * dy)


def _compute_connection(current_waypoint, next_waypoint):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
             RoadOption.STRAIGHT
             RoadOption.LEFT
             RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < 1.0:
        return ChauffeurCommand.GO_STRAIGHT
    elif diff_angle > 90.0:
        return ChauffeurCommand.TURN_LEFT
    else:
        return ChauffeurCommand.TURN_RIGHT


def angle(target_location, current_location, orientation):
    """
    Compute relative angle and distance between a target_location and a current_location

    :param target_location: location of the target object
    :param current_location: location of the reference object
    :param orientation: orientation of the reference object
    :return: a tuple composed by the distance to the object and the angle between both objects
    """
    target_vector = np.array(
        [target_location.x - current_location.x, target_location.y - current_location.y]
    )
    norm_target = np.linalg.norm(target_vector)

    forward_vector = np.array(
        [math.cos(math.radians(orientation)), math.sin(math.radians(orientation))]
    )
    d_angle = math.degrees(
        math.acos(np.dot(forward_vector, target_vector) / norm_target)
    )

    return d_angle


# # czerwone (od  gornego
# Putting mark at: Location(x=7.635864, y=-28.125000, z=0.500000)
# Latest marking id: 2
# Putting mark at: Location(x=-29.239136, y=-6.250000, z=0.500000)
# Latest marking id: 3
# Putting mark at: Location(x=-9.739136, y=29.000000, z=0.500000)
# Latest marking id: 4
# Putting mark at: Location(x=27.760864, y=6.000000, z=0.500000)
# Latest marking id: 1
#
# # zielone (dolny)
# Putting mark at: Location(x=-21.922626, y=1.375000, z=0.500000)
# Latest marking id: 5
# Putting mark at: Location(x=2.827374, y=22.375000, z=0.500000)
# Latest marking id: 6
# Putting mark at: Location(x=20.827374, y=-5.125000, z=0.500000)
# Latest marking id: 7
# Putting mark at: Location(x=-3.922626, y=-20.875000, z=0.500000)
# Latest marking id: 8
#
# # fioletowe od gornego
# Putting mark at: Location(x=6.077374, y=-62.000000, z=0.500000)
# Latest marking id: 10
# Putting mark at: Location(x=-51.834660, y=-2.930555, z=0.500000)
# Latest marking id: 11
# Putting mark at: Location(x=-8.239713, y=60.998199, z=0.500000)
# Latest marking id: 12
# Putting mark at: Location(x=54.827374, y=5.625000, z=0.500000)
# Latest marking id: 9


class CircleArea(NamedTuple):
    location: carla.Location
    radius: float

    def __contains__(self, loc: carla.Location) -> bool:
        dist = distance(loc, self.location)
        print("Distance to next area:", dist)
        return dist <= self.radius


class RoundaboutExit(NamedTuple):
    # id: int
    exit_area: CircleArea
    target_area: CircleArea


class RoundaboutEntrance(NamedTuple):
    # id: int
    spawn_point: carla.Transform
    enter_area: CircleArea
    next_exit_idx: int
    # next_entrance_id: int


class Area(Enum):
    WITHIN = auto()
    LEAVED = auto()


class ManeuverStatus(Enum):
    SPAWNED = auto()
    REACHED_ROUNDABOUT_ENTRANCE = auto()
    SHOULD_STAY_WITHIN_ROUNDABOUT = auto()
    SHOULD_TAKE_NEXT_EXIT = auto()
    REACHED_EXIT_AREA = auto()
    REACHED_FINAL_AREA = auto()


@dataclass
class RoundaboutNode:
    name: str
    spawn_point: carla.Location
    entrance_area: CircleArea
    next_exit: CircleArea
    final_area_for_next_exit: CircleArea
    next_node: "RoundaboutNode"


node3 = RoundaboutNode(
    name="Node3 - right",
    spawn_point=carla.Transform(
        # SP3
        carla.Location(x=4.894153594970703, y=61.459991455078125, z=0.5),
        carla.Rotation(yaw=270),
    ),
    # do poprawki entrance
    entrance_area=CircleArea(
        # E3
        location=carla.Location(x=5.050692, y=21.908991, z=0.500000),
        radius=5,
    ),
    next_exit=CircleArea(
        # X3
        location=carla.Location(x=27.760864, y=6.000000, z=0.500000),
        radius=3,
    ),
    final_area_for_next_exit=CircleArea(
        # F3
        location=carla.Location(x=54.827374, y=5.625000, z=0.500000),
        radius=3,
    ),
    # oba zle
    # next_exit=CircleArea(
    #     # X0
    #     location=carla.Location(x=7.635864, y=-28.125000, z=0.500000),
    #     radius=3,
    # ),
    # final_area_for_next_exit=CircleArea(
    #     # F0
    #     location=carla.Location(x=6.077374, y=-62.000000, z=0.500000),
    #     radius=3,
    # ),
    next_node=None,
)
node2 = RoundaboutNode(
    name="Node2 - bottom",
    spawn_point=carla.Transform(
        # SP2
        carla.Location(x=-60.9058464050293, y=1.0, z=0.5),
        carla.Rotation(yaw=358),
    ),
    entrance_area=CircleArea(
        # E2
        location=carla.Location(x=-21.922626, y=1.375000, z=0.500000),
        radius=5,
    ),
    next_exit=CircleArea(
        # X3
        location=carla.Location(x=-9.886812, y=27.106071, z=0.500000),
        radius=3,
    ),
    final_area_for_next_exit=CircleArea(
        # F3
        location=carla.Location(x=-7.886812, y=69.356071, z=0.500000),
        radius=3,
    ),
    next_node=node3,
)
node1 = RoundaboutNode(
    name="Node1 - left",
    spawn_point=carla.Transform(
        # SP1
        carla.Location(x=-6.230829238891602, y=-67.29000854492188, z=0.5),
        carla.Rotation(yaw=90),
    ),
    entrance_area=CircleArea(
        # E1
        location=carla.Location(x=-3.922626, y=-20.875000, z=0.500000),
        radius=7,
    ),
    next_exit=CircleArea(
        # X1
        location=carla.Location(x=-29.239136, y=-6.250000, z=0.500000),
        radius=3,
    ),
    final_area_for_next_exit=CircleArea(
        # F1
        location=carla.Location(x=-51.834660, y=-2.930555, z=0.500000),
        radius=3,
    ),
    # next_exit=CircleArea(
    #     # X2
    #     location=carla.Location(x=-9.739136, y=29.000000, z=0.500000),
    #     radius=3,
    # ),
    # final_area_for_next_exit=CircleArea(
    #     # F2
    #     location=carla.Location(x=-8.239713, y=60.998199, z=0.500000),
    #     radius=3,
    # ),
    next_node=node2,
)
node0 = RoundaboutNode(
    name="Node0 - top",
    spawn_point=carla.Transform(
        # SP0
        carla.Location(x=35.750507, y=-8.345556, z=0.5),
        carla.Rotation(yaw=180),
    ),
    entrance_area=CircleArea(
        # E0
        location=carla.Location(x=20.827374, y=-5.125000, z=0.500000),
        radius=5,
    ),
    next_exit=CircleArea(
        # X1
        location=carla.Location(x=7.575461, y=-28.649958, z=0.500000),
        radius=3,
    ),
    final_area_for_next_exit=CircleArea(
        # F1
        location=carla.Location(x=7.075461, y=-50.946571, z=0.500000),
        radius=3,
    ),
    next_node=node1,
)
node3.next_node = node0
NODES = [node0, node1, node2, node3]
EXITS = [
    RoundaboutExit(
        # XF0,
        exit_area=CircleArea(
            location=carla.Location(x=7.635864, y=-28.125000, z=0.500000), radius=5
        ),
        target_area=CircleArea(
            location=carla.Location(x=6.077374, y=-62.000000, z=0.500000), radius=2
        ),
    ),
    # RoundaboutExit(
    #     # XF1,
    #     exit_area=carla.Location(x=-29.239136, y=-6.250000, z=0.500000),
    #     target_area=carla.Location(x=-51.834660, y=-2.930555, z=0.500000),
    # ),
    # RoundaboutExit(
    #     # XF2,
    #     exit_area=carla.Location(x=-9.739136, y=29.000000, z=0.500000),
    #     target_area=carla.Location(x=-8.239713, y=60.998199, z=0.500000),
    # ),
    # RoundaboutExit(
    #     # XF3,
    #     exit_area=carla.Location(x=27.760864, y=6.000000, z=0.500000),
    #     target_area=carla.Location(x=54.827374, y=5.625000, z=0.500000),
    # ),
]


ENTRANCES = [
    RoundaboutEntrance(
        # id=0,
        spawn_point=carla.Transform(
            carla.Location(x=35.750507, y=-8.345556, z=0.5), carla.Rotation(yaw=180)
        ),
        enter_area=CircleArea(
            location=carla.Location(x=20.827374, y=-5.125000, z=0.500000), radius=5
        ),
        next_exit_idx=0,
        # next_entrance_id=1
    ),
    RoundaboutEntrance(
        # id=1,
        spawn_point=None,
        enter_area=CircleArea(
            location=carla.Location(x=-3.922626, y=-20.875000, z=0.500000), radius=5
        ),
        next_exit_idx=1,
        # next_entrance_id=2
    ),
    RoundaboutEntrance(
        # id=2,
        spawn_point=None,
        enter_area=CircleArea(
            location=carla.Location(x=-21.922626, y=1.375000, z=0.500000), radius=5
        ),
        next_exit_idx=2,
        # next_entrance_id=3
    ),
    RoundaboutEntrance(
        # id=3,
        spawn_point=None,
        enter_area=CircleArea(
            location=carla.Location(x=2.827374, y=22.375000, z=0.500000), radius=5
        ),
        next_exit_idx=3,
        # next_entrance_id=0
    ),
]
MAX_ROUNDABOUT_LAPS = 2


class RouteCheckpoint(NamedTuple):
    name: str
    area: CircleArea
    command: ChauffeurCommand


class ExitingRoundaboutScenario:
    def __init__(self, world):
        self._world = world
        self._map = world.get_map()

    # def reset(self, agent_vehicle):
    #     self.num_exits_to_miss = 0  # random.randrange(MAX_ROUNDABOUT_LAPS * len(EXITS))
    #     num_entrances_to_pass_by = self.num_exits_to_miss + 1
    #     # self.steps_left_to_reach_area = MAX_NUM_STEPS_TO_REACH_NEXT_AREA
    #
    #     start_entrance_idx = 0  # random.randrange(len(ENTRANCES))
    #     last_entrance_idx = start_entrance_idx + num_entrances_to_pass_by - 1
    #     self.entrances_to_reach = list(
    #         itertools.islice(
    #             itertools.cycle(ENTRANCES), start_entrance_idx, last_entrance_idx + 1
    #         )
    #     )
    #
    #     agent_vehicle.set_transform(self.entrances_to_reach[0].spawn_point)
    #     self.status = ManeuverStatus.SPAWNED
    #     self.entrances_reached = 0

    def reset(self, veh):
        self.take_nth_exit = random.randrange(1, 5)
        self.route = []
        start_node = random.choice(NODES)
        veh.set_transform(start_node.spawn_point)

        current_node = start_node
        for idx in range(self.take_nth_exit - 1):
            current_node = current_node.next_node
            checkpoint = RouteCheckpoint(
                name=f"{idx} entrance {current_node.name}",
                area=current_node.entrance_area,
                command=ChauffeurCommand.LANE_FOLLOW,
            )
            self.route.append(checkpoint)
        last_entrance_checkpoint = RouteCheckpoint(
            name=f"last entrance {current_node.name}",
            area=current_node.entrance_area,
            command=ChauffeurCommand.TURN_RIGHT,
        )
        exit_checkpoint = RouteCheckpoint(
            name=f"exit {current_node.next_node.name}",
            area=current_node.next_exit,
            command=ChauffeurCommand.LANE_FOLLOW,
        )
        final_checkpoint = RouteCheckpoint(
            name=f"final {current_node.next_node.name}",
            area=current_node.final_area_for_next_exit,
            command=ChauffeurCommand.LANE_FOLLOW,
        )
        self.route.append(last_entrance_checkpoint)
        self.route.append(exit_checkpoint)
        self.route.append(final_checkpoint)

        for idx, n in enumerate(NODES):
            c = carla.Color(50 * idx, 10 + 20 * idx, 66 + 5*idx)
            self._world.debug.draw_point(n.spawn_point.location, color=c)
            self._world.debug.draw_point(n.entrance_area.location, color=c)
            self._world.debug.draw_point(n.next_exit.location, color= c)
            self._world.debug.draw_point(n.final_area_for_next_exit.location, color=c)
        self.next_route_checkpoint_idx = 0
        self.command = ChauffeurCommand.LANE_FOLLOW

    def step(self, veh):
        trans = veh.get_transform()
        loc = trans.location
        next_checkpoint = self.route[self.next_route_checkpoint_idx]
        # self._world.debug.draw_point(next_checkpoint.area.location, life_time=0.1)
        checkpoint_area = next_checkpoint.area

        reward = 0
        done = False
        info = {}

        if loc in checkpoint_area:
            self.command = next_checkpoint.command
            print(
                f"Within checkpoint: {self.route[self.next_route_checkpoint_idx].name}"
            )
            self.next_route_checkpoint_idx += 1
        if self.next_route_checkpoint_idx == len(self.route):
            reward = 1
            done = True
            print("done")

        print(f"Command: {self.command.name}| Next checkpoint: {next_checkpoint.name}")
        return ScenarioStepResult(self.command, reward, done, info)

        # def step(self, agent_vehicle):
        #     # Agent
        #     done = False
        #     agent_location = agent_vehicle.get_transform().location
        #
        #     if self.status is ManeuverStatus.SPAWNED:
        #         command = ChauffeurCommand.LANE_FOLLOW
        #         self.area_to_reach = self.entrances_to_reach[0]
        #
        #         if self.area_to_reach.enter_area.__contains__(agent_location):
        #             self.status = ManeuverStatus.REACHED_ROUNDABOUT_ENTRANCE
        #             self.entrances_reached += 1
        #
        #     if self.status is ManeuverStatus.REACHED_ROUNDABOUT_ENTRANCE:
        #         command = ChauffeurCommand.LANE_FOLLOW
        #
        #         if self.entrances_reached < self.num_exits_to_miss:
        #             next_idx = self.entrances_reached
        #             self.area_to_reach = self.entrances_to_reach[next_idx]
        #
        #             if agent_location in self.area_to_reach.enter_area:
        #                 self.entrances_reached += 1
        #         else:
        #             self.status = ManeuverStatus.SHOULD_TAKE_NEXT_EXIT
        #
        #
        #     if self.status is ManeuverStatus.SHOULD_TAKE_NEXT_EXIT:
        #         command = ChauffeurCommand.TURN_RIGHT
        #         roundabout_exit = EXITS[self.area_to_reach.next_exit_idx]
        #
        #         if agent_location in roundabout_exit.exit_area:
        #             self.status = ManeuverStatus.REACHED_EXIT_AREA
        #
        #     if self.status is ManeuverStatus.REACHED_EXIT_AREA:
        #         command = ChauffeurCommand.LANE_FOLLOW
        #         roundabout_exit = EXITS[self.area_to_reach.next_exit_idx]
        #
        #         if agent_location in roundabout_exit.target_area:
        #             self.status = ManeuverStatus.REACHED_FINAL_AREA
        #
        #     if self.status is ManeuverStatus.REACHED_FINAL_AREA:
        #         done = True
        #     print(self.status)

        #
        #
        # if agent_location in self.area_to_reach:
        #         self.next_entrance_idx += 1
        #         self.area_to_reach = self.entrances_to_reach[self.next_entrance_idx]
        #     else:
        #         should_prepare_to_take_exit = True
        #
        #     self.steps_left_to_reach_area = MAX_NUM_STEPS_TO_REACH_NEXT_AREA
        # else:
        #     self.steps_left_to_reach_area -= 1
        #
        #
        # if should_prepare_to_take_exit:
        #     self.area_to_reach = EXITS[self.area_to_reach.next_exit_idx]
        #     command = ChauffeurCommand.TURN_RIGHT
        #
        # if
        # command = ChauffeurCommand.TURN_RIGHT
        # current_vehicle_waypoint = self._map.get_waypoint(agent_location)
        # final_required_location = carla.Location(x=-57.765064, y=-3.625000, z=0.500000)
        reward = 0
        info = {}
        return ScenarioStepResult(
            chauffeur_cmd=command, reward=reward, done=done, info=info
        )

    def close(self):
        pass
        # if self.dummy_vehicle is not None:
        #     self.dummy_vehicle.destroy()


# args_lateral_dict = {
#     'K_P': 1,
#     'K_D': 0.02,
#     'K_I': 0,
#     'dt': DT}
# args_longitudinal_dict = {"K_P": 1.0, "K_D": 0, "K_I": 1, "dt": DT}
# self._pid = VehiclePIDController(
#     agent_vehicle,
#     args_lateral=args_lateral_dict,
#     args_longitudinal=args_longitudinal_dict,
# )
#
# blueprints = self._world.get_blueprint_library()
# bp = blueprints.find("vehicle.audi.a2")
# dummy_vehicle_spawn_transform = carla.Transform(
#     carla.Location(-0.049107, -19.766399, 0.500000), carla.Rotation(yaw=180)
# )
# # Location(x=-20.049107, y=-1.266399, z=0.500000)
# # Location(x=0.700893, y=19.983601, z=0.500000)
# # Location(x=19.450893, y=0.233601, z=0.500000)
# self.dummy_vehicle = self._world.spawn_actor(bp, dummy_vehicle_spawn_transform)

# dummy_vehicle_waypoint = self._map.get_waypoint(self.dummy_vehicle.get_location())
# possible_next_waypoints = dummy_vehicle_waypoint.next(0.01)
# angles = [compute_magnitude_angle(wp.transform.location, dummy_vehicle_waypoint.transform.location, dummy_vehicle_waypoint.transform.rotation.yaw) for wp in possible_next_waypoints]
# print(angles)
# target_waypoint = possible_next_waypoints[np.argmin(angles)]
# target_speed_kmh = 10
#
# # self._world.debug.draw_point(target_waypoint.transform.location, life_time=0.01)
# # control = self._pid.run_step(target_speed_kmh, target_waypoint)
# # self.dummy_vehicle.apply_control(control)
# self.dummy_vehicle.set_transform(target_waypoint.transform)
# # print(angles, control)
