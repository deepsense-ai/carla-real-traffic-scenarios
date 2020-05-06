import itertools
import math
import random
from enum import Enum, auto
from pathlib import Path
from typing import NamedTuple, List


import carla
from libs.carla_real_traffic_scenarios.carla_real_traffic_scenarios.roundabouts.town03_roundabout_nodes import \
    TOWN03_ROUNDABOUT_NODES

from libs.carla_real_traffic_scenarios.carla_real_traffic_scenarios.roundabouts.types import CircleArea, RoundaboutNode
from libs.carla_real_traffic_scenarios.carla_real_traffic_scenarios.scenario import (
    ScenarioStepResult,
)
from sim2real.birdview.__main__ import get_speed
from sim2real.carla import ChauffeurCommand
import numpy as np

from sim2real.carla.maps.assets import markings
from sim2real.carla.maps.assets.actor_manager import ActorManager
from sim2real.carla.maps.assets.utils import clone_location





def circle_points(r, n):
    t = np.linspace(0, 2 * np.pi, n)
    xs = r * np.cos(t)
    ys = r * np.sin(t)
    return xs, ys


class RouteCheckpoint(NamedTuple):
    name: str
    area: CircleArea
    command: ChauffeurCommand

    def draw(self, world: carla.World, **kwargs):
        world.debug.draw_point(self.area.location, **kwargs)

        xs, ys = circle_points(self.area.radius, 10)
        for x, y in zip(xs, ys):
            a = clone_location(self.area.location)
            a.x += x
            a.y += y
            world.debug.draw_point(a, **kwargs)


SPAWN_STATIC = False
SPAWN_DRIVING = True

TELEPORTATION_DRIVING = False
AUTOPILOT_DRIVING = not TELEPORTATION_DRIVING

MAX_NUM_STEPS_TO_REACH_CHECKPOINT = 200


class ExitingRoundaboutScenario:
    def __init__(self, client):
        self.num_actors_to_spawn = random.randrange(20, 40)
        self._client = client
        self._world = client.get_world()
        self._map = self._world.get_map()
        self._map_area = CircleArea(location=carla.Location(0, 0, 0), radius=100)
        vehicle_blueprints = self._world.get_blueprint_library().filter("vehicle.*")
        self._car_blueprints = [
            bp
            for bp in vehicle_blueprints
            if int(bp.get_attribute("number_of_wheels")) == 4
        ]
        self._actors_ids = []
        self._actors = []
        self.actor_stuck_counter = {}
        markings_file = Path(
            "/home/michalmartyniak/repos/sim2real/tools/assets_placement_editor/output/Town03-roundabout-inner-ring-10:24:06.assets.json"
        )
        driving_actors_markings_file = Path(
            "/home/michalmartyniak/repos/sim2real/tools/assets_placement_editor/output/Town03-roundabout-spawnpoints-14:21:31.assets.json"
        )
        driving_actors_respawns_markings_file = Path(
            "/home/michalmartyniak/repos/sim2real/tools/assets_placement_editor/output/Town03-roundabout-respawnpoints-15:40:50.assets.json"
        )

        self.asset_spawn_markings = markings.deserialize_json_file(path=markings_file)
        self.driving_actors_markings = markings.deserialize_json_file(
            path=driving_actors_markings_file
        )
        self.driving_actors_respawn_markings = markings.deserialize_json_file(
            path=driving_actors_respawns_markings_file
        )
        self.respawn_usage_counter = np.zeros(len(self.driving_actors_respawn_markings))

        self.static_actors_manager = ActorManager(client)
        self.driving_actors_manager = ActorManager(client)

        self.steps_to_reach_next_checkpoint = MAX_NUM_STEPS_TO_REACH_CHECKPOINT

    def teleport_to_lru_location(self, actor):
        least_used_idx = np.argmin(self.respawn_usage_counter)
        spawn_point = self.driving_actors_respawn_markings[least_used_idx].transform
        actor.set_transform(spawn_point)
        self.respawn_usage_counter[least_used_idx] += 1

    def _build_checkpoint_route(
        self, start_node: RoundaboutNode, nth_exit_to_take: int
    ) -> List[RouteCheckpoint]:
        route = []
        current_node = start_node

        # From entrance to entrance: keep lane
        for idx in range(nth_exit_to_take - 1):
            checkpoint = RouteCheckpoint(
                name=f"{idx} entrance {current_node.name}",
                area=current_node.entrance_area,
                command=ChauffeurCommand.LANE_FOLLOW,
            )
            route.append(checkpoint)
            current_node = current_node.next_node

        # Reaching last entrance: turn right
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
        route.append(last_entrance_checkpoint)
        route.append(exit_checkpoint)
        route.append(final_checkpoint)
        return route

    def reset(self, veh):
        if SPAWN_STATIC:
            self.static_actors_manager.clean_up_all()
            for marking in self.asset_spawn_markings:
                marking.transform.location.z = 0.1
            self.static_actors_manager.spawn_random_assets_at_markings(
                markings=self.asset_spawn_markings, coverage=1
            )
            self.static_actors_manager.apply_physics_settings_to_spawned(enable=False)

        if SPAWN_DRIVING:
            self.driving_actors_manager.clean_up_all()
            self.driving_actors_manager.spawn_random_assets_at_markings(
                markings=self.driving_actors_markings, coverage=random.uniform(0.1, 0.4)
            )
            SetAutopilot = carla.command.SetAutopilot
            batch = [
                SetAutopilot(actor_id, True)
                for actor_id in self.driving_actors_manager.spawned
            ]
            self._client.apply_batch_sync(batch)
            # self.driving_actors_manager.apply_physics_settings_to_spawned(enable=False)

        start_node = random.choice(TOWN03_ROUNDABOUT_NODES)
        veh.set_transform(start_node.spawn_point)

        self.take_nth_exit = random.randrange(1, 5)
        self.route = self._build_checkpoint_route(
            start_node=start_node, nth_exit_to_take=self.take_nth_exit
        )
        self.next_route_checkpoint_idx = 0
        self.command = ChauffeurCommand.LANE_FOLLOW
        self.steps_to_reach_next_checkpoint = MAX_NUM_STEPS_TO_REACH_CHECKPOINT

    def step(self, veh):
        trans = veh.get_transform()
        loc = trans.location
        next_checkpoint = self.route[self.next_route_checkpoint_idx]

        # DEBUG
        # for route_checkpoint in self.route:
        #     color = carla.Color(random.randrange(255), random.randrange(255), random.randrange(255))
        #     route_checkpoint.draw(self._world, color=color, life_time=0.1)
        # next_checkpoint.draw(self._world, life_time=0.01)
        # DEBUG

        checkpoint_area = next_checkpoint.area
        reward = 0
        done = False
        info = {}

        if loc in checkpoint_area:
            self.command = next_checkpoint.command
            self.steps_to_reach_next_checkpoint = MAX_NUM_STEPS_TO_REACH_CHECKPOINT
            self.next_route_checkpoint_idx += 1

        if self.next_route_checkpoint_idx == len(self.route):
            reward = 1
            done = True

        if self.steps_to_reach_next_checkpoint >= 0:
            self.steps_to_reach_next_checkpoint -= 1
        else:
            reward = 0
            done = True

        for actor_id in self.driving_actors_manager.spawned:
            actor = self._world.get_actors().find(actor_id)
            if actor is None:
                self.driving_actors_manager.spawned.remove(actor_id)
                continue
            actor_trans = actor.get_transform()
            if actor_trans.location not in self._map_area:
                # self.teleport_to_lru_location(actor)

                # zabijanie i tworzenie na nowo

                blueprint = random.choice(self._car_blueprints)
                actor.destroy()
                self.driving_actors_manager.spawned.remove(actor_id)
                spawn_point_idx = np.argmin(self.respawn_usage_counter)
                spawn_point = self.driving_actors_respawn_markings[
                    spawn_point_idx
                ].transform
                actor = self.driving_actors_manager.spawn(spawn_point, blueprint)
                if actor is not None:
                    self.respawn_usage_counter[spawn_point_idx] += 1
                    actor.set_autopilot(True)
                    self.driving_actors_manager.spawned.append(actor.id)
            else:
                if TELEPORTATION_DRIVING:
                    nearest_waypoint = self._map.get_waypoint(actor_trans.location)
                    next_transform = random.choice(nearest_waypoint.next(0.1)).transform
                    actor.set_transform(next_transform)
                elif AUTOPILOT_DRIVING:
                    if get_speed(actor) < 3:
                        if self.actor_stuck_counter.get(actor_id):
                            self.actor_stuck_counter[actor_id] += 1
                        else:
                            self.actor_stuck_counter[actor_id] = 1
                    else:
                        self.actor_stuck_counter[actor_id] = 0

                    # if self.actor_stuck_counter[actor_id] > 100:
                    #     self.teleport_to_lru_location(actor)

        print(
            f"Steps to reach checkpoint: {self.steps_to_reach_next_checkpoint} | Command: {self.command.name}| Next checkpoint: {next_checkpoint.name} | Take nth exit: {self.take_nth_exit}"
        )
        return ScenarioStepResult(self.command, reward, done, info)

    def close(self):
        self.static_actors_manager.clean_up_all()
        self.driving_actors_manager.clean_up_all()
