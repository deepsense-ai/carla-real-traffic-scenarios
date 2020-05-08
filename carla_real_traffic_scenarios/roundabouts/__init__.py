import carla
import random
from pathlib import Path
from typing import Optional

from carla_real_traffic_scenarios import FPS
from carla_real_traffic_scenarios.assets import markings
from carla_real_traffic_scenarios.assets.actor_manager import ActorManager
from carla_real_traffic_scenarios.roundabouts import Town03
from carla_real_traffic_scenarios.roundabouts.Town03.nodes import (
    TOWN03_ROUNDABOUT_NODES,
)
from carla_real_traffic_scenarios.roundabouts import route
from carla_real_traffic_scenarios.roundabouts.types import CircleArea, RoundaboutNode
from carla_real_traffic_scenarios.scenario import (
    ScenarioStepResult,
    Scenario,
    ChauffeurCommand,
)

MAX_NUM_STEPS_TO_REACH_CHECKPOINT = FPS * 10


class RoundaboutExitingScenario(Scenario):
    """
    Randomly chooses which exit to take and gives "turn right" after passing by the last roundabout checkpoint.
    Only Town03 roundabout is currently supported, but it's trivial to use with custom maps (just provide new marking files)
    """

    def __init__(self, client: carla.Client, sparse_reward_mode: bool = False):
        super().__init__(client)
        self._client = client
        self._sparse_reward_mode = sparse_reward_mode
        self._world = client.get_world()
        self._map = self._world.get_map()

        # Saving state between consecutive steps
        self._next_route_checkpoint_idx: Optional[int] = None
        self._steps_to_reach_next_checkpoint: Optional[int] = None
        self._command: ChauffeurCommand = None

        # Driving actors
        vehicle_blueprints = self._world.get_blueprint_library().filter("vehicle.*")
        self._car_blueprints = [
            bp
            for bp in vehicle_blueprints
            if int(bp.get_attribute("number_of_wheels")) == 4
        ]

        # Map
        data_dir = Path(Town03.__file__).parent / "data"
        self._driving_actors_markings = markings.deserialize_json_file(
            path=data_dir / "on-reset-spawn-points.assets.json"
        )
        self._driving_actors_manager = ActorManager(client)
        self._scenario_area = CircleArea(location=carla.Location(0, 0, 0), radius=100)

    def reset(self, ego_vehicle: carla.Vehicle):
        # Actors
        self._driving_actors_manager.clean_up_all()
        self._driving_actors_manager.spawn_random_assets_at_markings(
            markings=self._driving_actors_markings, coverage=0.1
        )

        # Ego vehicle
        start_node = random.choice(TOWN03_ROUNDABOUT_NODES)
        start_node.spawn_point.location.z = 0.1

        # Physics trick is necessary to prevent vehicle from keeping the velocity
        ego_vehicle.set_simulate_physics(False)
        ego_vehicle.set_transform(start_node.spawn_point)
        ego_vehicle.set_simulate_physics(True)

        # Route
        self._take_nth_exit = random.randrange(1, 5)
        self._route = route.build_roundabout_checkpoint_route(
            start_node=start_node, nth_exit_to_take=self._take_nth_exit
        )

        # States
        self._next_route_checkpoint_idx = 0
        self._command = ChauffeurCommand.LANE_FOLLOW
        self._steps_to_reach_next_checkpoint = MAX_NUM_STEPS_TO_REACH_CHECKPOINT

    def step(self, ego_vehicle: carla.Vehicle) -> ScenarioStepResult:
        assert self._steps_to_reach_next_checkpoint is not None, (
            "No more steps are allowed to be done in this episode."
            "Must call `reset(ego_vehicle=...)` first"
        )
        ego_transform = ego_vehicle.get_transform()
        ego_location = ego_transform.location
        next_checkpoint = self._route[self._next_route_checkpoint_idx]

        # Displaying checkpoints' areas server-side
        # for route_checkpoint in self.route:
        #     color = carla.Color(random.randrange(255), random.randrange(255), random.randrange(255))
        #     route_checkpoint.draw(self._world, color=color, life_time=0.1)
        # next_checkpoint.draw(self._world, life_time=0.01)

        checkpoint_area = next_checkpoint.area
        reward = 0
        done = False
        info = {}

        if ego_location in checkpoint_area:
            if not self._sparse_reward_mode:
                num_checkpoints_excluding_final = len(self._route) - 1
                reward = 1 / num_checkpoints_excluding_final
            self._command = next_checkpoint.command
            self._steps_to_reach_next_checkpoint = MAX_NUM_STEPS_TO_REACH_CHECKPOINT
            self._next_route_checkpoint_idx += 1

        is_ego_offroad = (
            self._map.get_waypoint(ego_location, project_to_road=False) is None
        )
        if is_ego_offroad:
            # alternatively may want to give negative rewards and not end the episode
            reward = 0
            done = True

        if self._next_route_checkpoint_idx == len(self._route):
            # NOTE in default (dense) reward mode, agent will get 2 points for completing whole route
            reward = 1
            done = True

        if self._steps_to_reach_next_checkpoint > 0:
            self._steps_to_reach_next_checkpoint -= 1
        else:
            self._steps_to_reach_next_checkpoint = None
            reward = 0
            done = True

        return ScenarioStepResult(self._command, reward, done, info)

    def close(self):
        self._driving_actors_manager.clean_up_all()


# TODO reset()
# if SPAWN_STATIC:
#     self.static_actors_manager.clean_up_all()
#     for marking in self.asset_spawn_markings:
#         marking.transform.location.z = 0.1
#     self.static_actors_manager.spawn_random_assets_at_markings(
#         markings=self.asset_spawn_markings, coverage=1
#     )
#     self.static_actors_manager.apply_physics_settings_to_spawned(enable=False)

# NOTE Must not tick..., use of .then() is necessary
# SetAutopilot = carla.command.SetAutopilot
# batch = [
#     SetAutopilot(actor_id, True)
#     for actor_id in self.driving_actors_manager.spawned
# ]
# responses = self._client.apply_batch_sync(batch, True)
# errors = [r.actor_id for r in responses if r.has_error()]
# print(errors)
# self.driving_actors_manager.apply_physics_settings_to_spawned(enable=False)

# NOTE step()
# print(len(self.driving_actors_manager.spawned), len(self._world.get_actors()))
# for actor_id in self.driving_actors_manager.spawned:
#     actor = self._world.get_actors().find(actor_id)
#     if actor is None:
#         print("nie ma", actor_id)
#         self.driving_actors_manager.spawned.remove(actor_id)
#         continue
#     actor_trans = actor.get_transform()
#     if actor_trans.location not in self._map_area:
#         new_actor = self._respawn_randomly(actor)
#         if new_actor:
#             new_actor.set_autopilot(True)

# blueprint = random.choice(self._car_blueprints)
# actor.destroy()
# self.driving_actors_manager.spawned.remove(actor_id)
# spawn_point_idx = np.argmin(self.respawn_usage_counter)
# spawn_point = self.driving_actors_respawn_markings[
#     spawn_point_idx
# ].transform
# actor = self.driving_actors_manager.spawn(spawn_point, blueprint)
# if actor is not None:
#     self.respawn_usage_counter[spawn_point_idx] += 1
#     actor.set_autopilot(True)
#     self.driving_actors_manager.spawned.append(actor.id)
# else:
#     if TELEPORTATION_DRIVING:
#         nearest_waypoint = self._map.get_waypoint(actor_trans.location)
#         next_transform = random.choice(nearest_waypoint.next(0.1)).transform
#         actor.set_transform(next_transform)
#     elif AUTOPILOT_DRIVING:
#         if get_speed(actor) < 3:
#             if self.actor_stuck_counter.get(actor_id):
#                 self.actor_stuck_counter[actor_id] += 1
#             else:
#                 self.actor_stuck_counter[actor_id] = 1
#         else:
#             self.actor_stuck_counter[actor_id] = 0
#
#         if self.actor_stuck_counter[actor_id] > 50:
#             new_actor = self._respawn_randomly(actor)
#             if new_actor:
#                 new_actor.set_autopilot(True)
# print(
#     f"Steps to reach checkpoint: {self.steps_to_reach_next_checkpoint} | Command: {self.command.name}| Next checkpoint: {next_checkpoint.name} | Take nth exit: {self.take_nth_exit}"
# )

# NOTE useful
# def _respawn_randomly(self, actor) -> Optional[carla.Actor]:
#     # least_used_idx = np.argmin(self.respawn_usage_counter)
#     # spawn_point = self.driving_actors_respawn_markings[least_used_idx].transform
#     # actor.set_transform(spawn_point)
#     # self.respawn_usage_counter[least_used_idx] += 1
#
#     actor.destroy()
#     self.driving_actors_manager.spawned.remove(actor.id)
#
#     blueprint = random.choice(self._car_blueprints)
#     spawn_point = random.choice(self.driving_actors_respawn_markings).transform
#     actor = self.driving_actors_manager.spawn(spawn_point, blueprint)
#     return actor
