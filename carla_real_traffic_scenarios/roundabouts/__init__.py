import carla
import random
from pathlib import Path
from typing import Optional, List

from carla_real_traffic_scenarios import FPS
from carla_real_traffic_scenarios.assets import markings
from carla_real_traffic_scenarios.assets.actor_manager import ActorManager
from carla_real_traffic_scenarios.roundabouts import Town03
from carla_real_traffic_scenarios.roundabouts.Town03.nodes import (
    TOWN03_ROUNDABOUT_NODES,
)
from carla_real_traffic_scenarios.roundabouts import route
from carla_real_traffic_scenarios.roundabouts.types import (
    CircleArea,
    RoundaboutNode,
    RouteCheckpoint,
)
from carla_real_traffic_scenarios.scenario import (
    ScenarioStepResult,
    Scenario,
    ChauffeurCommand,
)
from carla_real_traffic_scenarios.utils import geometry

MAX_NUM_STEPS_TO_REACH_CHECKPOINT = FPS * 10
DEBUG = False


def debug_draw(area: CircleArea, world: carla.World, **kwargs):
    world.debug.draw_point(area.center, **kwargs)
    xs, ys = geometry.points_on_ring(radius=area.radius, num_points=10)
    for x, y in zip(xs, ys):
        center = carla.Location(x=area.center.x + x, y=area.center.y + y, z=0.2)
        world.debug.draw_point(center, **kwargs)


class RoundaboutScenario(Scenario):
    """
    Randomly chooses which roundabout exit the ego agent must take.
    "Turn right" command will be given just after entering the last checkpoint located on roundabout ring.
    Only Town03 roundabout is currently supported, but it's trivial to use with custom maps (just provide new marking files)

    Checkpoints can be visualized by toggling DEBUG flag.

    Specification:
    1. Ego vehicle must be on a road, otherwise reset() is triggered
       (alternatively may want to modify this behavior and just give negative rewards)

    2. Collision with other vehicle instantly triggers reset()
       
    3. Sparse reward mode: agent will get reward==1 when it reaches final checkpoint

    4. Dense reward mode (default): agent will get 1/n reward for each of n checkpoints
        and additional 1 for final checkpoint (max reward == 2)
    """

    def __init__(self, client: carla.Client, sparse_reward_mode: bool = False):
        super().__init__(client)
        self._sparse_reward_mode = sparse_reward_mode

        # Saving state between consecutive steps
        self._next_route_checkpoint_idx: Optional[int] = None
        self._steps_to_reach_next_checkpoint: Optional[int] = None
        self._command: Optional[ChauffeurCommand] = None
        self._collision_sensor: Optional[carla.Actor] = None
        self._collided: bool = False

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
        self._scenario_area = CircleArea(center=carla.Location(0, 0, 0), radius=100)
        self._route: Optional[List[RouteCheckpoint]] = None

    def reset(self, ego_vehicle: carla.Vehicle):
        # Actors
        self._driving_actors_manager.clean_up_all()
        self._driving_actors_manager.spawn_random_assets_at_markings(
            markings=self._driving_actors_markings, coverage=0.1
        )

        # Ego vehicle
        start_node = random.choice(TOWN03_ROUNDABOUT_NODES)
        start_node.spawn_point.location.z = 0.1

        if self._collision_sensor is not None:
            self._collision_sensor.destroy()
            self._collision_sensor = None

        blueprint_library = self._world.get_blueprint_library()
        collision_sensor_bp = blueprint_library.find("sensor.other.collision")
        self._collision_sensor = self._world.spawn_actor(
            collision_sensor_bp, ego_vehicle.get_transform(), attach_to=ego_vehicle
        )

        def on_collision(e):
            self._collided = True

        self._collision_sensor.listen(on_collision)
        self._collided = False

        # Physics trick is necessary to prevent vehicle from keeping the velocity
        ego_vehicle.set_simulate_physics(False)
        ego_vehicle.set_transform(start_node.spawn_point)
        ego_vehicle.set_simulate_physics(True)

        # Route
        take_nth_exit = random.randrange(1, 5)
        self._route = route.build_roundabout_checkpoint_route(
            start_node=start_node, nth_exit_to_take=take_nth_exit
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

        if DEBUG:
            color = carla.Color(153, 255, 51)  # light green
            for route_checkpoint in self._route:
                debug_draw(
                    route_checkpoint.area, self._world, color=color, life_time=0.01
                )
            debug_draw(next_checkpoint.area, self._world, life_time=0.01)

        checkpoint_area = next_checkpoint.area
        reward = 0
        done = False
        info = {}

        if self._collided is True:
            reward = 0
            done = True

        if ego_location in checkpoint_area:
            if not self._sparse_reward_mode:
                num_checkpoints_excluding_final = len(self._route) - 1
                reward = 1 / num_checkpoints_excluding_final
            self._command = next_checkpoint.command
            self._steps_to_reach_next_checkpoint = MAX_NUM_STEPS_TO_REACH_CHECKPOINT
            self._next_route_checkpoint_idx += 1

        is_ego_offroad = (
            self._world_map.get_waypoint(ego_location, project_to_road=False) is None
        )
        if is_ego_offroad:
            reward = 0
            done = True

        if self._next_route_checkpoint_idx == len(self._route):
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
