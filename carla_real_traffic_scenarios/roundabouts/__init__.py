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
from carla_real_traffic_scenarios.utils.carla import CollisionSensor

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

    It is advised (but not enforced by this implementation) to use CARLA synchronous mode - calling `world.tick()`
    after reset() or step(). Code example can be found in __main__.py (`roundabouts` package)
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
        self._collision_sensor: Optional[CollisionSensor] = None

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

        if self._collision_sensor:
            self._collision_sensor.destroy()

        self._collision_sensor = CollisionSensor(self._world, ego_vehicle)

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
            lightgreen_color = carla.Color(153, 255, 51)
            for route_checkpoint in self._route:
                debug_draw(
                    route_checkpoint.area,
                    self._world,
                    color=lightgreen_color,
                    life_time=1.0 / FPS,
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
        if self._collision_sensor:
            self._collision_sensor.destroy()
        self._driving_actors_manager.clean_up_all()
