import random

import carla
import logging
from typing import List, Optional

from carla_real_traffic_scenarios.assets import blueprints
from carla_real_traffic_scenarios.assets.markings import Marking

log = logging.getLogger(__name__)

ActorId = int


class ActorManager:
    """Responsible for tracking spawn/removal operations of `carla.Actor` objects."""

    def __init__(self, client: carla.Client):
        self.client = client
        self.world = client.get_world()
        self.library = self.world.get_blueprint_library()
        self.spawned: List[ActorId] = []

    def spawn(
        self, transform: carla.Transform, blueprint: carla.ActorBlueprint
    ) -> Optional[carla.Actor]:
        """Spawn an actor, then store it in buffer."""
        new_actor: carla.Actor = self.world.try_spawn_actor(blueprint, transform)

        if not new_actor:
            log.debug(f"Blueprint {blueprint.id} could not be spawned!")
            return None

        self.spawned.append(new_actor.id)
        log.debug(f"{new_actor.type_id} ({new_actor.id}) has been spawned.")
        return new_actor

    def spawn_random_assets_at_markings(
        self, markings: List[Marking], coverage: float
    ) -> int:
        """Select a subset of all markings, choose random blueprints, their attributes, then spawn them all.

        If markings are very densely distributed, there'll be collisions so not all spawns will be successful.
        Spawning order is not deterministic - depends on server timing, performance, vehicle sizes, etc.
        Return how much spawn operations actually succeeded.
        """
        if coverage < 0 or coverage > 1:
            raise ValueError("coverage argument must be between 0 and 1")
        take_k = int(len(markings) * coverage)
        selected_markings = random.sample(markings, k=take_k)
        log.debug(f"Attempting to spawn {len(selected_markings)} assets")

        batch = []
        for marking in selected_markings:
            wildcard_pattern: str = random.choice(marking.blueprint_patterns)
            bp_name: str = random.choice(self.library.filter(wildcard_pattern)).id
            bp: carla.ActorBlueprint = self.library.find(bp_name)
            blueprints.randomize_attributes(bp)
            if marking.yaw_agnostic:
                marking.transform.rotation.yaw = random.randint(0, 359)
            batch.append(carla.command.SpawnActor(bp, marking.transform).then(carla.command.SetAutopilot(carla.command.FutureActor, True)))

        responses = self.client.apply_batch_sync(batch, False)
        self.spawned += [r.actor_id for r in responses if not r.has_error()]

        log.info(f"Spawned {len(self.spawned)} assets")
        return len(self.spawned)

    def clean_up_all(self):
        """Destroy and forget all already-spawned actors."""
        log.debug(f"{len(self.spawned)} assets are going to be destroyed.")

        batch = [carla.command.DestroyActor(actor_id) for actor_id in self.spawned]
        responses = self.client.apply_batch_sync(batch, False)
        successful: List[ActorId] = [r.actor_id for r in responses if not r.has_error()]

        for actor_id in successful:
            self.spawned.remove(actor_id)

        log.info(f"{len(successful)} have been cleaned up.")

    def clean_up_most_recent(self):
        """Undo operation for recently spawned actor."""
        if len(self.spawned) > 0:
            actor_id: int = self.spawned.pop()
            most_recent: carla.Actor = self.world.get_actor(actor_id)
            most_recent.destroy()
            log.debug(f"{most_recent.type_id} has been cleaned up.")
        else:
            log.debug("There are no actors to clean up!")

    def apply_physics_settings_to_spawned(self, enable: bool):
        set_physics = carla.command.SetSimulatePhysics
        batch = [set_physics(actor_id, enable) for actor_id in self.spawned]
        self.client.apply_batch(batch)
