import logging
import random
from typing import Dict, Optional, NamedTuple

import carla
from carla_real_traffic_scenarios.utils.collections import smallest_by
from carla_real_traffic_scenarios.utils.geometry import jaccard_rectangles
from carla_real_traffic_scenarios.utils.transforms import Transform, Vector3
from carla_real_traffic_scenarios.vehicles import VehicleModel, VEHICLES

LOGGER = logging.getLogger(__name__)


class RealTrafficVehicle(NamedTuple):
    id: int
    type_id: str
    timestamp_s: float
    width_m: float
    length_m: float
    transform: Transform
    speed: float
    debug: Optional[str]


class RealTrafficVehiclesInCarla:

    def __init__(self, client: carla.Client, world: carla.World):
        self._vehicle_by_vehicle_id: Dict[int, carla.Vehicle] = {}
        self._client = client
        self._world = world
        self._ignored_real_traffic_vehicle_ids = set()

    def step(self, vehicles):
        commands = []

        real_vehicle: RealTrafficVehicle
        for real_vehicle in vehicles:
            if real_vehicle.id in self._ignored_real_traffic_vehicle_ids:
                continue

            target_transform = real_vehicle.transform  # transform in carla coordinates
            if real_vehicle.id in self._vehicle_by_vehicle_id:
                carla_vehicle = self._vehicle_by_vehicle_id[real_vehicle.id]
                target_transform = Transform(target_transform.position.with_z(carla_vehicle.get_transform().location.z),
                                             target_transform.orientation)
                commands.append(carla.command.ApplyTransform(carla_vehicle, target_transform.as_carla_transform()))
            else:
                spawn_transform = Transform(target_transform.position.with_z(500), target_transform.orientation)
                vehicle_blueprint = self._get_vehicle_blueprint(real_vehicle.type_id)
                vehicle_blueprint.set_attribute('role_name', 'real_traffic_replay')
                carla_vehicle = self._world.try_spawn_actor(vehicle_blueprint, spawn_transform.as_carla_transform())
                if carla_vehicle is None:
                    LOGGER.info(
                        f"Error spawning vehicle with id {real_vehicle.id}. "
                        f"Ignoring it now in the future. Model: {real_vehicle.type_id}.")
                    # Without ignoring such vehicles till the end of episode a vehicle might suddenly appears mid-road
                    # in future frames
                    self._ignored_real_traffic_vehicle_ids.add(real_vehicle.id)
                    continue
                commands.append(
                    carla.command.ApplyTransform(carla_vehicle, target_transform.as_carla_transform())
                )
                self._vehicle_by_vehicle_id[real_vehicle.id] = carla_vehicle

            # Debug-only
            # if real_vehicle.debug:
            #     self._world.debug.draw_string((target_transform.position + Vector3(2, 0, 4)).as_carla_location(),
            #                                   str(real_vehicle.debug))

            now_vehicle_ids = {v.id for v in vehicles}
            previous_vehicles_ids = set(self._vehicle_by_vehicle_id.keys())

            for to_remove_id in previous_vehicles_ids - now_vehicle_ids:
                actor = self._vehicle_by_vehicle_id[to_remove_id]
                commands.append(carla.command.DestroyActor(actor.id))
                del self._vehicle_by_vehicle_id[to_remove_id]

        # TODO batch spawn and batch destroy
        self._client.apply_batch_sync(commands, False)

    def _get_vehicle_blueprint(self, type_id: str, randomize_color=True):
        """obtain vehicle blueprint based on vehicle filter; randomize if more than one match filter"""
        blueprints = self._world.get_blueprint_library().filter(type_id)
        blueprints = [b for b in blueprints if int(b.get_attribute('number_of_wheels')) == 4]
        blueprint = random.choice(blueprints)
        if randomize_color and blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        return blueprint

    def close(self):
        for v in self._vehicle_by_vehicle_id.values():
            if v.is_alive:
                v.destroy()


def find_best_matching_model(vehicle_width_m, vehicle_length_m) -> Optional[VehicleModel]:
    USE_STRICTLY_SMALLER = False

    if USE_STRICTLY_SMALLER:
        # using strictly smaller models ensures that there will be no collisions
        models = [
            m for m in VEHICLES if
            m.bounding_box.extent.x * 2 < vehicle_length_m and
            m.bounding_box.extent.y * 2 < vehicle_width_m
        ]
    else:
        models = list(VEHICLES)

    if len(models) == 0:
        return None

    def calc_fitness(vehicle_model: VehicleModel):
        return jaccard_rectangles(
            vehicle_length_m, vehicle_width_m,
            vehicle_model.bounding_box.extent.x * 2, vehicle_model.bounding_box.extent.y * 2
        )

    return smallest_by(models, lambda m: -calc_fitness(m))


def setup_carla_settings(client: carla.Client, synchronous: bool, time_delta_s: float):
    world = client.get_world()
    settings = world.get_settings()
    changed = False
    if settings.synchronous_mode != synchronous:
        LOGGER.warning(f"Switch synchronous_mode={synchronous}")
        settings.synchronous_mode = synchronous
        changed = True
    if settings.fixed_delta_seconds != time_delta_s:
        LOGGER.warning(f"Change fixed_delta_seconds={time_delta_s}")
        settings.fixed_delta_seconds = time_delta_s
        changed = True
    if changed:
        world.apply_settings(settings)


class CollisionSensor:

    def __init__(self, world: carla.World, carla_vehicle: carla.Vehicle):
        self.has_collided = False

        def on_collision(e):
            self.has_collided = True

        blueprint_library = world.get_blueprint_library()
        blueprint = blueprint_library.find('sensor.other.collision')
        self._collision_sensor = world.spawn_actor(blueprint, carla_vehicle.get_transform(), attach_to=carla_vehicle)
        self._collision_sensor.listen(on_collision)

    def destroy(self):
        if self._collision_sensor and self._collision_sensor.is_alive:
            self._collision_sensor.destroy()
