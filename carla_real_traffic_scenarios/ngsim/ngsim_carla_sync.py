import logging
import random
from typing import Dict, Optional

import carla

from carla_real_traffic_scenarios.ngsim import NGSimDataset
from carla_real_traffic_scenarios.ngsim.cords_mapping import MAPPER_BY_NGSIM_DATASET
from carla_real_traffic_scenarios.ngsim.ngsim_recording import NGSimCar
from carla_real_traffic_scenarios.utils.collections import smallest_by
from carla_real_traffic_scenarios.utils.geometry import jaccard_rectangles
from carla_real_traffic_scenarios.utils.transforms import Transform
from carla_real_traffic_scenarios.vehicles import VehicleModel, VEHICLES, VEHICLE_BY_TYPE_ID

LOGGER = logging.getLogger(__name__)


class NGSimVehiclesInCarla:

    def __init__(self, client: carla.Client, world: carla.World, ngsim_dataset: NGSimDataset):
        self._vehicle_by_vehicle_id: Dict[int, carla.Vehicle] = {}
        self._client = client
        self._world = world
        self._ignored_ngsim_vehicle_ids = set()
        self._mapper = MAPPER_BY_NGSIM_DATASET[ngsim_dataset]

    def step(self, vehicles):
        commands = []

        ngsim_v: NGSimCar
        for ngsim_v in vehicles:
            if ngsim_v.id in self._ignored_ngsim_vehicle_ids:
                continue

            if ngsim_v.id in self._vehicle_by_vehicle_id:
                carla_v = self._vehicle_by_vehicle_id[ngsim_v.id]
                v_data = VEHICLE_BY_TYPE_ID[carla_v.type_id]
                t = self._mapper.ngsim_to_carla(ngsim_v.get_transform(), carla_v.get_transform().location.z,
                                                v_data.rear_axle_offset)
                commands.append(carla.command.ApplyTransform(carla_v, t.as_carla_transform()))
            else:
                model = find_best_matching_model(ngsim_v)

                if model is None:
                    LOGGER.debug(f"Not found matching vehicle model for vehicle {ngsim_v}")
                    continue

                target_transform = self._mapper.ngsim_to_carla(ngsim_v.get_transform(), model.z_offset,
                                                               model.rear_axle_offset)
                spawn_transform = Transform(target_transform.position.with_z(500), target_transform.orientation)
                vehicle_blueprint = self._get_vehicle_blueprint(model.type_id)
                vehicle_blueprint.set_attribute('role_name', 'ngsim_replay')
                carla_v = self._world.try_spawn_actor(vehicle_blueprint, spawn_transform.as_carla_transform())
                if carla_v is None:
                    LOGGER.info(
                        f"Error spawning vehicle with id {ngsim_v.id}. Ignoring it now in the future. Model: {model.type_id}.")
                    # Without ignoring such vehicles till the end of episode a vehicle might suddenly appears mid-road
                    # in future frames
                    self._ignored_ngsim_vehicle_ids.add(ngsim_v.id)
                    continue
                commands.append(
                    carla.command.ApplyTransform(carla_v, target_transform.as_carla_transform())
                )
                self._vehicle_by_vehicle_id[ngsim_v.id] = carla_v

            now_vehicle_ids = {v.id for v in vehicles}
            previous_vehicles_ids = set(self._vehicle_by_vehicle_id.keys())

            for to_remove_id in previous_vehicles_ids - now_vehicle_ids:
                self._vehicle_by_vehicle_id[to_remove_id].destroy()
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
            v.destroy()


def find_best_matching_model(ngsim_car: NGSimCar) -> Optional[VehicleModel]:
    USE_STRICTLY_SMALLER = False

    if USE_STRICTLY_SMALLER:
        # using stricly smaller models ensures that there will be no collisions
        models = [
            m for m in VEHICLES if
            m.bounding_box.extent.x * 2 < ngsim_car.length_m and m.bounding_box.extent.y * 2 < ngsim_car.width_m
        ]
    else:
        models = list(VEHICLES)

    if len(models) == 0:
        return None

    def calc_fitness(ngsim_car: NGSimCar, vehicle_model: VehicleModel):
        return jaccard_rectangles(
            ngsim_car.length_m, ngsim_car.width_m,
            vehicle_model.bounding_box.extent.x * 2, vehicle_model.bounding_box.extent.y * 2
        )

    return smallest_by(models, lambda m: -calc_fitness(ngsim_car, m))
