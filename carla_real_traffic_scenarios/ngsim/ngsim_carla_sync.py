import logging
from typing import Dict, Optional

import carla

from sim2real.carla import VehicleModel, MODEL_BY_VEHICLE_NAME
from sim2real.carla.server import CarlaServerController
from sim2real.carla.transforms import Transform
from sim2real.carla.vehicle import Vehicle, VehicleRole, CouldNotSpawnVehicleError
from sim2real.ngsim_dataset import NGSimDataset
from carla_real_traffic_scenarios.ngsim.cords_mapping import MAPPER_BY_NGSIM_DATASET
from carla_real_traffic_scenarios.ngsim.ngsim_recording import NGSimCar
from sim2real.utils.collections import smallest_by
from sim2real.utils.geometry import jaccard_rectangles

LOGGER = logging.getLogger(__name__)


class NGSimVehiclesInCarla:

    def __init__(self, carla_server_controller: CarlaServerController, ngsim_dataset: NGSimDataset):
        self._vehicle_by_vehicle_id: Dict[int, Vehicle] = {}
        self._carla_server_controller = carla_server_controller
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
                t = self._mapper.ngsim_to_carla(ngsim_v.get_transform(), carla_v.transform.position.z,
                                                carla_v.vehicle_model.rear_axle_offset)
                commands.append(carla.command.ApplyTransform(carla_v._carla_vehicle_ref, t.as_carla_transform()))
            else:
                model = find_best_matching_model(ngsim_v)

                if model is None:
                    LOGGER.debug(f"Not found matching vehicle model for vehicle {ngsim_v}")
                    continue

                try:
                    target_transform = self._mapper.ngsim_to_carla(ngsim_v.get_transform(), model.z_offset,
                                                                   model.rear_axle_offset)
                    spawn_transform = Transform(target_transform.position.with_z(500), target_transform.orientation)

                    carla_v = Vehicle(self._carla_server_controller, vehicle_model=model, spawn_point=spawn_transform,
                                      sensor_specs=[], role=VehicleRole.PPUU_DATASET_REPLAY)
                    commands.append(
                        carla.command.ApplyTransform(carla_v._carla_vehicle_ref, target_transform.as_carla_transform())
                    )

                    self._vehicle_by_vehicle_id[ngsim_v.id] = carla_v
                except CouldNotSpawnVehicleError:
                    LOGGER.warning(
                        f"Error spawning vehicle with id {ngsim_v.id}. Ignoring it now in the future. Model: {model.model}.")
                    # Without ignoring such vehicles till the end of episode a vehicle might suddenly appears mid-road
                    # in future frames
                    self._ignored_ngsim_vehicle_ids.add(ngsim_v.id)
                    continue

            now_vehicle_ids = {v.id for v in vehicles}
            previous_vehicles_ids = set(self._vehicle_by_vehicle_id.keys())

            for to_remove_id in previous_vehicles_ids - now_vehicle_ids:
                self._vehicle_by_vehicle_id[to_remove_id].destroy()
                del self._vehicle_by_vehicle_id[to_remove_id]

        self._carla_server_controller._client.apply_batch_sync(commands, False)

    def close(self):
        for v in self._vehicle_by_vehicle_id.values():
            v.destroy()


def find_best_matching_model(ngsim_car: NGSimCar) -> Optional[VehicleModel]:
    USE_STRICTLY_SMALLER = False

    if USE_STRICTLY_SMALLER:
        # using stricly smaller models ensures that there will be no collisions
        models = [
            m for m in MODEL_BY_VEHICLE_NAME.values() if
            m.bounding_box.extent.x * 2 < ngsim_car.length_m and m.bounding_box.extent.y * 2 < ngsim_car.width_m
        ]
    else:
        models = list(MODEL_BY_VEHICLE_NAME.values())

    if len(models) == 0:
        return None

    def calc_fitness(ngsim_car: NGSimCar, vehicle_model: VehicleModel):
        return jaccard_rectangles(
            ngsim_car.length_m, ngsim_car.width_m,
            vehicle_model.bounding_box.extent.x * 2, vehicle_model.bounding_box.extent.y * 2
        )

    return smallest_by(models, lambda m: -calc_fitness(ngsim_car, m))
