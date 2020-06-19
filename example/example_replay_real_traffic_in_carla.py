#!/usr/bin/env python3
import os
import random
from collections import Iterator

import carla
from carla_real_traffic_scenarios import DT
from carla_real_traffic_scenarios.ngsim import NGSimDatasets, US101Timeslots
from carla_real_traffic_scenarios.ngsim.ngsim_recording import NGSimRecording
from carla_real_traffic_scenarios.opendd.dataset import OpenDDDataset
from carla_real_traffic_scenarios.opendd.recording import OpenDDRecording
from carla_real_traffic_scenarios.utils.carla import RealTrafficVehiclesInCarla

NGSIM_DIR = os.environ.get('NGSIM_DIR', '/home/adam/src/rl/xy-trajectories')
OPENDD_DIR = os.environ.get('OPENDD_DIR', '/mnt/ml-team/rl/carla/opendd/')
OPENDD_DIR = os.environ.get('OPENDD_DIR', '/home/pawel/sandbox/opendd/')


def _parse_server_endpoint(server_endpoint):
    server_endpoint_items = server_endpoint.split(':')
    hostname = server_endpoint_items[0]
    port = 2000
    if len(server_endpoint_items) == 2:
        port = int(server_endpoint_items[1])
    return hostname, port


def _create_ngsim_simulator():
    ngsim_dataset = NGSimDatasets.US101
    return NGSimRecording(NGSIM_DIR, ngsim_dataset)


def create_simulator(dataset_name):
    dataset_type, *dataset_details = dataset_name.split('/')
    dataset_type = dataset_type.lower()

    simulator = None
    if dataset_type == 'ngsim':
        simulator = _create_ngsim_simulator()
        simulator.reset(US101Timeslots.TIMESLOT_3, 1350)
    elif dataset_type == 'opendd':
        dataset_details = '/'.join(dataset_details)

        dataset = OpenDDDataset(OPENDD_DIR)
        simulator = OpenDDRecording(dataset=dataset)
        time_slots = [ts for ts in dataset.session_names if dataset_details in ts]
        time_slot = random.choice(time_slots)

        simulator.reset(time_slot, 0)

    return simulator


class SimulatorIterator(Iterator):

    def __init__(self, simulator):
        self._simulator = simulator

    def __next__(self):
        try:
            return self._simulator.step()
        except IndexError:
            raise StopIteration


def carla_setup(carla_client, level_path):
    world = carla_client.load_world(level_path)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = DT
    world.apply_settings(settings)
    return world


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Replay real-traffic scenario on carla simulator')
    parser.add_argument('--server', '-s', help='Address where carla simulator is listening; (host:[port])',
                        required=False, default='localhost:2000')
    parser.add_argument('dataset', help='Name of dataset to replay')
    args = parser.parse_args()

    hostname, port = _parse_server_endpoint(args.server)
    carla_client = carla.Client(hostname, port)
    carla_client.set_timeout(60)

    carla_synchronizer = None
    try:
        simulator = create_simulator(args.dataset)
        print("Trying to connect to CARLA server. Make sure its up and running.")
        world = carla_setup(carla_client, simulator.place_params.name)
        carla_synchronizer = RealTrafficVehiclesInCarla(carla_client, world)

        for vehicles in SimulatorIterator(simulator):
            carla_synchronizer.step(vehicles)
            world.tick()
    finally:
        if carla_synchronizer:
            carla_synchronizer.close()


if __name__ == '__main__':
    main()
