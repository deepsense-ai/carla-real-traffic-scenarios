import carla
from carla_real_traffic_scenarios import DT
from carla_real_traffic_scenarios.ngsim import NGSimDatasets
from carla_real_traffic_scenarios.ngsim.ngsim_recording import NGSimRecording
from carla_real_traffic_scenarios.opendd.recording import OpenDDRecording
from carla_real_traffic_scenarios.utils.carla import RealTrafficVehiclesInCarla
from carla_real_traffic_scenarios.utils.transforms import Transform


def _parse_server_endpoint(server_endpoint):
    server_endpoint_items = server_endpoint.split(':')
    hostname = server_endpoint_items[0]
    port = 2000
    if len(server_endpoint_items) == 2:
        port = int(server_endpoint_items[1])
    return hostname, port


def _create_ngsim_simulator():
    ngsim_dataset = NGSimDatasets.US101
    data_dir = '/home/adam/src/rl/xy-trajectories'
    return NGSimRecording(data_dir, ngsim_dataset)


def _create_opendd_simulator():
    db_path = '/home/pawel/sandbox/opendd/rdb1to6/rdb1to6.sqlite'
    return OpenDDRecording(db_path)


class IdentityMapper:

    def real_traffic_to_carla(self, real_traffic_transform: Transform, z: float, rear_axle_offset: float) -> Transform:
        return real_traffic_transform


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Replay real-traffic scenario on carla simulator')
    parser.add_argument('--server', '-s', help='Address where carla simulator is listening; (host:[port])',
                        required=False, default='localhost:2000')
    args = parser.parse_args()

    hostname, port = _parse_server_endpoint(args.server)
    carla_client = carla.Client(hostname, port)
    carla_client.set_timeout(60)

    carla_synchronizer = None
    try:
        # simulator = _create_ngsim_simulator()
        # simulator.reset(US101Timeslots.TIMESLOT_3, 1350)

        simulator = _create_opendd_simulator()
        simulator.reset('rdb5_E1DJI_0001_data', 0)

        # ngsim_dataset = NGSimDatasets.I80
        print("Trying to connect to CARLA server. Make sure its up and running.")
        # world = carla_client.load_world(ngsim_dataset.carla_map.level_path)
        world = carla_client.load_world('rdb5')

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        world.apply_settings(settings)

        carla_synchronizer = RealTrafficVehiclesInCarla(
            carla_client,
            world
        )
        for _ in range(10000):
            vehicles = simulator.step()
            carla_synchronizer.step(vehicles)
            world.tick()
    finally:
        if carla_synchronizer is not None:
            carla_synchronizer.close()


if __name__ == '__main__':
    main()
