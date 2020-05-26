import carla

from carla_real_traffic_scenarios import DT
from carla_real_traffic_scenarios.ngsim import NGSimDatasets, US101Timeslots
from carla_real_traffic_scenarios.ngsim.ngsim_recording import NGSimRecording
from carla_real_traffic_scenarios.utils.carla import RealTrafficVehiclesInCarla


def main():
    ngsim_vehicles_in_carla = None
    try:
        ngsim_dataset = NGSimDatasets.US101
        data_dir = '/home/adam/src/rl/xy-trajectories'
        ngsim_recording = NGSimRecording(data_dir, ngsim_dataset)
        ngsim_recording.reset(US101Timeslots.TIMESLOT_3, 1350)

        carla_client = carla.Client('localhost', 2000)
        carla_client.set_timeout(60)

        ngsim_dataset = NGSimDatasets.I80
        print("Trying to connect to CARLA server. Make sure its up and running.")
        world = carla_client.load_world(ngsim_dataset.carla_map.level_path)
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        world.apply_settings(settings)

        ngsim_vehicles_in_carla = RealTrafficVehiclesInCarla(carla_client, world)
        for _ in range(10000):
            vehicles = ngsim_recording.step()
            ngsim_vehicles_in_carla.step(vehicles)
            world.tick()
    finally:
        if ngsim_vehicles_in_carla is not None:
            ngsim_vehicles_in_carla.close()


if __name__ == '__main__':
    main()
