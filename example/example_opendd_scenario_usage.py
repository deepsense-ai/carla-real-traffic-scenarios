#!/bin/env python3
import carla
from carla_real_traffic_scenarios.opendd.scenario import OpenDDScenario


def main():
    carla_client = carla.Client('localhost', 2000)
    carla_client.set_timeout(60)

    # Download from http://bit.ly/PPUU-data
    data_dir = '/home/pawel/sandbox/opendd'

    print("Trying to connect to CARLA server. Make sure its up and running.")
    world = carla_client.load_world('rdb1')
    print("Connected!")

    car_blueprint = world.get_blueprint_library().find('vehicle.audi.a2')
    # spawn points doesnt matter - scenario sets up position in reset
    dummy_spawn_point = carla.Transform(carla.Location(0, 0, 500), carla.Rotation())
    ego_car = world.spawn_actor(car_blueprint, dummy_spawn_point)
    # Setup car sensors. Later use to it make predictions

    scenario = OpenDDScenario(carla_client, data_dir)
    scenario.reset(ego_car)

    spectator = world.get_spectator()

    # OPEN-AI gym like loop:
    EPISODES_N = 100
    for ep_ix in range(EPISODES_N):
        print(f"Running episode {ep_ix}")

        # NGSim scenario places ego_agent in a place of one of real-world vehicles and asks it to replicate
        # its either LANECHANGE_LEFT or LANECHANGE_RIGHT manuveur.
        scenario.reset(ego_car)
        done = False
        while not done:
            # Read sensors, use policy to generate action and apply it as vehicle control to ego_car
            # ego_car.apply_control(carla.VehicleControl(throttle=1.0, steer=-1.0))
            chauffeur_cmd, reward, done, info = scenario.step(ego_car)
            world.tick()

            birds_eye_view = carla.Transform(
                ego_car.get_transform().location + carla.Vector3D(x=0, y=0, z=20),
                carla.Rotation(pitch=-90),
            )
            spectator.set_transform(birds_eye_view)

    print("Scenario finished!")
    scenario.close()


if __name__ == '__main__':
    main()
