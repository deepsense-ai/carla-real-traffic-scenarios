import carla

from carla_real_traffic_scenarios.ngsim import NGSimDatasets, DatasetMode
from carla_real_traffic_scenarios.ngsim.ngsim_lanechange_scenario import NGSimLaneChangeScenario

if __name__ == '__main__':
    carla_client = carla.Client('localhost', 2000)
    carla_client.set_timeout(60)

    data_dir = '/home/adam/src/rl/xy-trajectories'

    ngsim_dataset = NGSimDatasets.I80
    world = carla_client.load_world(ngsim_dataset.carla_map.level_path)

    car_blueprint = world.get_blueprint_library().find('vehicle.audi.a2')
    # spawn points doesnt matter - scenario sets up position in reset
    dummy_spawn_point = carla.Transform(carla.Location(0, 0, 500), carla.Rotation())
    ego_car = world.spawn_actor(car_blueprint, dummy_spawn_point)

    scenario = NGSimLaneChangeScenario(
        ngsim_dataset, DatasetMode.TRAIN,
        data_dir=data_dir, client=carla_client
    )
    scenario.reset(ego_car)

    # OPEN-AI gym like loop:
    EPISODES_N = 10
    for _ in range(EPISODES_N):
        scenario.reset(ego_car)
        done = False
        while not done:
            # Read sensors, use policy to generate action and apply it as vehicle control to ego_car
            # ego_car.apply_control(carla.VehicleControl(throttle=1.0, steer=-1.0))
            chauffeur_cmd, reward, done, info = scenario.step(ego_car)
            world.tick()

    print("Scenario finished!")
    scenario.close()
