import os
import carla
import random
import argparse

from carla_real_traffic_scenarios.carla_maps import CarlaMaps
from carla_real_traffic_scenarios.ngsim import NGSimDatasets, DatasetMode
from carla_real_traffic_scenarios.ngsim.scenario import NGSimLaneChangeScenario
from carla_real_traffic_scenarios.opendd.scenario import OpenDDScenario
from carla_real_traffic_scenarios.reward import RewardType
from carla_real_traffic_scenarios.scenario import Scenario


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ngsim", "opendd"], default="opendd")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=2000, type=int)
    parser.add_argument("--num-episodes", default=10, type=int)
    args = parser.parse_args()
    return args


def prepare_ngsim_scenario(client: carla.Client) -> Scenario:
    data_dir = os.environ["NGSIM_DIR"]
    ngsim_map = NGSimDatasets.list()
    ngsim_dataset = random.choice(ngsim_map)
    client.load_world(ngsim_dataset.carla_map.level_path)
    return NGSimLaneChangeScenario(
        ngsim_dataset,
        dataset_mode=DatasetMode.TRAIN,
        data_dir=data_dir,
        reward_type=RewardType.DENSE,
        client=client,
    )


def prepare_opendd_scenario(client: carla.Client) -> Scenario:
    data_dir = os.environ["OPENDD_DIR"]
    maps = ["rdb1", "rdb2", "rdb3", "rdb4", "rdb5", "rdb6", "rdb7"]
    map_name = random.choice(maps)
    carla_map = getattr(CarlaMaps, map_name.upper())
    client.load_world(carla_map.level_path)
    return OpenDDScenario(
        client,
        dataset_dir=data_dir,
        dataset_mode=DatasetMode.TRAIN,
        reward_type=RewardType.DENSE,
        place_name=map_name,
    )


def prepare_ego_vehicle(world: carla.World) -> carla.Actor:
    car_blueprint = world.get_blueprint_library().find("vehicle.audi.a2")

    # This will allow external scripts like manual_control.py or no_rendering_mode.py
    # from the official CARLA examples to take control over the ego agent
    car_blueprint.set_attribute("role_name", "hero")

    # spawn points doesnt matter - scenario sets up position in reset
    ego_vehicle = world.spawn_actor(
        car_blueprint, carla.Transform(carla.Location(0, 0, 500), carla.Rotation())
    )

    assert ego_vehicle is not None, "Ego vehicle could not be spawned"

    # Setup any car sensors you like, collect observations and then use them as input to your model
    return ego_vehicle


if __name__ == "__main__":
    args = parser_args()

    print(f"Trying to connect to CARLA server via {args.host}:{args.port}", end="...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(60)

    if args.dataset == "ngsim":
        scenario = prepare_ngsim_scenario(client)
    elif args.dataset == "opendd":
        scenario = prepare_opendd_scenario(client)

    world = client.get_world()
    spectator = world.get_spectator()
    ego_vehicle = prepare_ego_vehicle(world)
    scenario.reset(ego_vehicle)

    # OpenAI Gym interface:
    for ep_idx in range(args.num_episodes):
        print(f"Running episode {ep_idx+1}/{args.num_episodes}")

        # Ego vehicle replaces one of the real-world vehicles
        scenario.reset(ego_vehicle)
        done = False
        while not done:
            # Read sensors, use policy to generate action and apply it to control ego agent
            # ego_vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1.0))

            chauffeur_cmd, reward, done, info = scenario.step(ego_vehicle)
            print(f"Step: command={chauffeur_cmd.name}, reward={reward}")
            world.tick()

            # Server camera follows ego
            topdown_view = carla.Transform(
                ego_vehicle.get_transform().location + carla.Vector3D(x=0, y=0, z=30),
                carla.Rotation(pitch=-90),
            )
            spectator.set_transform(topdown_view)

    scenario.close()
