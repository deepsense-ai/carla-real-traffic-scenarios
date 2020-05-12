import carla
from carla_real_traffic_scenarios.carla_maps import CarlaMaps
from carla_real_traffic_scenarios.roundabouts import RoundaboutScenario
from carla_real_traffic_scenarios import DT

SYNCHRONOUS_MODE = True


def set_birds_eye_view_spectator(
    spectator: carla.Actor, followed_location: carla.Location, above: float
):
    birds_eye_view = carla.Transform(
        carla.Location(x=followed_location.x, y=followed_location.y, z=above),
        carla.Rotation(pitch=-90),
    )
    spectator.set_transform(birds_eye_view)


client = carla.Client("localhost", 2000)
client.set_timeout(3.0)

world = client.load_world(CarlaMaps.TOWN03.level_path)
if SYNCHRONOUS_MODE:
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = DT
    world.apply_settings(settings)

map = world.get_map()
blueprints = world.get_blueprint_library()
bp = blueprints.find("vehicle.audi.a2")
bp.set_attribute("role_name", "hero")
agent_vehicle = world.spawn_actor(bp, map.get_spawn_points()[0])

spectator = world.get_spectator()
set_birds_eye_view_spectator(spectator, carla.Location(), above=80)

scenario = RoundaboutScenario(client)
scenario.reset(agent_vehicle)
if SYNCHRONOUS_MODE:
    world.tick()

print("Scenario has been loaded")
done = False
episode_reward = 0
try:
    while True:
        result = scenario.step(agent_vehicle)
        if SYNCHRONOUS_MODE:
            world.tick()
        episode_reward += result.reward

        if result.done:
            scenario.reset(agent_vehicle)
            episode_reward = 0
            print("Episode reward:", episode_reward)
            print("Scenario has been reset")
finally:
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    scenario.close()
    agent_vehicle.destroy()
