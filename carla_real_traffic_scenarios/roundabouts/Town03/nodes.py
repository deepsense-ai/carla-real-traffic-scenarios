import carla
from carla_real_traffic_scenarios.roundabouts.types import RoundaboutNode, CircleArea

node3 = RoundaboutNode(
    name="Node3 - right",
    spawn_point=carla.Transform(
        carla.Location(x=4.894153594970703, y=61.459991455078125, z=0.5),
        carla.Rotation(yaw=270),
    ),
    entrance_area=CircleArea(
        center=carla.Location(x=5.050692, y=21.908991, z=0.500000), radius=7
    ),
    next_exit=CircleArea(
        center=carla.Location(x=27.760864, y=6.000000, z=0.500000), radius=3
    ),
    final_area_for_next_exit=CircleArea(
        center=carla.Location(x=54.827374, y=5.625000, z=0.500000), radius=3
    ),
    next_node=None,
)
node2 = RoundaboutNode(
    name="Node2 - bottom",
    spawn_point=carla.Transform(
        carla.Location(x=-60.9058464050293, y=1.0, z=0.5), carla.Rotation(yaw=358)
    ),
    entrance_area=CircleArea(
        center=carla.Location(x=-21.922626, y=1.375000, z=0.500000), radius=7.5
    ),
    next_exit=CircleArea(
        center=carla.Location(x=-9.886812, y=27.106071, z=0.500000), radius=3.5
    ),
    final_area_for_next_exit=CircleArea(
        center=carla.Location(x=-7.886812, y=69.356071, z=0.500000), radius=3
    ),
    next_node=node3,
)
node1 = RoundaboutNode(
    name="Node1 - left",
    spawn_point=carla.Transform(
        # SP1
        carla.Location(x=-6.230829238891602, y=-67.29000854492188, z=0.5),
        carla.Rotation(yaw=90),
    ),
    entrance_area=CircleArea(
        center=carla.Location(x=-3.922626, y=-20.875000, z=0.500000), radius=7.5
    ),
    next_exit=CircleArea(
        center=carla.Location(x=-29.239136, y=-6.250000, z=0.500000), radius=3
    ),
    final_area_for_next_exit=CircleArea(
        center=carla.Location(x=-51.834660, y=-2.930555, z=0.500000), radius=3
    ),
    next_node=node2,
)
node0 = RoundaboutNode(
    name="Node0 - top",
    spawn_point=carla.Transform(
        carla.Location(x=35.750507, y=-8.345556, z=0.5), carla.Rotation(yaw=180)
    ),
    entrance_area=CircleArea(
        center=carla.Location(x=20.827374, y=-5.125000, z=0.500000), radius=7.5
    ),
    next_exit=CircleArea(
        center=carla.Location(x=7.575461, y=-28.649958, z=0.500000), radius=3
    ),
    final_area_for_next_exit=CircleArea(
        center=carla.Location(x=7.075461, y=-50.946571, z=0.500000), radius=3
    ),
    next_node=node1,
)
node3.next_node = node0
TOWN03_ROUNDABOUT_NODES = [node0, node1, node2, node3]
