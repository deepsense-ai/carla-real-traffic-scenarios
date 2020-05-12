from typing import List

from carla_real_traffic_scenarios.roundabouts.types import (
    RoundaboutNode,
    RouteCheckpoint,
)
from carla_real_traffic_scenarios.scenario import ChauffeurCommand


def build_roundabout_checkpoint_route(
    start_node: RoundaboutNode, nth_exit_to_take: int
) -> List[RouteCheckpoint]:
    route = []
    current_node = start_node

    # From entrance to entrance: keep lane
    for idx in range(nth_exit_to_take - 1):
        checkpoint = RouteCheckpoint(
            name=f"{idx} entrance {current_node.name}",
            area=current_node.entrance_area,
            command=ChauffeurCommand.LANE_FOLLOW,
        )
        route.append(checkpoint)
        current_node = current_node.next_node

    # Reaching last entrance: turn right
    last_entrance_checkpoint = RouteCheckpoint(
        name=f"last entrance {current_node.name}",
        area=current_node.entrance_area,
        command=ChauffeurCommand.TURN_RIGHT,
    )
    exit_checkpoint = RouteCheckpoint(
        name=f"exit {current_node.next_node.name}",
        area=current_node.next_exit,
        command=ChauffeurCommand.LANE_FOLLOW,
    )
    final_checkpoint = RouteCheckpoint(
        name=f"final {current_node.next_node.name}",
        area=current_node.final_area_for_next_exit,
        command=ChauffeurCommand.LANE_FOLLOW,
    )
    route.append(last_entrance_checkpoint)
    route.append(exit_checkpoint)
    route.append(final_checkpoint)
    return route
