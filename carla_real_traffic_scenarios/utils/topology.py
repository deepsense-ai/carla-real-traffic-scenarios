# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides implementation for GlobalRoutePlannerDAO
"""
import logging
from queue import Queue
from typing import List

import carla
import networkx as nx
import numpy as np

from carla_real_traffic_scenarios.utils.transforms import Vector3

LOGGER = logging.getLogger(__name__)

_ROUTE_STEP_M = 5


def get_lane_id(wp):
    return wp.road_id, wp.section_id, wp.lane_id


def same_lane(wp1, wp2):
    return get_lane_id(wp1) == get_lane_id(wp2)


def same_waypoint(wp1, wp2):
    return wp1.id == wp2.id


class Topology:
    """
    This class is the data access layer for fetching data
    from the carla server instance for GlobalRoutePlanner
    """

    def __init__(self, world_map: carla.Map, sampling_resolution: float = 1.0):
        """
        Constructor method.
            :param world_map: carla.world object
            :param sampling_resolution: sampling distance between waypoints
        """
        self._sampling_resolution_m = sampling_resolution
        self._world_map = world_map
        topology = self._get_topology()
        self._graph, self._id_map, self._lane_id_to_edge = self._build_graph(topology)

    def _get_topology(self):
        """
        Accessor for topology.
        This function retrieves topology from the server as a list of
        road segments as pairs of waypoint objects, and processes the
        topology into a list of dictionary objects.

            :return topology: list of dictionary objects with the following attributes
                entry   -   waypoint of entry point of road segment
                entryxyz-   (x,y,z) of entry point of road segment
                exit    -   waypoint of exit point of road segment
                exitxyz -   (x,y,z) of exit point of road segment
                path    -   list of waypoints separated by 1m from entry
                            to exit
        """
        topology = []
        # Retrieving waypoints to construct a detailed topology
        for segment in self._world_map.get_topology():
            wp1, wp2 = segment[0], segment[1]
            l1, l2 = wp1.transform.location, wp2.transform.location
            # Rounding off to avoid floating point imprecision
            x1, y1, z1, x2, y2, z2 = np.round([l1.x, l1.y, l1.z, l2.x, l2.y, l2.z], 0)
            wp1.transform.location, wp2.transform.location = l1, l2
            seg_dict = dict()
            seg_dict['entry'], seg_dict['exit'] = wp1, wp2
            seg_dict['entryxyz'], seg_dict['exitxyz'] = (x1, y1, z1), (x2, y2, z2)
            seg_dict['path'] = []
            endloc = wp2.transform.location
            if wp1.transform.location.distance(endloc) > self._sampling_resolution_m:
                w = wp1.next(self._sampling_resolution_m)[0]
                while w.transform.location.distance(endloc) > self._sampling_resolution_m:
                    seg_dict['path'].append(w)
                    w = w.next(self._sampling_resolution_m)[0]
            else:
                seg_dict['path'].append(wp1.next(self._sampling_resolution_m)[0])
            topology.append(seg_dict)
        return topology

    def _build_graph(self, topology):
        """
        slimmed version of carla agents.navigation.GlobalRoutePlanner._build_graph method

        This function builds a networkx graph representation of topology.
        The topology is read from self._topology.
        graph node properties:
            vertex   -   (x,y,z) position in world map
        graph edge properties:
            entry_vector    -   unit vector along tangent at entry point
            exit_vector     -   unit vector along tangent at exit point
            net_vector      -   unit vector of the chord from entry to exit
            intersection    -   boolean indicating if the edge belongs to an
                                intersection
        return      :   graph -> networkx graph representing the world map,
                        id_map-> mapping from (x,y,z) to node id
                        road_id_to_edge-> map from road id to edge in the graph
        """
        graph = nx.DiGraph()
        id_map = dict()  # Map with structure {(x,y,z): id, ... }
        road_id_to_edge = dict()  # Map with structure {road_id: {lane_id: edge, ... }, ... }

        for segment in topology:
            path = segment['path']
            entry_wp, exit_wp = segment['entry'], segment['exit']
            intersection = entry_wp.is_junction

            road_id, section_id, lane_id = entry_wp.road_id, entry_wp.section_id, entry_wp.lane_id

            entry_xyz, exit_xyz = segment['entryxyz'], segment['exitxyz']
            for vertex in entry_xyz, exit_xyz:
                # Adding unique nodes and populating id_map
                if vertex not in id_map:
                    new_id = len(id_map)
                    id_map[vertex] = new_id
                    graph.add_node(new_id, vertex=vertex)
            n1 = id_map[entry_xyz]
            n2 = id_map[exit_xyz]
            if road_id not in road_id_to_edge:
                road_id_to_edge[road_id] = dict()
            if section_id not in road_id_to_edge[road_id]:
                road_id_to_edge[road_id][section_id] = dict()
            road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)

            entry_carla_vector = entry_wp.transform.rotation.get_forward_vector()
            exit_carla_vector = exit_wp.transform.rotation.get_forward_vector()
            net_vector = Vector3.from_carla_location(exit_wp.transform.location) - \
                         Vector3.from_carla_location(entry_wp.transform.location)
            net_vector /= np.linalg.norm(net_vector.as_numpy()) + np.finfo(float).eps

            segment = [entry_wp] + path
            length = self._calc_segment_length(segment)

            # Adding edge with attributes
            graph.add_edge(
                n1, n2,
                length=length, path=path,
                entry_waypoint=entry_wp, exit_waypoint=exit_wp,
                entry_vector=np.array([entry_carla_vector.x, entry_carla_vector.y, entry_carla_vector.z]),
                exit_vector=np.array([exit_carla_vector.x, exit_carla_vector.y, exit_carla_vector.z]),
                net_vector=net_vector, intersection=intersection)

        return graph, id_map, road_id_to_edge

    def _calc_segment_length(self, segment):
        return np.abs(segment[0].s - segment[-1].s)

    def get_sampling_resolution_m(self):
        return self._sampling_resolution_m

    def get_predecessors(self, waypoint: carla.Waypoint, distance: float) -> List[carla.Waypoint]:
        assert distance > 0
        forward = waypoint.lane_id > 0
        signed_distance = distance if forward else -distance

        n1, n2 = self._lane_id_to_edge[waypoint.road_id][waypoint.section_id][waypoint.lane_id]
        segment = self._graph.edges[n1, n2]
        segment_waypoints = [segment['entry_waypoint']] + segment['path']

        # check distance over s attribute instead of calculating euclidean distance between locations
        # because there are some differences between calculation from s attribute and euclidean distance
        # 'length' property of segment shall be also calculated with this method
        distances = np.array([wp.s - waypoint.s for wp in segment_waypoints])
        current_waypoint_idx = np.argmin(np.abs(distances))

        ref_idx = -1 if forward else 0
        relative_m = segment_waypoints[current_waypoint_idx].s - segment_waypoints[ref_idx].s
        remaining_lane_length = segment['length'] - relative_m if forward else relative_m
        assert remaining_lane_length >= 0 or np.isclose(remaining_lane_length, 0, atol=0.1)

        # If after subtracting the distance we are still in the same lane, return waypoint which is distanced
        # same current waypoint with the extra distance.
        if distance <= remaining_lane_length:
            result_waypoint_idx = np.argmin(np.abs(distances - signed_distance))
            result = segment_waypoints[int(result_waypoint_idx)]
            assert result.s - waypoint.s - signed_distance < self._sampling_resolution_m
            return [result]

        # If we run out of remaining_lane_length we have to go to the predecessors
        results = []
        edges_to_check = [(n0, n1) for n0 in self._graph.predecessors(n1)]
        while edges_to_check:
            n0, n1 = edges_to_check.pop(0)
            if n0 == n1:
                continue

            edge = self._graph.edges[n0, n1]
            edge_waypoints = [edge['entry_waypoint']] + edge['path']
            last_waypoint = edge_waypoints[-1]

            if same_lane(last_waypoint, waypoint):  # if on same lane - just take predecessors
                edges_to_check.extend([(n_1, n0) for n_1 in self._graph.predecessors(n0)])
                continue
            remaining_distance = distance - remaining_lane_length
            predecessors = self.get_predecessors(last_waypoint, remaining_distance)
            results.extend(predecessors)
        return results

    def get_successors(self, waypoint: carla.Waypoint, distance: float) -> List[carla.Waypoint]:
        return waypoint.next(distance)

    def get_forward_routes(self, start_waypoint: carla.Waypoint, min_length_m: float) -> List[List[carla.Transform]]:

        def _unroll(wp: carla.Waypoint, remaining_distance_m: float):
            if remaining_distance_m <= 0:
                return [[]]

            results = []
            for nwp in self.get_successors(wp, _ROUTE_STEP_M):
                distance = nwp.transform.location.distance(wp.transform.location)
                for sub_route in _unroll(nwp, remaining_distance_m - distance):
                    route = [wp.transform] + sub_route
                    results.append(route)
            return results

        return _unroll(start_waypoint, min_length_m)

    def get_backward_routes(self, start_waypoint: carla.Waypoint, min_length_m: float) -> List[List[carla.Transform]]:
        def _unroll(wp: carla.Waypoint, remaining_distance_m: float):
            if remaining_distance_m <= 0:
                return [[]]

            results = []
            for pwp in self.get_predecessors(wp, _ROUTE_STEP_M):
                distance = pwp.transform.location.distance(wp.transform.location)
                for sub_route in _unroll(pwp, remaining_distance_m - distance):
                    route = sub_route + [wp.transform]
                    results.append(route)
            return results

        return _unroll(start_waypoint, min_length_m)


def _unroll_waypoint(wp, max_distance, step, backward=True):
    ref_wp = wp
    q = Queue()
    q.put(wp)
    waypoints = [wp]
    while not q.empty():
        wp = q.get()
        tmp = wp.previous(step) if backward else wp.next(step)
        for w in tmp:
            if w.transform.location.distance(ref_wp.transform.location) < max_distance:
                q.put(w)
        waypoints.extend(tmp)
    return waypoints


def get_lane_ids(lane_wp, max_distances=(300, 300), step=2):
    forward_distance, back_distance = max_distances
    back = [get_lane_id(wp) for wp in _unroll_waypoint(lane_wp, back_distance, step)]
    forward = [get_lane_id(wp) for wp in _unroll_waypoint(lane_wp, forward_distance, step, backward=False)]
    return sorted(set(back + forward))
