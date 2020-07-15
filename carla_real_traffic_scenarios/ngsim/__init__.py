import enum
from enum import auto
from typing import NamedTuple, FrozenSet, Sequence

from carla_real_traffic_scenarios.carla_maps import CarlaMap, CarlaMaps

FRAMES_BEFORE_MANUVEUR = 50
FRAMES_AFTER_MANUVEUR = 50


class NGSimTimeslot(NamedTuple):
    file_suffix: str
    blacklisted_vehicle_ids: FrozenSet[int]


class US101Timeslots:
    TIMESLOT_1 = NGSimTimeslot(
        '0750am-0805am',
        frozenset({2691, 2809, 2820, 2871, 2851, 2873})
    )
    TIMESLOT_2 = NGSimTimeslot(
        '0805am-0820am',
        frozenset({649, 806, 1690, 1725, 1734, 1773, 1949, 1877}),
    )
    TIMESLOT_3 = NGSimTimeslot(
        '0820am-0835am',
        frozenset({183, 329, 791, 804, 1187, 1183, 1107, 1247, 1202, 1371, 1346, 1435, 1390, 1912}),
    )

    @staticmethod
    def list():
        return [US101Timeslots.TIMESLOT_1, US101Timeslots.TIMESLOT_2, US101Timeslots.TIMESLOT_3]


class I80Timeslots:
    TIMESLOT_1 = NGSimTimeslot(
        '0400-0415',
        frozenset({
            1628, 2089, 2834, 2818, 2874,  # ground truth errors (GTE)
            1383, 1430, 1456, 1589, 1913
        })  # kinematic modelling errors (KME)
    )
    TIMESLOT_2 = NGSimTimeslot(
        '0500-0515',
        frozenset({
            537, 1119, 1261, 1215, 1288, 1381, 1382, 1348, 2512, 2462, 2442, 2427,
            2407, 2486, 2296, 2427, 2552, 2500, 2616, 2555, 2586, 2669,
            876, 882, 953, 1290, 1574, 2053, 2054, 2134, 2332, 2117, 2301, 2488,  # KME
            2519, 2421, 2788
        }),  # KME
    )
    TIMESLOT_3 = NGSimTimeslot(
        '0515-0530',
        frozenset({
            269, 567, 722, 790, 860, 1603, 1651, 1734, 1762, 1734,
            1800, 1722, 1878, 2056, 2075, 2258, 2252, 2285, 2362,
            3004, 401, 510, 682, 680, 815, 827, 1675, 1780, 1751, 1831,  # KME
            2200, 2080, 2119, 2170, 2369, 2480, 1797  # KME
        }),
    )

    @staticmethod
    def list():
        return [I80Timeslots.TIMESLOT_1, I80Timeslots.TIMESLOT_2, I80Timeslots.TIMESLOT_3]


class NGSimDataset(NamedTuple):
    name: str
    data_dir: str
    carla_map: CarlaMap
    rightmost_lane_id_for_lanechange_scenarios: int
    timeslots: Sequence[NGSimTimeslot]


class NGSimDatasets:
    I80 = NGSimDataset(
        "I80", "i80", CarlaMaps.I80,
        rightmost_lane_id_for_lanechange_scenarios=5,  # 6 == powel street onramp
        timeslots=tuple(I80Timeslots.list())
    )
    US101 = NGSimDataset(
        "US101", "us101", CarlaMaps.US101,
        rightmost_lane_id_for_lanechange_scenarios=5,  # [6, 7, 8] - auxilary/ramp_on/ramp_off lanes
        timeslots=tuple(US101Timeslots.list())
    )

    @staticmethod
    def list():
        return [NGSimDatasets.I80, NGSimDatasets.US101]


class DatasetMode(enum.Enum):
    TRAIN = auto()
    VALIDATION = auto()