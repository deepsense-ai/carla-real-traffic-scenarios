# -*- coding: utf-8 -*-
from typing import NamedTuple


class CarlaMap(NamedTuple):
    level_name: str
    level_path: str


class CarlaMaps:
    TOWN01 = CarlaMap("Town01", "/Game/Carla/Maps/Town01")
    TOWN02 = CarlaMap("Town02", "/Game/Carla/Maps/Town02")
    TOWN03 = CarlaMap("Town03", "/Game/Carla/Maps/Town03")
    TOWN04 = CarlaMap("Town04", "/Game/Carla/Maps/Town04")
    TOWN05 = CarlaMap("Town05", "/Game/Carla/Maps/Town05")
    TOWN06 = CarlaMap("Town06", "/Game/Carla/Maps/Town06")
    TOWN07 = CarlaMap("Town07", "/Game/Carla/Maps/Town07")
    I80 = CarlaMap("I-80", "/Game/real-traffic/Maps/I-80/I-80")
    US101 = CarlaMap("US-101", "/Game/real-traffic/Maps/US-101/US-101")
    RDB1 = CarlaMap("RDB1", "/Game/real-traffic/Maps/rdb1/rdb1")
    RDB2 = CarlaMap("RDB2", "/Game/real-traffic/Maps/rdb2/rdb2")
    RDB3 = CarlaMap("RDB3", "/Game/real-traffic/Maps/rdb3/rdb3")
    RDB4 = CarlaMap("RDB4", "/Game/real-traffic/Maps/rdb4/rdb4")
    RDB5 = CarlaMap("RDB5", "/Game/real-traffic/Maps/rdb5/rdb5")
    RDB6 = CarlaMap("RDB6", "/Game/real-traffic/Maps/rdb6/rdb6")

