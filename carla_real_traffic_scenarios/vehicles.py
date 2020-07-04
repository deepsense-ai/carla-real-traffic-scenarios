# -*- coding: utf-8 -*-
from typing import NamedTuple

from carla_real_traffic_scenarios.utils.transforms import Vector3


class BoundingBox(NamedTuple):
    location: Vector3  # position of center of bb w.r.t. car transform position
    extent: Vector3  # distance to walls from center point. Side len is extent*2


class VehicleModel(NamedTuple):
    type_id: str
    wheelbase_m: float
    bounding_box: BoundingBox  # See tools/extract_vehicle_bounding_boxes.main.py script
    z_offset: float  # car z-coordinate when standing on z=0 plane. See tools/extract_vehicle_z_offsets.main.py
    front_axle_offset: float  # extract_vehicle_axle_positions.main.py

    @property
    def rear_axle_offset(self):
        return self.front_axle_offset - self.wheelbase_m


VEHICLES = [
    VehicleModel('vehicle.audi.a2', 2.5063290535550626,
                 BoundingBox(Vector3(x=0.000000, y=0.000000, z=0.780000),
                             Vector3(x=1.859311, y=0.896224, z=0.767747)), -0.010922241024672985,
                 1.249703369140633),
    VehicleModel('vehicle.audi.tt', 2.6529807899879163,
                 BoundingBox(Vector3(x=0.000000, y=0.000000, z=0.700000),
                             Vector3(x=2.110247, y=1.004570, z=0.695304)), -0.005904807709157467,
                 1.2471203613281148),
    VehicleModel('vehicle.carlamotors.carlacola', 2.6650390625,
                 BoundingBox(Vector3(x=0.000000, y=0.000000, z=1.230000),
                             Vector3(x=2.599615, y=1.307510, z=1.238425)),
                 -0.00285758962854743,
                 1.4586614990234352),
    VehicleModel('vehicle.citroen.c3', 2.6835025857246966,
                 BoundingBox(Vector3(x=0.000000, y=0.000000, z=0.770000),
                             Vector3(x=1.987674, y=0.931037, z=0.812886)),
                 0.03734729811549187,
                 1.273968505859358),
    VehicleModel('vehicle.dodge_charger.police', 3.0097724778183363,
                 BoundingBox(Vector3(x=0.000000, y=0.000000, z=0.790000),
                             Vector3(x=2.496583, y=1.067964, z=0.789848)), -0.018992232158780098,
                 1.4837268066406182),
    VehicleModel('vehicle.jeep.wrangler_rubicon', 2.307527086106001,
                 BoundingBox(Vector3(x=0.000000, y=0.000000, z=0.940000),
                             Vector3(x=1.934053, y=0.953285, z=0.940000)), 0.004644622560590506,
                 1.2908074951171784),
    VehicleModel('vehicle.nissan.patrol', 2.763547976308006,
                 BoundingBox(Vector3(x=0.000000, y=0.000000, z=0.911131),
                             Vector3(x=2.260000, y=0.960000, z=0.943565)),
                 0.009825591929256916,
                 1.4191088867187602),
    VehicleModel('vehicle.mustang.mustang',
                 2.8861885854672398, BoundingBox(Vector3(x=0.000000, y=0.000000, z=0.660000),
                                                 Vector3(x=2.340000, y=0.940000, z=0.650000)), -0.007357253693044186,
                 1.5617626953124955),
    VehicleModel('vehicle.bmw.isetta',
                 1.5423288329195992, BoundingBox(Vector3(x=0.000000, y=0.000000, z=0.682651),
                                                 Vector3(x=1.101248, y=0.742662, z=0.673248)), 0.002044105436652899,
                 0.682403564453125),
    VehicleModel('vehicle.audi.etron',
                 2.900054053559186, BoundingBox(Vector3(x=0.000000, y=0.000000, z=0.837088),
                                                Vector3(x=2.427029, y=1.004570, z=0.834399)), -0.050091702491045,
                 1.5144543457031148),
    VehicleModel('vehicle.mercedes-benz.coupe',
                 3.289219825531166, BoundingBox(Vector3(x=0.000000, y=0.000000, z=0.680000),
                                                Vector3(x=2.521893, y=1.081499, z=0.820067)), 0.12632092833518982,
                 1.767652587890609),
    VehicleModel('vehicle.bmw.grandtourer',
                 2.9348160710126505, BoundingBox(Vector3(x=0.000000, y=0.000000, z=0.760000),
                                                 Vector3(x=2.319544, y=1.129661, z=0.839089)), 0.0727667585015297,
                 1.4915612792968602),
    VehicleModel('vehicle.toyota.prius',
                 2.819617633400026, BoundingBox(Vector3(x=0.000000, y=0.000000, z=0.710000),
                                                Vector3(x=2.269167, y=1.000565, z=0.768681)), 0.03443630039691925,
                 1.4198394775390568),
    VehicleModel('vehicle.tesla.model3',
                 2.9996323194424814, BoundingBox(Vector3(x=0.060645, y=0.000000, z=0.747342),
                                                 Vector3(x=2.452351, y=1.030329, z=0.739199)), -0.004468612372875214,
                 1.5844549560546852),
    VehicleModel('vehicle.seat.leon',
                 2.7359874402623756, BoundingBox(Vector3(x=0.000000, y=0.000000, z=0.730000),
                                                 Vector3(x=2.103937, y=0.909933, z=0.734730)), -0.006410293281078339,
                 1.315521240234375),
    VehicleModel('vehicle.lincoln.mkz2017',
                 2.8711236779512466, BoundingBox(Vector3(x=0.000000, y=0.000000, z=0.758087),
                                                 Vector3(x=2.454375, y=1.056094, z=0.761592)), -0.011359786614775658,
                 1.4775164794921807),
    VehicleModel('vehicle.volkswagen.t2',
                 2.4436633778147536, BoundingBox(Vector3(x=0.000000, y=0.000000, z=1.040000),
                                                 Vector3(x=2.259523, y=1.038367, z=1.013224)), -0.011650161817669868,
                 1.1672204589843602),
    VehicleModel('vehicle.nissan.micra', 2.4901966364914356,
                 BoundingBox(Vector3(x=0.000000, y=-0.059209, z=0.770000),
                             Vector3(x=1.834424, y=0.928076, z=0.761607)), -0.01624458283185959,
                 1.1744427490234273),
    VehicleModel('vehicle.chevrolet.impala',
                 2.9818149474553275, BoundingBox(Vector3(x=0.000000, y=0.000000, z=0.710000),
                                                 Vector3(x=2.684597, y=1.026368, z=0.707550)), -0.04786643758416176,
                 1.6519848632812568),
    VehicleModel('vehicle.mini.cooperst', 2.5112504763009977,
                 BoundingBox(Vector3(x=0.000000, y=0.000000, z=0.690000),
                             Vector3(x=1.901096, y=0.987017, z=0.737500)), 0.041275788098573685,
                 1.1446356201171852),
]
VEHICLE_BY_TYPE_ID = {v.type_id: v for v in VEHICLES}