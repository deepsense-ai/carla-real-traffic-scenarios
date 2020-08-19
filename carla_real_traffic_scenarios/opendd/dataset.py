import sqlite3
from pathlib import Path
from typing import NamedTuple, Tuple, Dict, Union, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
import skimage.transform

from carla_real_traffic_scenarios.utils.transforms import Vector2

_BLACKLISTED_SESSION_NAMES_DUE_TO_WRONG_GEOREFERENCE = [
    'rdb6_2DJI_0006', 'rdb6_2DJI_0007', 'rdb6_2DJI_0008', 'rdb6_2DJI_0009', 'rdb6_DJI_0016', 'rdb3_M1DJI_0021'
]
BLACKLISTED_SESSION_NAMES = _BLACKLISTED_SESSION_NAMES_DUE_TO_WRONG_GEOREFERENCE

_ROUNDABOUTS_TOPOLOGIES = {
    'rdb1': {
        'roundabout_center_utm': Vector2(619304.556058351, 5809147.733704892),  # (52.419519, 10.754369)
        'map_center_utm': Vector2(619301.201, 5809149.760),  # (1920, 1080),
        'roads_utm': [  # (entry_utm, exit_utm)
            (Vector2(619323.465, 5809153.332), Vector2(619324.167, 5809145.383)),  # ((2622, 894), (2670, 1146))
            (Vector2(619292.150, 5809163.147), Vector2(619299.362, 5809166.923)),  # ((1587, 681), (1806, 537))
            (Vector2(619296.386, 5809130.642), Vector2(619289.733, 5809135.251)),  # ((1827, 1707), (1599, 1581))
            (Vector2(619312.708, 5809131.905), None)  # ((2346, 1614), None)
        ],
        'correction_transform': skimage.transform.AffineTransform(np.array(
            [[0.99996232, -0.02487794, -0.27454128],
             [0.02757512, 1.00178269, 0.80360617],
             [0., 0., 1.]]
        ))
    },
    'rdb2': {
        'roundabout_center_utm': Vector2(618082.795, 5805733.190),  # (1896, 1023),
        'map_center_utm': Vector2(618080.465, 5805732.301),  # (1920, 1080),
        'roads_utm': [  # (entry_utm, exit_utm)
            (Vector2(618068.770, 5805715.815), Vector2(618063.439, 5805721.421)),  # ((2349, 1359), (2208, 1494))
            (Vector2(618097.317, 5805715.819), Vector2(618088.719, 5805711.901)),  # ((2334, 654), (2439, 864))
            (Vector2(618097.793, 5805750.661), Vector2(618103.576, 5805743.643)),  # ((1440, 663), (1617, 516))
            (Vector2(618067.654, 5805750.105), Vector2(618075.653, 5805754.386)),  # ((1470, 1407), (1356, 1212))
            (Vector2(618067.654, 5805750.105), Vector2(618075.653, 5805754.386)),  # ((1470, 1407), (1356, 1212))
        ],
        'correction_transform': skimage.transform.AffineTransform(np.array(
            [[1.00012827, -0.02409896, -0.74671188],
             [0.02559905, 1.00138109, 0.91278798],
             [0., 0., 1.]]
        ))
    },
    'rdb3': {
        'roundabout_center_utm': Vector2(618136.801, 5806279.151),  # (2484, 1215),
        'map_center_utm': Vector2(618151.801, 5806283.277),  # (1920, 1080),
        'roads_utm': [  # (entry_utm, exit_utm)
            (Vector2(618121.182, 5806271.077), Vector2(618119.930, 5806275.946)),  # ((3138, 1242), (3093, 1410))
            (Vector2(618145.261, 5806261.260), Vector2(618138.078, 5806259.628)),  # ((2523, 531), (2787, 606))
            (Vector2(618153.691, 5806287.755), Vector2(618155.962, 5806279.979)),  # ((1779, 1182), (1842, 909))
            (Vector2(618121.460, 5806291.793), Vector2(618127.530, 5806296.676)),  # ((2763, 1860), (2478, 1902))
        ],
        'correction_transform': skimage.transform.AffineTransform(np.array(
            [[1.00023874, -0.02393488, -0.94744426],
             [0.02418768, 0.99811481, 0.99248746],
             [0., 0., 1.]]
        ))
    },
    'rdb4': {
        'roundabout_center_utm': Vector2(624451.746, 5809525.809),  # (1881, 954),
        'map_center_utm': Vector2(624446.967, 5809523.881),  # (1920, 1080),
        'roads_utm': [  # (entry_utm, exit_utm)
            (Vector2(624456.002, 5809501.784), Vector2(624450.510, 5809501.336)),  # ((2499, 897), (2499, 1038))
            (Vector2(624474.967, 5809530.897), Vector2(624475.470, 5809524.904)),  # ((1800, 351), (1953, 351))
            (Vector2(624444.706, 5809549.370), Vector2(624450.081, 5809549.808)),  # ((1269, 1083), (1269, 945))
            (Vector2(624427.862, 5809521.613), Vector2(624427.398, 5809527.135)),  # ((1938, 1572), (1797, 1572))
        ],
        'correction_transform': skimage.transform.AffineTransform(np.array(
            [[1.00001208, -0.02528081, -0.28851676],
             [0.02511984, 0.99966216, 0.66167869],
             [0., 0., 1.]]
        ))
    },
    'rdb5': {
        'roundabout_center_utm': Vector2(617383.221, 5806791.758),  # (2139, 1188),
        'map_center_utm': Vector2(617381.021, 5806798.145),  # (1920, 1080),
        'roads_utm': [  # (entry_utm, exit_utm)
            (Vector2(617395.259, 5806776.203), Vector2(617387.709, 5806772.706)),  # ((2844, 1281), (2739, 1563))
            (Vector2(617399.114, 5806804.625), Vector2(617402.784, 5806797.365)),  # ((2220, 453), (2499, 546))
            (Vector2(617371.940, 5806808.550), Vector2(617379.373, 5806811.929)),  # ((1422, 1044), (1527, 768))
            (Vector2(617368.100, 5806781.204), Vector2(617364.000, 5806788.267)),  # ((2019, 1844), (1734, 1767))
        ],
        'correction_transform': skimage.transform.AffineTransform(np.array(
            [[0.9999156, -0.02381287, -0.38847143],
             [0.023269, 0.99917217, 0.29227672],
             [0., 0., 1.]]
        ))
    },
    'rdb6': {
        'roundabout_center_utm': Vector2(674606.262, 5407008.742),  # (2019, 1146),
        'map_center_utm': Vector2(674609.071, 5407005.899),  # (1920, 1080),
        'roads_utm': [  # (entry_utm, exit_utm)
            (Vector2(674584.776, 5407014.296), Vector2(674588.797, 5407022.220)),  # ((2670, 1179), (2607, 1434))
            (Vector2(674607.453, 5406986.053), Vector2(674598.449, 5406987.494)),  # ((1836, 492), (2103, 480))
            (Vector2(674628.216, 5407003.400), Vector2(674623.837, 5406994.829)),  # ((1356, 1122), (1425, 846))
            (Vector2(674606.370, 5407030.237), Vector2(674615.197, 5407028.935)),  # ((2157, 1773), (1896, 1788))
        ],
        'correction_transform': skimage.transform.AffineTransform(np.array(
            [[0.99955446, -0.03083389, -0.69345472],
             [0.03073315, 0.99923909, 0.0890304],
             [0., 0., 1., ]]
        ))
    },
    'rdb7': {
        'roundabout_center_utm': Vector2(673956.833, 5410629.680),  # (1995, 1053),
        'map_center_utm': Vector2(673954.441, 5410629.990),  # (1920, 1080),
        'roads_utm': [  # (entry_utm, exit_utm)
            (Vector2(673978.811, 5410628.785), Vector2(673976.680, 5410619.544)),  # ((2655, 747), (2730, 1053))
            (Vector2(673949.422, 5410650.680), Vector2(673959.035, 5410651.543)),  # ((1464, 543), (1734, 372))
            (Vector2(673934.246, 5410630.581), Vector2(673937.553, 5410639.728)),  # ((1317, 1368), (1278, 1047))
            (Vector2(673963.999, 5410608.197), Vector2(673955.035, 5410607.610)),  # ((2526, 1581), (2271, 1734))
        ],
        'correction_transform': skimage.transform.AffineTransform(np.array(
            [[0.99968292, -0.03095198, 0.00113838],
             [0.02923595, 0.99971446, -0.06023742],
             [0., 0., 1.]]

        ))
    },
}


class Place(NamedTuple):
    name: str
    image_size: Tuple[int, int]
    image_path: str
    world_params: np.ndarray
    roundabout_center_utm: Vector2
    map_center_utm: Vector2
    roads_utm: List[Tuple[Optional[Vector2], Optional[Vector2]]]
    correction_transform: skimage.transform.ProjectiveTransform


class OpenDDDataset:

    def __init__(self, dataset_dir: Union[str, Path]):
        dataset_dir = Path(dataset_dir)
        self.dataset_dir = dataset_dir.as_posix()
        self.db_path = (dataset_dir / 'rdb1to7.sqlite').as_posix()
        self.session_names = self._fetch_session_names()  # TODO: divide on TRAIN and TEST
        self.places: Dict[str, Place] = self._fetch_places(dataset_dir)

    def _fetch_session_names(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT tbl_name FROM sqlite_master WHERE type='table'")
            session_names = [row[0] for row in cursor.fetchall()]
            session_names = [n for n in session_names if n not in BLACKLISTED_SESSION_NAMES]
            return session_names

    def _fetch_places(self, dataset_dir):
        georeferenced_images_dir = dataset_dir / 'image_georeferenced'

        places = {}
        for image_path in georeferenced_images_dir.glob('*.jpg'):
            with Image.open(image_path.as_posix()) as img:
                image_size = img.size

            name = image_path.stem
            world_file_path = image_path.parent / f'{name}.tfw'
            world_params = np.loadtxt(world_file_path.as_posix())

            place_topology = _ROUNDABOUTS_TOPOLOGIES[name]
            places[name] = Place(name, image_size, image_path.as_posix(), world_params,
                                 **place_topology)
        return places

    def get_session_df(self, session_name: str):
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql(f'select * from {session_name}', conn)
