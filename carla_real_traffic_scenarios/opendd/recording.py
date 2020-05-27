import sqlite3
from typing import List, Optional

import numpy as np
import pandas as pd
import skimage.transform

from carla_real_traffic_scenarios import DT
from carla_real_traffic_scenarios.opendd.dataset import OpenDDDataset
from carla_real_traffic_scenarios.utils.carla import RealTrafficVehicle, find_best_matching_model
from carla_real_traffic_scenarios.utils.transforms import Transform, Vector3, Vector2


class Utm2CarlaMapper:

    def __init__(self, world_file_params, image_size):
        image_middle = np.array(image_size) // 2
        pix2utm_transform = skimage.transform.AffineTransform(np.array(
            [[world_file_params[0], world_file_params[2], world_file_params[4]],
             [world_file_params[1], world_file_params[3], world_file_params[5]],
             [0, 0, 1]]))
        utm2pix_transform = skimage.transform.AffineTransform(pix2utm_transform._inv_matrix)
        self._affine_transform = utm2pix_transform + \
                                 skimage.transform.AffineTransform(translation=-image_middle) + \
                                 skimage.transform.AffineTransform(scale=1 / np.array(utm2pix_transform.scale))

    def utm2carla(self, transform: Transform):
        utm_position = transform.position.as_numpy()[:2]
        utm_orientation = transform.orientation.as_numpy() + utm_position

        position = self._affine_transform(utm_position.reshape(-1, 2)).squeeze()
        orientation = self._affine_transform(utm_orientation.reshape(-1, 2)).squeeze()
        orientation = orientation - position

        position = Vector2.from_numpy(position).to_vector3(0)
        orientation = Vector2.from_numpy(orientation)

        return Transform(position, orientation)


class OpenDDVehicle:

    def __init__(self, df, transformer) -> None:
        self._df = df

        self.id = int(df.OBJID.iloc[0])
        self.width_m = float(df.WIDTH.iloc[0])
        self.length_m = float(df.LENGTH.iloc[0])
        self._model = find_best_matching_model(self.width_m, self.length_m)

        self._frame = 0
        self._max_frame = len(df)
        self._transformer = transformer

    def step(self):
        self._frame += 1

    @property
    def type_id(self):
        return self._model.type_id

    @property
    def speed_mps(self):
        return self._df.V.iloc[self._frame]

    @property
    def transform(self):
        utm_x = self._df.UTM_X.iloc[self._frame]
        utm_y = self._df.UTM_Y.iloc[self._frame]
        utm_angle_rad = self._df.UTM_ANGLE.iloc[self._frame]
        x, y = np.cos(utm_angle_rad), np.sin(utm_angle_rad)

        position = Vector3(utm_x, utm_y, 0)
        direction = Vector2(x, y)
        return Transform(position, direction)

    @property
    def carla_transform(self):
        carla_transform = self._transformer.utm2carla(self.transform)
        carla_transform = Transform(carla_transform.position.with_z(self._model.z_offset), carla_transform.orientation)
        return carla_transform

    def is_offscreen(self):
        return self._frame >= self._max_frame - 1

    def as_real_traffic_car(self):
        timestamp = self._df.TIMESTAMP.iloc[self._frame]
        debug_string = f'id={self.id} ts={timestamp:0.2f}'
        return RealTrafficVehicle(self.id, self.type_id, self.width_m, self.length_m, self.carla_transform,
                                  self.speed_mps,
                                  debug_string)


class OpenDDRecording():

    def __init__(self, dataset: OpenDDDataset, timedelta_s: float = DT) -> None:
        self._dataset = dataset
        self._env_vehicles = []
        self._vehicles_history_ids = set()
        self._df: Optional[pd.DataFrame] = None
        self._frame = 0
        self._simulation_timedelta_s = timedelta_s

    def reset(self, table_name, frame: int = 0):
        if self._df:
            del self._df

        place_name = table_name.split('_')[0]
        with sqlite3.connect(self._dataset.db_path) as conn:
            df = pd.read_sql(f'select * from {table_name}', conn)

            # create timedelta index from TIMESTAMP column (pd.Grouper uses it)
            df = df.set_index(pd.TimedeltaIndex(df.TIMESTAMP, 's'))
            # group by OBJID and resample TimedeltaIndex to target fps
            freq_ms = int(self._simulation_timedelta_s * 1000)
            grouper = df.groupby([pd.Grouper(freq=f'{freq_ms}ms'), 'OBJID'])
            df = grouper.last()  # take last observation from grouped bins
            self._df = df.reset_index(level=['OBJID'])  # recover OBJID column
            self._timestamps = sorted(set(self._df.TIMESTAMP.values))

        self._frame = frame
        self._env_vehicles = []
        self._vehicles_history_ids = set()

        place_params = self._dataset.places_params[place_name]
        self._transformer = Utm2CarlaMapper(place_params.world_params, place_params.image_size)
        self.place_params = place_params

    def step(self) -> List[RealTrafficVehicle]:
        timestamp = self._timestamps[self._frame]
        df = self._df[self._df.TIMESTAMP == timestamp]

        new_vehicles_ids = set(df.OBJID) - self._vehicles_history_ids
        for vehicle_id in new_vehicles_ids:
            new_vehicle_df = self._df[(self._df.OBJID == vehicle_id) & (self._df.TIMESTAMP >= timestamp)]
            # TODO: check if x/y smoothing is not required (in ngsim dataset there is smoothing in 15 frames wnd)
            self._env_vehicles.append(OpenDDVehicle(new_vehicle_df, self._transformer))
        self._vehicles_history_ids |= new_vehicles_ids

        self._env_vehicles = [v for v in self._env_vehicles if not v.is_offscreen()]
        for v in self._env_vehicles:
            v.step()

        self._frame += 1
        return [v.as_real_traffic_car() for v in self._env_vehicles]

    def close(self):
        pass
