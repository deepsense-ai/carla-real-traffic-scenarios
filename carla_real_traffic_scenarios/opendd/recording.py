import sqlite3
from typing import List, Optional

import more_itertools
import numpy as np
import pandas as pd
import skimage.transform

from carla_real_traffic_scenarios import DT
from carla_real_traffic_scenarios.opendd.dataset import OpenDDDataset, Place
from carla_real_traffic_scenarios.utils.carla import RealTrafficVehicle, find_best_matching_model
from carla_real_traffic_scenarios.utils.transforms import Transform, Vector3, Vector2


class Utm2CarlaMapper:

    def __init__(self, world_file_params, image_size):
        image_middle = np.array(image_size) // 2
        pix2utm_transform = skimage.transform.AffineTransform(np.array(
            [[world_file_params[0], world_file_params[2], world_file_params[4]],
             [world_file_params[1], world_file_params[3], world_file_params[5]],
             [0, 0, 1]]))
        self.pix2utm_transformer = pix2utm_transform

        utm2pix_transform = skimage.transform.AffineTransform(pix2utm_transform._inv_matrix)
        self.utm2pix_transformer = utm2pix_transform

        self.utm2carla_transformer = utm2pix_transform + \
                                     skimage.transform.AffineTransform(translation=-image_middle) + \
                                     skimage.transform.AffineTransform(scale=1 / np.array(utm2pix_transform.scale))

    def utm2pix(self, transform: Transform):
        return self._transform_with_convert(transform, self.utm2pix_transformer)

    def pix2utm(self, transform: Transform):
        return self._transform_with_convert(transform, self.pix2utm_transformer)

    def utm2carla(self, transform: Transform):
        return self._transform_with_convert(transform, self.utm2carla_transformer)

    def _transform_with_convert(self, transform: Transform, transformer: skimage.transform.AffineTransform):
        position = transform.position.as_numpy()[:2]
        position = position.reshape(-1, 2)
        orientation = transform.orientation.as_numpy()
        orientation = orientation.reshape(-1, 2)

        position, orientation = self.transform(position, orientation, transformer)

        position = Vector2.from_numpy(position.squeeze()).to_vector3(0)
        orientation = Vector2.from_numpy(orientation.squeeze())
        return Transform(position, orientation)

    def transform(self, positions: np.ndarray, orientations: np.ndarray,
                  transformer: skimage.transform.AffineTransform):
        orientations = positions + orientations
        positions = transformer(positions)
        orientations = transformer(orientations)
        orientations = orientations - positions
        return positions, orientations


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

        self.trajectory_utm = self._extract_trajectory_utm()
        self.trajectory_carla = self._map_trajectory_to_carla(self.trajectory_utm)

    def step(self):
        self._frame += 1

    @property
    def type_id(self):
        return self._model.type_id

    @property
    def speed_mps(self):
        return self._df.V.iloc[self._frame]

    @property
    def velocity(self):
        return (self.transform_carla.orientation * self.speed_mps).to_vector3(0)

    @property
    def transform_utm(self):
        return self.trajectory_utm[self._frame]

    @property
    def transform_carla(self):
        return self.trajectory_carla[self._frame]

    @property
    def has_finished(self) -> bool:
        return self._frame >= self._max_frame - 1

    def as_real_traffic_car(self):
        timestamp = self._df.TIMESTAMP.iloc[self._frame]
        debug_string = f'id={self.id} fm={self._frame} ts={timestamp:0.2f}'
        return RealTrafficVehicle(self.id, self.type_id, timestamp,
                                  self.width_m, self.length_m, self.transform_carla,
                                  self.speed_mps,
                                  debug_string)

    def _extract_trajectory_utm(self) -> List[Transform]:
        trajectory = self._df[['UTM_X', 'UTM_Y', 'UTM_ANGLE']].values
        return [Transform(Vector3(x, y, 0), Vector2(np.cos(angle), np.sin(angle))) for x, y, angle in trajectory]

    def _map_trajectory_to_carla(self, trajectory_utm) -> List[Transform]:
        trajectory_carla = []
        for transform_utm in trajectory_utm:
            transform_carla = self._transformer.utm2carla(transform_utm)
            transform_carla = \
                Transform(transform_carla.position.with_z(self._model.z_offset), transform_carla.orientation)
            trajectory_carla.append(transform_carla)
        return trajectory_carla


MIN_EPISODE_LENGTH_STEPS = 10 / DT

class OpenDDRecording():

    def __init__(self, dataset: OpenDDDataset, timedelta_s: float = DT) -> None:
        self._dataset = dataset
        self._env_vehicles = {}
        self._df: Optional[pd.DataFrame] = None
        self._frame = 0
        self._timedelta_s = timedelta_s
        self._timestamps = []
        self._session_name: Optional[str] = None

        self._transformer: Optional[Utm2CarlaMapper] = None

    def reset(self, session_name, frame: Optional[int] = 0):
        if self._df is not None:
            del self._df

        self._session_name = session_name
        with sqlite3.connect(self._dataset.db_path) as conn:
            df = pd.read_sql(f'select * from {session_name}', conn)
            df = df[~df.CLASS.str.contains('Pedestrian|Bicycle')]  # for now do not extract pedestrians and bicycles

            # create timedelta index from TIMESTAMP column (pd.Grouper uses it)
            df = df.set_index(pd.TimedeltaIndex(df.TIMESTAMP, 's'))
            # group by OBJID and resample TimedeltaIndex to target fps
            freq_ms = int(self._timedelta_s * 1000)
            grouper = df.groupby([pd.Grouper(freq=f'{freq_ms}ms'), 'OBJID'])
            df = grouper.first()  # take last observation from grouped bins
            df = df.reset_index(level=['OBJID'])  # recover OBJID column
            df['TIMESTAMP'] = df.index.to_series().dt.total_seconds()

            self._timestamps = np.arange(df.TIMESTAMP.min(),
                                         df.TIMESTAMP.max() + self._timedelta_s,
                                         self._timedelta_s)
            self._df = df

        if frame is None:
            frame = 0
            # random choice timestamp but pay attention that there shall be at least single vehicle
            if len(self._timestamps) > MIN_EPISODE_LENGTH_STEPS:
                ntries = 10
                while ntries:
                    tmp_frame = np.random.randint(0, len(self._timestamps) - MIN_EPISODE_LENGTH_STEPS)
                    timestamp_s = self._timestamps[tmp_frame]
                    vehicles_current_ids = self._df[self._df.TIMESTAMP == timestamp_s].OBJID.to_list()
                    if vehicles_current_ids:
                        frame = tmp_frame
                        break
                    ntries -= 1
        self._frame = frame
        self._env_vehicles = {}

        self._transformer = Utm2CarlaMapper(self.place.world_params, self.place.image_size)

    def step(self) -> List[RealTrafficVehicle]:
        timestamp_s = self._timestamps[self._frame]
        vehicles_current_ids = self._df[self._df.TIMESTAMP == timestamp_s].OBJID.to_list()

        for vehicle_id in vehicles_current_ids:
            if vehicle_id not in self._env_vehicles:
                # TODO: check if x/y smoothing is not required (in ngsim dataset there is smoothing in 15 frames wnd)
                new_vehicle_df = self._df[(self._df.OBJID == vehicle_id) & (self._df.TIMESTAMP >= timestamp_s)]
                self._env_vehicles[vehicle_id] = OpenDDVehicle(new_vehicle_df, self._transformer)

        self._env_vehicles = {k: v for k, v in self._env_vehicles.items() if not v.has_finished}

        real_traffic_vehicles = [v.as_real_traffic_car() for v in self._env_vehicles.values()]
        if real_traffic_vehicles:
            if len(real_traffic_vehicles) > 1:
                assert all([
                    np.isclose(v1.timestamp_s, v2.timestamp_s)
                    for v1, v2 in more_itertools.windowed(real_traffic_vehicles, 2)
                ]), (
                    self._session_name,
                    [v.debug for v in real_traffic_vehicles],
                )
            assert np.isclose(real_traffic_vehicles[0].timestamp_s, timestamp_s), \
                (real_traffic_vehicles[0].timestamp_s, timestamp_s)

        self._frame += 1
        for v in self._env_vehicles.values():
            v.step()

        return real_traffic_vehicles

    def close(self):
        pass

    @property
    def place(self) -> Place:
        place_name = self._session_name.split('_')[0]
        return self._dataset.places[place_name]

    @property
    def session_name(self) -> str:
        return self._session_name

    @property
    def timestamp_s(self) -> float:
        return self._timestamps[self._frame]

    @property
    def transformer(self):
        return self._transformer

    def get_df_by_objid(self, ego_id):
        return self._df[self._df.OBJID == ego_id]
