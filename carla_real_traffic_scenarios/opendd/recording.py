import hashlib
import random
import sqlite3
from typing import List, Optional

import more_itertools
import numpy as np
import pandas as pd
import scipy.spatial
import skimage.transform

from carla_real_traffic_scenarios import DT
from carla_real_traffic_scenarios.ngsim import DatasetMode
from carla_real_traffic_scenarios.opendd.dataset import OpenDDDataset, Place
from carla_real_traffic_scenarios.utils.carla import RealTrafficVehicle, find_best_matching_model
from carla_real_traffic_scenarios.utils.transforms import Transform, Vector3, Vector2


def extract_utm_trajectory_from_df(df) -> List[Transform]:
    trajectory = df[['UTM_X', 'UTM_Y', 'UTM_ANGLE']].values
    return [Transform(Vector3(x, y, 0), Vector2(np.cos(angle), np.sin(angle))) for x, y, angle in trajectory]


class Utm2CarlaMapper:

    def __init__(self, place: Place):
        image_middle = np.array(place.image_size) // 2
        pix2utm_transform = skimage.transform.AffineTransform(np.array(
            [[place.world_params[0], place.world_params[2], place.world_params[4]],
             [place.world_params[1], place.world_params[3], place.world_params[5]],
             [0, 0, 1]]))
        self.pix2utm_transformer = pix2utm_transform

        utm2pix_transform = skimage.transform.AffineTransform(pix2utm_transform._inv_matrix)
        self.utm2pix_transformer = utm2pix_transform

        map_center_utm = np.array(place.map_center_utm.as_numpy()[:2])
        reflect_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype='float32')  # reflect over Y axis
        self.utm2carla_transformer = skimage.transform.AffineTransform(translation=-map_center_utm) + \
                                     skimage.transform.AffineTransform(matrix=reflect_matrix) + \
                                     place.correction_transform

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
        self._transformer = transformer

        self._max_frame = len(df)
        self.trajectory_utm = extract_utm_trajectory_from_df(self._df)
        self.trajectory_carla = self._map_trajectory_to_carla(self.trajectory_utm)

    def set_end_of_trajectory_timestamp(self, timestamp_end_s):
        df = self._df
        df = df[df.TIMESTAMP < timestamp_end_s]
        self._max_frame = len(df)
        self.trajectory_utm = extract_utm_trajectory_from_df(df)
        self.trajectory_carla = self._map_trajectory_to_carla(self.trajectory_utm)
        self._df = df

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

    def _map_trajectory_to_carla(self, trajectory_utm) -> List[Transform]:
        trajectory_carla = []
        for transform_utm in trajectory_utm:
            transform_carla = self._transformer.utm2carla(transform_utm)
            transform_carla = \
                Transform(transform_carla.position.with_z(self._model.z_offset), transform_carla.orientation)
            trajectory_carla.append(transform_carla)
        return trajectory_carla


MIN_EPISODE_LENGTH_STEPS = 10 / DT


def _resample_df(df, target_timedelta_s):
    # create timedelta index from TIMESTAMP column (pd.Grouper uses it)
    df = df.set_index(pd.TimedeltaIndex(df.TIMESTAMP, 's'))
    # group by OBJID and resample TimedeltaIndex to target fps
    freq_ms = int(target_timedelta_s * 1000)
    grouper = df.groupby([pd.Grouper(freq=f'{freq_ms}ms'), 'OBJID'])
    df = grouper.first()  # take last observation from grouped bins
    df = df.reset_index(level=['OBJID'])  # recover OBJID column
    df['TIMESTAMP'] = df.index.to_series().dt.total_seconds()
    return df


def _find_ego_vehicle_with_time_frame(place, session_df, ego_id=None):
    all_objids = list(set(session_df.OBJID.to_list()))
    explicit_ego_id = ego_id is not None
    while True:
        ego_id = ego_id if explicit_ego_id else random.choice(all_objids)
        obj_df = session_df[session_df.OBJID == ego_id]
        start_idx, stop_idx = _trim_trajectory_utm_to_entry_end_exit(place, obj_df)
        if not explicit_ego_id and (start_idx is None or stop_idx is None or start_idx >= stop_idx):
            continue

        timestamp_start_s = obj_df.iloc[start_idx].TIMESTAMP if start_idx is not None else None
        timestamp_end_s = obj_df.iloc[stop_idx].TIMESTAMP if stop_idx is not None else None
        return ego_id, timestamp_start_s, timestamp_end_s


def _trim_trajectory_utm_to_entry_end_exit(place, obj_df):
    exits_utm = np.array([exit.as_numpy() if exit else np.zeros(2) for entry, exit in place.roads_utm])
    entries_utm = np.array([entry.as_numpy() if exit else np.zeros(2) for entry, exit in place.roads_utm])
    trajectory_utm = obj_df[['UTM_X', 'UTM_Y']].values

    dm_entries = scipy.spatial.distance_matrix(entries_utm, trajectory_utm)
    entries_distances_m = np.min(dm_entries, axis=1)
    nearest_entry_idx = np.argmin(entries_distances_m)  # idx of nearest entry
    # trajectory idx where vehicle pass nearest roundabout entry
    trajectory_start_idx = np.argmin(dm_entries[nearest_entry_idx])
    min_distance_from_nearest_entry = dm_entries[nearest_entry_idx][trajectory_start_idx]

    MAX_DISTANCE_FROM_WP_M = 2
    PRE_ENTRY_DISTANCE_M = 20
    # ensure that it passes entry not more than MAX_DISTANCE_FROM_WP_M
    if min_distance_from_nearest_entry > MAX_DISTANCE_FROM_WP_M:
        trajectory_start_idx = None
    elif trajectory_start_idx > 0:
        # take 1st index from part of trajectory distanced not more than PRE_ENTRY_DISTANCE_M
        trajectory_start_idx = np.where(
            dm_entries[nearest_entry_idx][:trajectory_start_idx] < PRE_ENTRY_DISTANCE_M
        )[0][0]

    dm_exits = scipy.spatial.distance_matrix(exits_utm, trajectory_utm)
    exit_distances_m = np.min(dm_exits, axis=1)
    nearest_exit_idx = np.argmin(exit_distances_m)
    trajectory_end_idx = np.argmin(dm_exits[nearest_exit_idx])
    min_distance_from_nearest_exit = dm_exits[nearest_exit_idx][trajectory_end_idx]
    # ensure that it passes exit not more than MAX_DISTANCE_FROM_WP_M
    if min_distance_from_nearest_exit > MAX_DISTANCE_FROM_WP_M:
        trajectory_end_idx = None
    else:
        trajectory_end_idx = trajectory_end_idx + np.where(
            dm_exits[nearest_exit_idx][trajectory_end_idx:] < PRE_ENTRY_DISTANCE_M
        )[0][-1]

    return trajectory_start_idx, trajectory_end_idx


def _determine_split(session_name, ego_id, start, stop) -> DatasetMode:
    split_frac = 0.8
    start, stop = int(round(start, 0)), int(round(stop, 0))
    hash_num = int(hashlib.sha1(f'{session_name},{ego_id},{start},{stop}'.encode('utf-8')).hexdigest(), 16)
    if (hash_num % 100) / 100 < split_frac:
        return DatasetMode.TRAIN
    else:
        return DatasetMode.VALIDATION


class OpenDDRecording():

    def __init__(self, *, dataset: OpenDDDataset, timedelta_s: float = DT,
                 dataset_mode: DatasetMode = DatasetMode.TRAIN) -> None:
        self._dataset = dataset
        self._dataset_mode = dataset_mode
        self._env_vehicles = {}
        self._df: Optional[pd.DataFrame] = None
        self._frame = 0
        self._timedelta_s = timedelta_s
        self._timestamps = []
        self._session_name: Optional[str] = None

        self._transformer: Optional[Utm2CarlaMapper] = None

    def reset(self, session_name, seed=None):
        if self._df is not None:
            del self._df

        self._session_name = session_name
        with sqlite3.connect(self._dataset.db_path) as conn:
            df = pd.read_sql(f'select * from {session_name}', conn)
            # for now do not extract pedestrians, bicycles and trailers
            df = df[~df.CLASS.str.contains('Pedestrian|Bicycle|Trailer')]
            df = _resample_df(df, self._timedelta_s)
            self._timestamps = np.arange(df.TIMESTAMP.min(),
                                         df.TIMESTAMP.max() + self._timedelta_s,
                                         self._timedelta_s)
            self._df = df

        # search for train/validation roundabout pass
        dataset_mode = None
        if seed is not None:
            random.seed(seed)
        while dataset_mode != self._dataset_mode:
            ego_id, timestamp_start_s, timestamp_end_s = _find_ego_vehicle_with_time_frame(self.place, self._df)
            dataset_mode = _determine_split(session_name, ego_id, timestamp_start_s, timestamp_end_s)

        self._frame = np.where(np.isclose(self._timestamps, timestamp_start_s, 0.0001))[0][0] + 1
        self._env_vehicles = {}

        self._transformer = Utm2CarlaMapper(self.place)
        return ego_id, timestamp_start_s, timestamp_end_s

    def step(self) -> List[RealTrafficVehicle]:
        timestamp_s = self._timestamps[self._frame]
        vehicles_current_ids = self._df[
            np.isclose(self._df.TIMESTAMP, timestamp_s)
        ].OBJID.to_list()

        for vehicle_id in vehicles_current_ids:
            if vehicle_id not in self._env_vehicles:
                # TODO: check if x/y smoothing is not required (in ngsim dataset there is smoothing in 15 frames wnd)
                new_vehicle_df = self._df[
                    (self._df.OBJID == vehicle_id) &
                    ((self._df.TIMESTAMP >= timestamp_s) | np.isclose(self._df.TIMESTAMP, timestamp_s))
                ]
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

    @property
    def has_finished(self):
        return self._frame >= len(self._timestamps) - 1
