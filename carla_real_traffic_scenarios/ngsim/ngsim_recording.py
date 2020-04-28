import logging
import math
import os
import platform
import sys
from os.path import isfile
from typing import List, NamedTuple, Dict

import numpy as np
import pandas as pd
import pygame

from sim2real.carla import ChauffeurCommand, DT
from sim2real.carla.transforms import Transform, Vector2
from sim2real.carla.wrappers.action import KMH_TO_MPS
from sim2real.ngsim_dataset import FRAMES_BEFORE_MANUVEUR, FRAMES_AFTER_MANUVEUR, NGSimDataset, NGSimTimeslot, \
    NGSimDatasets
from sim2real.storage import get_storage
from sim2real.utils.pandas import swap_columns_inplace

LANE_WIDTH_METERS = 3.7
LANE_WIDTH_PIXELS = 24  # pixels / 3.7 m, lane width
METER_TO_PIXELS = LANE_WIDTH_PIXELS / LANE_WIDTH_METERS
PIXELS_TO_METERS = 1. / METER_TO_PIXELS
FOOT_TO_METERS = 0.3048  # metres per foot
X_OFFSET_PIXELS = 470  # horizontal offset (camera 2 leftmost view)
MAX_SPEED = 130

colours = {
    'w': (255, 255, 255),
    'k': (000, 000, 000),
    'r': (255, 000, 000),
    'g': (000, 255, 000),
    'm': (255, 000, 255),
    'b': (000, 000, 255),
    'c': (000, 255, 255),
    'y': (255, 255, 000),
    'gray': (128, 128, 128),
}

LOGGER = logging.getLogger(__name__)


class Point:
    # constructed using a normal tupple
    def __init__(self, point_t=(0, 0)):
        self.x = float(point_t[0])
        self.y = float(point_t[1])

    # define all useful operators
    def __add__(self, other):
        return Point((self.x + other.x, self.y + other.y))

    def __sub__(self, other):
        return Point((self.x - other.x, self.y - other.y))

    def __mul__(self, scalar):
        return Point((self.x * scalar, self.y * scalar))

    def __truediv__(self, scalar):
        return Point((self.x / scalar, self.y / scalar))

    def norm(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    # get back values in original tuple format
    def get(self):
        return self.x, self.y


def draw_dashed_line(surf, color, start_pos, end_pos, width=1, dash_length=10):
    origin = Point(start_pos)
    target = Point(end_pos)
    displacement = target - origin
    length = displacement.norm()
    slope = displacement / length

    for index in range(0, round(length / dash_length), 2):
        start = origin + (slope * index * dash_length)
        end = origin + (slope * (index + 1) * dash_length)
        pygame.draw.line(surf, color, start.get(), end.get(), width)


def draw_text(screen, text, xy, font_size=30, colour=(255, 255, 255), font=None):
    if font is None:
        font = pygame.font.SysFont(None, font_size)
    text = font.render(text, True, colour)
    text_rect = text.get_rect()
    text_rect.left = xy[0]
    text_rect.top = xy[1]
    screen.blit(text, text_rect)


def draw_rect(screen, colour, rect, direction=(1, 0), thickness=0):
    x, y, l, w = rect
    xy = np.array(((x, y - w / 2), (x, y + w / 2), (x + l, y + w / 2), (x + l, y - w / 2)))
    c, s = direction
    rot = np.array(((c, -s), (s, c)))
    xy = (rot @ (xy - (x, y)).T).T + (x, y)
    return pygame.draw.polygon(screen, colour, xy, thickness)


# Car coordinate system, origin under the centre of the rear axis
#
#      ^ y                       (x, y, x., y.)
#      |
#   +--=-------=--+
#   |  | z        |
# -----o-------------->
#   |  |          |    x
#   +--=-------=--+
#      |
#
# Will approximate this as having the rear axis on the back of the car!
#
# Car sizes:
# type    | width [m] | length [m]
# ---------------------------------
# Sedan   |    1.8    |    4.8
# SUV     |    2.0    |    5.3
# Compact |    1.7    |    4.5


assert DT == 0.1, "I80 dataset is sampled with dt=0.1 which conveniently matches our DT. Is it no longer true?"


class Simulator:

    def __init__(self, use_display=True, nb_lanes=4, fps=30):
        self.offset = int(1.5 * LANE_WIDTH_PIXELS)
        self.screen_size = (80 * LANE_WIDTH_PIXELS, nb_lanes * LANE_WIDTH_PIXELS + self.offset + LANE_WIDTH_PIXELS // 2)
        self.fps = fps  # updates per second
        self.nb_lanes = nb_lanes  # total number of lanes
        self.frame = 0  # frame index
        self.lanes = self.build_lanes(nb_lanes)  # create lanes object, list of dicts
        self.env_cars = None  # vehicles list
        self.mean_fps = None
        self.look_ahead = MAX_SPEED * KMH_TO_MPS * METER_TO_PIXELS
        self.user_is_done = None

        self.use_display = use_display
        if self.use_display:  # if display is required
            pygame.init()  # init PyGame
            self.screen = pygame.display.set_mode(self.screen_size)  # set screen size
            self.clock = pygame.time.Clock()  # set up timing
            self.font = {
                20: pygame.font.SysFont(None, 20),
                30: pygame.font.SysFont(None, 30),
            }

    def build_lanes(self, nb_lanes):
        return tuple(
            {'min': self.offset + n * LANE_WIDTH_PIXELS,
             'mid': self.offset + LANE_WIDTH_PIXELS / 2 + n * LANE_WIDTH_PIXELS,
             'max': self.offset + (n + 1) * LANE_WIDTH_PIXELS}
            for n in range(nb_lanes)
        )

    def render(self):
        # measure time elapsed, enforce it to be >= 1/fps
        fps = int(1 / self.clock.tick(self.fps) * 1e3)
        self.mean_fps = 0.9 * self.mean_fps + 0.1 * fps if self.mean_fps is not None else fps

        # clear the screen
        self.screen.fill(colours['k'])

        # draw lanes
        self._draw_lanes(self.screen)

        for v in self.env_cars:
            v.draw(self.screen)

        draw_text(self.screen, f'# cars: {len(self.env_cars)}', (10, 2), font=self.font[30])
        draw_text(self.screen, f'frame #: {self.frame}', (120, 2), font=self.font[30])
        draw_text(self.screen, f'fps: {self.mean_fps:.0f}', (270, 2), font=self.font[30])

        pygame.display.flip()

        # capture the closing window and mouse-button-up event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONUP:
                self._pause()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                self.user_is_done = True

    def _draw_lanes(self, surface, offset=0):
        raise Exception("Abstract method call")

    def _pause(self):
        pause = True
        while pause:
            self.clock.tick(15)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    sys.exit()
                elif e.type == pygame.MOUSEBUTTONUP or e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                    pause = False


class NGSimCar:
    max_a = 40
    max_b = 0.01

    def __init__(self, df, y_offset, screen_w, font=None, kernel=0):
        k = kernel  # running window size
        self.length_m = df.at[df.index[0], 'Vehicle Length'] * FOOT_TO_METERS
        self.width_m = df.at[df.index[0], 'Vehicle Width'] * FOOT_TO_METERS
        self._length = self.length_m * METER_TO_PIXELS
        self._width = self.width_m * METER_TO_PIXELS

        self.id = df.at[df.index[0], 'Vehicle ID']  # extract scalar <'Vehicle ID'> <at> <index[0]>

        x = df['Local X'].rolling(window=k).mean().shift(
            1 - k).values * FOOT_TO_METERS * METER_TO_PIXELS - X_OFFSET_PIXELS - self._length
        y = df['Local Y'].rolling(window=k).mean().shift(1 - k).values * FOOT_TO_METERS * METER_TO_PIXELS + y_offset
        self._max_t = len(x) - np.count_nonzero(np.isnan(x)) - 2  # 2 for computing the acceleration

        self._trajectory = np.column_stack((x, y))
        self._position = self._trajectory[0]
        self._df = df
        self._frame = 0
        # self._direction = np.array((1, 0), np.float)  # assumes horizontal if initially unknown
        self._direction = self._get('init_direction', 0)
        self._speed = self._get('speed', 0)
        self._colour = colours['c']
        self._braked = False
        self.off_screen = self._max_t <= 0
        self._states = list()
        self._states_image = list()
        self._ego_car_image = None
        self._actions = list()
        self.screen_w = screen_w
        if font is not None:
            self._text = self.get_text(self.id, font)
        self._lane_list = df['Lane Identification'].values
        self.collisions_per_frame = 0

    def step(self, action):  # takes also the parameter action = state temporal derivative
        """
        Update current position, given current velocity and acceleration
        """
        # Actions: acceleration (a), steering (b)
        a, b = action

        # State integration
        self._position += self._speed * self._direction * DT

        ortho_direction = np.array((self._direction[1], -self._direction[0]))
        direction_vector = self._direction + ortho_direction * b * self._speed * DT
        self._direction = direction_vector / (np.linalg.norm(direction_vector) + 1e-3)

        self._speed += a * DT

    @staticmethod
    def get_text(n, font):
        text = font.render(str(n), True, colours['b'])
        text_rect = text.get_rect()
        return text, text_rect

    @property
    def front(self):
        return self._position + self._length * self._direction

    @property
    def back(self):
        return self._position

    def __repr__(self) -> str:
        cls = self.__class__
        return f'{cls.__module__}.{cls.__name__}.{self.id}'

    def get_transform(self) -> Transform:
        return Transform(
            Vector2.from_numpy(self.front).to_vector3(0),
            Vector2.from_numpy(self._direction),
        )

    def get_velocity(self) -> Vector2:
        direction = Vector2.from_numpy(self._direction)
        return direction * self._speed

    def draw(self, surface, offset=0):
        """
        Draw current car on screen with a specific colour
        :param surface: PyGame ``Surface`` where to draw
        :param offset: for representation cropping
        """
        x, y = self._position + offset
        rectangle = (int(x), int(y), self._length, self._width)

        d = self._direction

        _r = draw_rect(surface, self._colour, rectangle, d)

        # Drawing vehicle number
        if x < self.front[0]:
            self._text[1].left = x
        else:
            self._text[1].right = x
        self._text[1].top = y - self._width // 2
        surface.blit(self._text[0], self._text[1])

        if self._braked: self._colour = colours['g']
        return _r

    def _get(self, what, k):
        direction_vector = self._trajectory[k + 1] - self._trajectory[k]
        norm = np.linalg.norm(direction_vector)
        if what == 'direction':
            if norm < 1e-6: return self._direction  # if static returns previous direction
            return direction_vector / norm
        if what == 'speed':
            return norm / DT
        if what == 'init_direction':  # valid direction can be computed when speed is non-zero
            t = 1  # check if the car is in motion the next step
            while self._df.at[self._df.index[t], 'Vehicle Velocity'] < 5 and t < self._max_t: t += 1
            # t point to the point in time where speed is > 5
            direction_vector = self._trajectory[t] - self._trajectory[t - 1]
            norm = np.linalg.norm(direction_vector)
            # assert norm > 1e-6, f'norm: {norm} -> too small!'
            if norm < 1e-6:
                print(f'{self} has undefined direction, assuming horizontal')
                return np.array((1, 0), dtype=np.float)
            return direction_vector / norm

    def policy(self):
        self._frame += 1
        self.off_screen = self._frame >= self._max_t

        new_speed = self._get('speed', self._frame)
        a = (new_speed - self._speed) / DT

        ortho_direction = np.array((self._direction[1], -self._direction[0]))
        new_direction = self._get('direction', self._frame)
        b = (new_direction - self._direction).dot(ortho_direction) / (self._speed * DT + 1e-6)

        # From an analysis of the action histograms -> limit a, b to sensible range
        a, b = self.action_clipping(a, b)

        return np.array((a, b))

    def action_clipping(self, a, b):
        max_a = self.max_a
        max_b = self.max_b * min((25 / self._length) ** 2, 1)
        a = a if abs(a) < max_a else np.sign(a) * max_a
        b = b if abs(b) < max_b else np.sign(b) * max_b
        return a, b


class LaneChangeInstant(NamedTuple):
    timeslot: NGSimTimeslot
    frame_start: int
    vehicle_id: int
    lane_from: int  # 1-indexed
    lane_to: int  # 1-indexed

    @property
    def chauffeur_command(self):
        if self.lane_to < self.lane_from:
            return ChauffeurCommand.TURN_LEFT
        elif self.lane_to > self.lane_from:
            return ChauffeurCommand.TURN_RIGHT
        else:
            raise Exception(f"{self.lane_from} != {self.lane_to}")

    @staticmethod
    def from_pandas_row(timeslot: NGSimTimeslot, row):
        return LaneChangeInstant(
            timeslot=timeslot,
            frame_start=int(row['Frame ID']),
            vehicle_id=int(row['Vehicle ID']),
            lane_from=int(row['lane_from']),
            lane_to=int(row['lane_to']),
        )


class NGSimRecording(Simulator):

    def __init__(self, ngsim_dataset: NGSimDataset, x_max_meters=315, use_display=True):
        self._ngsim_dataset = ngsim_dataset

        super().__init__(nb_lanes=6,
                         use_display=use_display)  # dlaczego podajemy inne nb_lanes do base niz tutaj (6 != 7)

        self.screen_size = (85 * LANE_WIDTH_PIXELS, self.nb_lanes * LANE_WIDTH_PIXELS + 5 * LANE_WIDTH_PIXELS)
        if self.use_display:  # if display is required
            self.screen = pygame.display.set_mode(self.screen_size)  # set screen size

        self._df_by_timeslot: Dict[NGSimTimeslot, pd.DataFrame] = {}
        self._init_df(x_max_meters=x_max_meters, x_offset_meters=X_OFFSET_PIXELS * PIXELS_TO_METERS)

        self.vehicles_history_ids = None
        self.nb_lanes = 7
        self.smoothing_window = 15
        self.max_frame = -1

    def _init_df(self, x_max_meters, x_offset_meters):
        data_dir = os.path.join(get_storage(), 'datasets', 'ppuu', 'xy-trajectories')

        if platform.node() == 'adam-jakubowski':  # for faster development
            data_dir = '/home/adam/src/rl/xy-trajectories'

        self.lane_change_instants = []

        for timeslot in self._ngsim_dataset.timeslots:
            file_name = f'{data_dir}/{self._ngsim_dataset.data_dir}/trajectories-{timeslot.file_suffix}.txt'
            assert isfile(file_name), f'{file_name}.{{txt}} not found.'

            LOGGER.info(f'Loading trajectories from {file_name}')
            df = pd.read_csv(file_name, sep=r'\s+', header=None, names=(
                'Vehicle ID',
                'Frame ID',
                'Total Frames',
                'Global Time',
                'Local X',
                'Local Y',
                'Global X',
                'Global Y',
                'Vehicle Length',
                'Vehicle Width',
                'Vehicle Class',
                'Vehicle Velocity',
                'Vehicle Acceleration',
                'Lane Identification',
                'Preceding Vehicle',
                'Following Vehicle',
                'Spacing',
                'Headway'
            ))

            assert self._ngsim_dataset in [NGSimDatasets.I80, NGSimDatasets.US101], "Swapping XY only for I80 and US101"
            swap_columns_inplace(df, 'Local X', 'Local Y')  # in i80 dataset those two are mistakenly swapped

            df = df.drop(columns=['Global X', 'Global Y'])  # unused, possibly broken in I80

            df = df[~df['Vehicle ID'].isin(timeslot.blacklisted_vehicle_ids)]

            # Get valid x coordinate rows
            valid_x = (df['Local X'] * FOOT_TO_METERS - x_offset_meters).between(0, x_max_meters)
            df = df[valid_x]

            self._df_by_timeslot[timeslot] = df

            # Lane change instant calculations:
            df = df.sort_values(by=['Vehicle ID', 'Frame ID'])

            df = df[df['Lane Identification'] <= self._ngsim_dataset.rightmost_lane_id_for_lanechange_scenarios]

            df['lane_from'] = df['Lane Identification']
            df['lane_to'] = df['Lane Identification'].shift(-1)
            # Make sure selected car exists MANEUVEUR_PLAYAHEAD frames ago and FRAMES_BEFORE_MANUVEUR after
            df['vehicle_in_the_past'] = df['Vehicle ID'].shift(FRAMES_BEFORE_MANUVEUR)
            df['vehicle_in_the_future'] = df['Vehicle ID'].shift(-FRAMES_AFTER_MANUVEUR)
            df['x_in_the_past'] = df['Local X'].shift(FRAMES_BEFORE_MANUVEUR)
            df = df[df['Vehicle ID'] == df['vehicle_in_the_past']]
            df = df[df['Vehicle ID'] == df['vehicle_in_the_future']]

            df = df[(df['Vehicle ID'] == df['Vehicle ID'].shift(1)) & (df['lane_from'] != df['lane_to'])]

            self.lane_change_instants.extend(
                LaneChangeInstant.from_pandas_row(timeslot, row) for index, row in df.iterrows()
            )

    def reset(self, timeslot: NGSimTimeslot, frame: int):
        self.frame = frame
        self._timeslot = timeslot
        self.env_cars = list()
        self.mean_fps = None
        self.time_counter = 0

        if self.use_display:
            pygame.display.set_caption(f'Traffic simulator, start from frame {self.frame}')

        self.user_is_done = False
        self.max_frame = max(self._df_by_timeslot[self._timeslot]['Frame ID'])
        self.vehicles_history_ids = set()

    def step(self) -> (List[NGSimCar], bool):
        assert self.frame < self.max_frame

        df = self._df_by_timeslot[self._timeslot]
        now = df['Frame ID'] == self.frame
        vehicles_ids = set(df[now]['Vehicle ID']) - self.vehicles_history_ids

        now_and_on = df['Frame ID'] >= self.frame
        for vehicle_id in vehicles_ids:
            this_vehicle = df['Vehicle ID'] == vehicle_id
            car_df = df[this_vehicle & now_and_on]
            if len(car_df) < self.smoothing_window + 1: continue
            f = self.font[20] if self.use_display else None
            car = NGSimCar(car_df, self.offset, self.screen_size[0], f, self.smoothing_window)
            self.env_cars.append(car)
        self.vehicles_history_ids |= vehicles_ids  # union set operation

        for v in self.env_cars[:]:
            if v.off_screen:
                self.env_cars.remove(v)

        for v in self.env_cars:
            action = v.policy()
            v.step(action)

        self.frame += 1

        return self.env_cars

    def _draw_lanes(self, surface, offset=0):
        slope = 0.035

        lanes = self.lanes  # lanes

        s = surface  # screen
        draw_line = pygame.draw.line  # shortcut
        w = colours['w']  # colour white
        gray = colours['gray']
        sw = self.screen_size[0]  # screen width

        for lane in lanes:
            draw_line(s, gray, (0, lane['min']), (sw, lane['min']), 1)
            # draw_dashed_line(s, colours['r'], (0, lane['mid']), (sw, lane['mid']))

        draw_line(s, w, (0, lanes[0]['min']), (sw, lanes[0]['min']), 3)
        bottom = lanes[-1]['max']
        draw_line(s, w, (0, bottom), (18 * LANE_WIDTH_PIXELS, bottom), 3)
        draw_line(s, w, (0, bottom + 29), (18 * LANE_WIDTH_PIXELS, bottom + 29 - slope * 18 * LANE_WIDTH_PIXELS), 3)
        draw_line(s, gray, (18 * LANE_WIDTH_PIXELS, bottom + 13), (31 * LANE_WIDTH_PIXELS, bottom), 1)
        # draw_line(s, g, (0, bottom + 42), (60 * LANE_W, bottom + 42 - slope * 60 * LANE_W), 1)
        draw_line(s, w, (0, bottom + 53), (60 * LANE_WIDTH_PIXELS, bottom + 53 - slope * 60 * LANE_WIDTH_PIXELS), 3)
        draw_line(s, w, (60 * LANE_WIDTH_PIXELS, bottom + 3), (sw, bottom + 2), 3)

        look_ahead = MAX_SPEED * 1000 / 3600 * METER_TO_PIXELS
        o = self.offset
        draw_line(s, (255, 255, 0), (look_ahead, o), (look_ahead, 9.4 * LANE_WIDTH_PIXELS))
        draw_line(s, (255, 255, 0), (sw - 1.75 * look_ahead, o), (sw - 1.75 * look_ahead, bottom))
        draw_line(s, (255, 255, 0), (sw - 0.75 * look_ahead, o), (sw - 0.75 * look_ahead, bottom), 5)
