"""
__author__: bishwarup
created: Tuesday, 24th March 2020 11:29:06 pm
"""

from __future__ import print_function, division
import os
from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import random
import string
import sys
import json
import logging
import webbrowser
import operator
import matplotlib.pyplot as plt
from collections import defaultdict
import itertools
import names
from multiprocessing import Process
from utils import calc_area, distance, spawn
from redis import Redis
from app import app


def locate(coords, noise_level=0, center=False):
    if len(coords) == 2:
        if center:
            pos = (coords[0][0] + coords[1][0]) // 2, (coords[0][1] + coords[1][1]) // 2
        else:
            pos = (
                random.randint(coords[0][0], coords[1][0]),
                random.randint(coords[0][1], coords[1][1]),
            )
    elif len(coords) == 4:
        if center:
            pos = (coords[0] + coords[2]) // 2, (coords[1] + coords[3]) // 2
        else:
            pos = (
                random.randint(coords[0], coords[2]),
                random.randint(coords[1], coords[3]),
            )
    else:
        raise NotImplementedError

    if noise_level > 0 and not center:
        x_noise, y_noise = (
            random.choice([-1, 1]) * noise_level,
            random.choice([-1, 1]) * noise_level,
        )
        return pos[0] + x_noise, pos[1] + y_noise
    return pos


# we assume Euclidean distance for simplitcity
# def distance(x1, x2):
#     return np.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2)


# def spawn(shape, grid, fill=1, start=None):
#     done = False
#     found = None
#     while not done:
#         if start is not None:
#             start_x, start_y = start
#         else:
#             start_x, start_y = (
#                 np.random.randint(0, grid.shape[0] - shape[0]),
#                 np.random.randint(0, grid.shape[1] - shape[1]),
#             )
#         # print(start_x, start_y)
#         for i in range(start_x, grid.shape[0] - shape[0]):
#             for j in range(start_y, grid.shape[1] - shape[1]):
#                 if grid[i, j] == 1:
#                     continue
#                 elif np.sum(grid[i: (i + shape[0]), j: (j + shape[1])]) == 0:
#                     done = True
#                     found = i, j
#                     grid[i: (i + shape[0]), j: (j + shape[1])] = fill
#                     break
#                 else:
#                     continue
#             if done:
#                 break
#     x_max, y_max = found[0] + shape[0], found[1] + shape[1]
#     bbox = found, (x_max, y_max)
#     return bbox, grid


class Structure(ABC):
    @abstractproperty
    def shape(self):
        pass

    @abstractproperty
    def fill(self):
        pass

    def spawn(self, city):
        return spawn(self.shape, city, self.fill)


class Office(Structure):
    min_x, min_y = 2, 2
    max_x, max_y = 12, 18
    _fill = 1

    def __init__(self):
        self._shape = (
            random.choice(np.arange(self.min_x, self.max_x)),
            random.choice(np.arange(self.min_y, self.max_y)),
        )
        self.color = (255, 201, 22)

    @property
    def shape(self):
        return self._shape

    @property
    def fill(self):
        return self._fill


class School(Structure):
    min_x, min_y = 2, 2
    max_x, max_y = 8, 8
    _fill = 2

    def __init__(self):
        self._shape = (
            random.choice(np.arange(self.min_x, self.max_x)),
            random.choice(np.arange(self.min_y, self.max_y)),
        )
        self.color = (34, 229, 183)

    @property
    def shape(self):
        return self._shape

    @property
    def fill(self):
        return self._fill


class Hospital(Structure):
    min_x, min_y = 5, 5
    max_x, max_y = 15, 15
    _fill = 3

    def __init__(self):
        self._shape = (
            random.choice(np.arange(self.min_x, self.max_x)),
            random.choice(np.arange(self.min_y, self.max_y)),
        )
        self.color = (243, 43, 73)

    @property
    def shape(self):
        return self._shape

    @property
    def fill(self):
        return self._fill


class PublicPlaces(Structure):
    min_x, min_y = 2, 2
    max_x, max_y = 20, 20
    _fill = 6

    def __init__(self):
        self._shape = (
            random.choice(np.arange(self.min_x, self.max_x)),
            random.choice(np.arange(self.min_y, self.max_y)),
        )
        self.color = (155, 155, 155)

    @property
    def shape(self):
        return self._shape

    @property
    def fill(self):
        return self._fill


class SmallHouse(Structure):
    min_x, min_y = 2, 2
    max_x, max_y = 4, 4
    _fill = 4

    def __init__(self):
        self._shape = (
            random.choice(np.arange(self.min_x, self.max_x)),
            random.choice(np.arange(self.min_y, self.max_y)),
        )
        self.color = (211, 70, 200)

    @property
    def shape(self):
        return self._shape

    @property
    def fill(self):
        return self._fill


class Apartment(Structure):
    min_x, min_y = 6, 6
    max_x, max_y = 10, 10
    _fill = 5

    def __init__(self):
        self._shape = (
            random.choice(np.arange(self.min_x, self.max_x)),
            random.choice(np.arange(self.min_y, self.max_y)),
        )
        self.color = (182, 0, 255)

    @property
    def shape(self):
        return self._shape

    @property
    def fill(self):
        return self._fill


def get_logger(file=None, level=logging.INFO):
    logger = logging.getLogger(__name__)
    handler = logging.FileHandler(file) if file is not None else logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s:%(levelname)s:%(name)s:%(processName)s:%(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class City:
    KEY = "city19"
    structures = [
        "office",
        "school",
        "hospital",
        "public_place",
        "small_house",
        "apartment",
    ]
    COLORS = {
        "hospital": (173, 50, 100),
        "school": (100, 200, 189),
        "office": (254, 166, 0),
        "public_place": (10, 56, 239),
        "apartment": (128, 19, 128),
        "small_house": (182, 129, 207),
        "healthy": (76, 205, 67),
        "infected": (175, 46, 34),
    }

    def __init__(self, grid_size, **kwargs):
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
        self.rgb_grid = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)

        self.spawn_random = kwargs.get("spawn_random", False)
        if self.spawn_random:
            self.n_office = kwargs.get("office", random.randint(10, 100))
            self.n_school = kwargs.get("school", random.randint(5, 25))
            self.n_hospital = kwargs.get("hospital", random.randint(5, 20))
            self.n_public = kwargs.get("public_place", random.randint(10, 20))
            self.n_small_house = kwargs.get("small_house", self.initial_pop * 0.7 / 5)
            self.n_apartments = kwargs.get("apartment", self.initial_pop * 0.3 / 20)
        else:
            self.n_office = kwargs.get("office", 0)
            self.n_school = kwargs.get("school", 0)
            self.n_hospital = kwargs.get("hospital", 0)
            self.n_public = kwargs.get("public_place", 0)
            self.n_small_house = kwargs.get("small_house", 0)
            self.n_apartments = kwargs.get("apartment", 0)

        self.stream = dict()
        self.streaming = False
        stream = kwargs.get("stream", None)
        if stream:
            self.stream["cache"] = stream.get("cache", "redis").lower()
            self.stream["port"] = stream.get("port", 6379)
            self.stream["db"] = stream.get("db", "db0").lower()

        log = kwargs.get("log", True)
        if log is not None:
            if isinstance(log, str):
                self.logger = get_logger(log)
            else:
                self.logger = get_logger()

        if len(self.stream) > 0:
            if not self.stream["cache"] == "redis":
                self.logger.error(
                    "only `redis` is supported as streaming backend for now."
                )
            else:
                self.streaming = True
                self.server = Redis()
                if self.server.get(City.KEY) is not None:
                    self.server.delete(City.KEY)
                self.listener = self.server.pubsub(ignore_subscribe_messages=True)
                self.latency = 1
                self.app = app

        self._locs = defaultdict(lambda: defaultdict(tuple))
        self._occupancy = dict()
        self._gen_ids = []
        # City._suppress_flask_server_out()
        if self.streaming:
            self.process = Process(target=self.app.run)
            self.process.start()
            # $webbrowser.open_new_tab("http://127.0.0.1:5000")
            os.system("google-chrome -incognito http://127.0.0.1:5000")

    @property
    def city_map(self):
        return self._locs

    @staticmethod
    def _get_area(coord):
        return calc_area(coord)
        # return (coord[2] - coord[0]) * (coord[3] - coord[1])

    @staticmethod
    def _suppress_flask_server_out(level=logging.ERROR):
        lgr = logging.getLogger("werkzeug")
        lgr.setLevel(level)

    def _generate_id(self, size=6, chars=string.ascii_lowercase + string.digits):
        id_ = "".join(random.choice(chars) for _ in range(size))
        if id_ not in self._gen_ids:
            return id_
        return self._generate_id()

    def _publish(self, content):
        r = Redis()
        id_ = self._generate_id(8)
        r.append(City.KEY, "\n" + json.dumps(content))

    def _check_if_buildings(self):
        return (self.spawn_random) or (
            self.n_office
            + self.n_school
            + self.n_hospital
            + self.n_public
            + self.n_apartments
            + self.n_small_house
        ) > 0

    def _update_occupancy(self):
        self._occupancy["apartment"] = dict(
            zip(
                list(self.city_map["apartment"].keys()),
                [City._get_area(x) for x in list(self.city_map["apartment"].values())],
            )
        )
        self._occupancy["small_house"] = dict(
            zip(
                list(self.city_map["small_house"].keys()),
                [
                    City._get_area(x)
                    for x in list(self.city_map["small_house"].values())
                ],
            )
        )

    def _get_open_coords(self):
        flat_pos = np.arange(self.grid_size * self.grid_size)[self.grid.flatten() == 0]
        xs = flat_pos // self.grid_size
        ys = flat_pos % self.grid_size
        return list(zip(xs, ys))

    def _update_locs(self, loc):
        """
        Arguments:
            loc {tuple} -- (`loc_type`, `loc_id`, `loc_coords`)
        """
        self._locs[loc[0]].update({loc[1]: loc[2]})

    def spawn(self, building_type, **kwargs):
        building_type = building_type.lower()
        assert (
            building_type in self.structures
        ), f"`building_type` must be one of {', '.join(self.structures)}"
        if building_type == "office":
            struct = Office()
        elif building_type == "public_place":
            struct = PublicPlaces()
        elif building_type == "hospital":
            struct = Hospital()
        elif building_type == "school":
            struct = School()
        elif building_type == "apartment":
            struct = Apartment()
        elif building_type == "small_house":
            struct = SmallHouse()
        else:
            raise NotImplementedError

        loc, self.grid = struct.spawn(self.grid)
        self._update_locs(
            (building_type, f"{building_type}_{self._generate_id()}", loc)
        )

        self._update_occupancy()

        if self.streaming:
            box = (loc[0], loc[1], loc[2] - loc[0], loc[3] - loc[1])
            self._publish(
                {"shape": box, "color": City.COLORS[building_type], "id": building_type}
            )

    def spawn_buildings(self):
        if self._check_if_buildings():

            for i in range(self.n_public):
                public_place = PublicPlaces()
                loc, self.grid = public_place.spawn(self.grid)
                self.rgb_grid[self.grid == public_place.fill] = public_place.color
                self._update_locs(("public_place", f"pbl_{i}", loc))

            for i in range(self.n_office):
                office = Office()
                loc, self.grid = office.spawn(self.grid)
                self.rgb_grid[self.grid == office.fill] = office.color
                self._update_locs(("office", f"office_{i}", loc))

            for i in range(self.n_school):
                school = School()
                loc, self.grid = school.spawn(self.grid)
                self.rgb_grid[self.grid == school.fill] = school.color
                self._update_locs(("school", f"school_{i}", loc))

            for i in range(self.n_hospital):
                hospital = Hospital()
                loc, self.grid = hospital.spawn(self.grid)
                self.rgb_grid[self.grid == hospital.fill] = hospital.color
                self._update_locs(("hospital", f"hospital_{i}", loc))

            for i in range(self.n_apartments):
                apt = Apartment()
                loc, self.grid = apt.spawn(self.grid)
                self.rgb_grid[self.grid == apt.fill] = apt.color
                self._update_locs(("apartment", f"apartment_{i}", loc))

            for i in range(self.n_small_house):
                smh = SmallHouse()
                loc, self.grid = smh.spawn(self.grid)
                self.rgb_grid[self.grid == smh.fill] = smh.color
                self._update_locs(("small_house", f"small_house{i}", loc))

            self._update_occupancy()

    def destroy(self):
        try:
            self.process.terminate()
            self.process.join()
        except Exception as exc:
            raise exc

    def draw_city(self, figsize=(8, 8)):
        plt.figure(figsize=figsize)
        plt.imshow(self.rgb_grid)
        plt.show()

    def spawn_person(self, id_):
        occupation = np.random.choice(
            ["DeskJob", "FieldJob", "Student", "Retired"], p=[0.15, 0.5, 0.25, 0.1]
        )

        if occupation == "DeskJob":
            home_type = np.random.choice(["apartment", "small_house"], p=(0.8, 0.2))
        elif occupation == "Student":
            home_type = np.random.choice(["apartment", "small_house"], p=(0.5, 0.5))
        elif occupation == "FieldJob":
            home_type = np.random.choice(["apartment", "small_house"], p=(0.3, 0.7))
        elif occupation == "Retired":
            home_type = random.choice(["apartment", "small_house"])
        else:
            raise NotImplementedError

        home = list(
            sorted(
                self._occupancy[home_type].items(),
                key=operator.itemgetter(1),
                reverse=True,
            )
        )[0][0]

        self._occupancy[home_type][home] = self._occupancy[home_type][home] - 1

        # home = np.argmin(
        #     [City._get_area(x) for x in list(self.city_map[home_type].values())]
        # )
        # home = "apt_" + str(home) if home_type == "Apartment" else "smh_" + str(home)
        home = (
            locate(self.city_map["apartment"][home], center=True)
            if home_type == "apartment"
            else locate(self.city_map["small_house"][home], center=True)
        )

        if occupation == "DeskJob":
            offices = self.city_map["office"]
            work = "office_" + str(
                np.argmin([City._get_area(x) for x in list(offices.values())])
            )
            work = [locate(self.city_map["office"][work])]
        elif occupation == "Student":
            schools = self.city_map["school"]
            work = "school_" + str(
                np.argmin([City._get_area(x) for x in list(schools.values())])
            )
            work = [locate(self.city_map["school"][work])]
        elif occupation == "FieldJob":
            n_work = random.choice(np.arange(5, 10))
            work = []
            for _ in range(n_work):
                open_coords = self._get_open_coords()
                work.append(random.choice(open_coords))
        elif occupation == "Retired":
            work = home
        else:
            raise NotImplementedError

        person = Person(
            id_,
            work,
            home,
            work_type=occupation,
            home_type=home_type,
            city_map=self.city_map,
            boundary=self.grid_size,
        )
        return person


class Occupation(ABC):
    @abstractproperty
    # def travel(self):
    #     pass
    # @abstractproperty
    # def meet(self):
    #     pass
    # @abstractproperty
    # def working_hours(self):
    #     pass
    @abstractmethod
    def step(self):
        pass


class Student(Occupation):
    def __init__(self):
        self.exposure = 30

    def step(self, pos, time):
        if time <= 9:
            return pos
        elif time >= 9 and time <= 18:
            x = pos[0] + 2 * np.random.uniform()
            y = pos[1] + 2 * np.random.uniform()
            return x, y
        else:
            return None


class DeskJob(Occupation):
    def __init__(self):
        # self._travel = 2
        # self._working_hours = 8
        self.exposure = 20

    # @property
    # def travel(self):
    #     return self._travel

    # @property
    # def meet(self):
    #     return self._meet

    # @property
    # def working_hours(self):
    #     return self._working_hours

    def step(self, pos, time):
        if (time >= 9) and (time <= 18):
            return pos
        else:
            return None


class FieldJob(Occupation):
    def __init__(self):
        # self._travel = random.choice(np.arange(5, 10))
        # self._working_hours = random.choice(np.arange(5, 10))
        self.exposure = np.random.randint(30, 150)

    # @property
    # def travel(self):
    #     return self._travel

    # @property
    # def meet(self):
    #     return self.travel * np.clip(5 * np.random.randn() + 5, 0, 50)

    # @property
    # def working_hours(self):
    #     return self._working_hours

    def step(self, pos, time):
        if time < 9:
            x = pos[0] + 4 * np.random.uniform()
            y = pos[1] + 4 * np.random.uniform()
            return x, y
        elif time >= 9 and time <= 20:
            x = pos[0] + 10 * np.random.uniform()
            y = pos[1] + 10 * np.random.uniform()
            return x, y
        else:
            x = pos[0] + 2 * np.random.uniform()
            y = pos[1] + 2 * np.random.uniform()
        return x, y


class Person:
    def __init__(self, id_, work, home, **kwargs):
        # self.age = int(np.clip(5 * np.random.randn() + 30, 0, 90))
        # self._time = 0
        # self.occupation = np.random.choice(['DeskJob', 'FieldJob', 'Student'], p = [0.15, 0.5, 0.35])
        # self.is_compromised = np.random.choice([0, 1], p=[0.2, 0.8])
        # self.immunity = np.clip(np.random.randn(1) + 1, 0, 1)
        # self.precaution = np.random.randn()
        self.id_ = id_
        self.home = home
        self.work = work
        self.name = names.get_first_name()
        self.infected = kwargs.get("infected", False)
        self.occupation = kwargs.get("work_type", None)
        self.home_type = kwargs.get("home_type", None)
        self.city_map = kwargs.get("city_map", None)

        self.boundary = kwargs.get("boundary", 200)
        if self.occupation is not None:
            if self.occupation == "Student":
                self.age = random.randint(5, 18)
            elif self.occupation == "Retired":
                self.age = random.randint(60, 90)
            else:
                self.age = random.randint(18, 60)
        else:
            self.age = int(np.clip(5 * np.random.randn() + 30, 0, 90))
        self.immunity = (
            np.clip(0.5 * np.random.randn() + 0.5, 0.1, 0.9)
            if self.age < 60
            else np.clip(0.4 * np.random.randn() + 0.2, 0.05, 0.7)
        )
        self.hygiene = np.clip(0.5 * np.random.randn() + 0.6, 0.1, 0.9)
        self.pos = self.home
        # self.steps = 0
        self.lock = 0
        # self._freezed = self.lock > 0

    @property
    def freezed(self):
        return self.lock > 0

    def freeze(self, lock_period=1):
        self.lock = lock_period + 1

    def reset_work(self):
        if self.occupation == "FieldJob":
            n_work = random.choice(np.arange(5, 10))
            self.work = []
            for _ in range(n_work):
                self.work.append(
                    (
                        random.choice(np.arange(self.boundary)),
                        random.choice(np.arange(self.boundary)),
                    )
                )

    def about(self, details=False):
        print(f"Name: {self.name}")
        print(f"Age: {self.age}")
        print(f"Occupation: {self.occupation}")
        print(f"Work Location: {self.work}")
        print(f"Stays at: {self.home}")
        if details:
            print(f"Immunity: {self.immunity}")
            print(f"Hygiene: {self.hygiene}")
            print(f"Infected: {self.infected}")

    def step(self, time):
        if not self.freezed:

            if self.occupation == "Retired":
                if (time >= 10) and (time <= 17):
                    if random.random() < 0.8:
                        self.pos = (
                            self.pos[0] + random.choice(np.arange(-10, 10)),
                            self.pos[1] + random.choice(np.arange(-10, 10)),
                        )
                        return self.pos
                    else:
                        pbl = np.argmin(
                            [
                                distance(self.home, x[:2])
                                for x in list(self.city_map["public_place"].values())
                            ]
                        )
                        self.pos = locate(
                            self.city_map["public_place"]["pbl_" + str(pbl)],
                            noise_level=10,
                        )
                        self.freeze(3)
                elif time < 10:
                    self.pos = self.home
                else:
                    self.pos = self.home
                    self.freeze(99)

            elif self.occupation == "DeskJob":
                if (time < 9) or (time >= 22):
                    self.pos = self.home
                elif time == 9:
                    self.pos = self.work[0]
                    self.freeze(8)
                elif time >= 17:
                    if random.random() <= 0.5:
                        visit = random.choice(
                            list(self.city_map["public_place"].keys())
                        )
                        self.pos = locate(self.city_map["public_place"][visit])
                        self.freeze(3)
                    else:
                        self.pos = self.home
                        self.freeze(99)

            elif self.occupation == "FieldJob":
                if time < 8:
                    self.pos = self.home
                elif (time >= 8) and (time <= 20):
                    self.pos = random.choice(self.work)
                    self.pos = (
                        self.pos[0] + random.choice(np.arange(-5, 5)),
                        self.pos[1] + random.choice(np.arange(-5, 5)),
                    )
                    if random.random() <= 0.5:
                        self.freeze(1)
                elif time > 20:
                    self.pos = self.home
                    self.freeze(99)

            elif self.occupation == "Student":
                if time < 8:
                    self.pos = self.home
                elif (time >= 8) and (time < 18):
                    self.pos = self.work[0]
                    if random.random() > 0.7:
                        self.pos = (
                            self.pos[0] + random.choice(np.arange(-3, 3)),
                            self.pos[1] + random.choice(np.arange(-3, 3)),
                        )
                elif (time >= 18) and (time <= 21):
                    if random.random() <= 0.5:
                        visit = random.choice(
                            list(self.city_map["public_place"].keys())
                        )
                        self.pos = locate(self.city_map["public_place"][visit])
                        self.freeze(3)
                    else:
                        self.pos = self.home
                        self.freeze(99)

        else:
            self.pos = (
                self.pos[0] + random.choice([-1, 1]),
                self.pos[1] + random.choice([-1, 1]),
            )

        self.lock = np.clip(self.lock - 1, 0, 99)
        if time == 23:
            self.lock = 0
        return self.pos


if __name__ == "__main__":
    city = City(
        200,
        100,
        school=3,
        office=5,
        hospital=2,
        apartment=30,
        smallhouse=50,
        public_places=10,
    )
    city._spawn_buildings()

    # person = city.spawn_person()
    # print(f"Age: {person.age}")
    # print(f"Home Type: {person.home_type}")
    # print(f"Home: {person.home}")
    # print(f"Occupation: {person.work_type}")
    # print(f"Work: {person.work}")
    city.draw_city()

    # for k, v in city.locations.items():
    #     print(k, dict(v))
    #     print("****")
    # city.draw_city()

    # grid = np.zeros((100, 100), dtype=np.uint8)
    # office = Office()
    # found, grid = office.spawn(grid)
    # print(tuple(itertools.chain(*found)))
    # print(found)
    # print(grid.sum())
    # print(locate(found))
    # print(sys.getsizeof(office))
    # school = School()
    # hospital = Hospital()
    # grid = np.zeros((100, 100), dtype=np.uint8)
    # _, grid = office.spawn(grid)
    # _, grid = school.spawn(grid)
    # _, grid = hospital.spawn(grid)

    # # print(found)
    # print(f"offices: {(grid == 1).sum()}")
    # print(f"schools: {(grid == 2).sum()}")
    # print(f"hospitals: {(grid == 3).sum()}")
