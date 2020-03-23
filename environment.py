from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import random
import sys
import cv2
import operator
import matplotlib.pyplot as plt
from collections import defaultdict
import itertools
import names


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
def distance(x1, x2):
    return np.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2)


def spawn(shape, grid, fill=1, start=None):
    done = False
    found = None
    while not done:
        if start is not None:
            start_x, start_y = start
        else:
            start_x, start_y = (
                np.random.randint(0, grid.shape[0] - shape[0]),
                np.random.randint(0, grid.shape[1] - shape[1]),
            )
        # print(start_x, start_y)
        for i in range(start_x, grid.shape[0] - shape[0]):
            for j in range(start_y, grid.shape[1] - shape[1]):
                if grid[i, j] == 1:
                    continue
                elif np.sum(grid[i : (i + shape[0]), j : (j + shape[1])]) == 0:
                    done = True
                    found = i, j
                    grid[i : (i + shape[0]), j : (j + shape[1])] = fill
                    break
                else:
                    continue
            if done:
                break
    x_max, y_max = found[0] + shape[0], found[1] + shape[1]
    bbox = found, (x_max, y_max)
    return bbox, grid


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


class City:
    def __init__(self, grid_size, initial_pop_size, **kwargs):
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
        self.rgb_grid = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
        self.initial_pop = initial_pop_size
        self.n_office = kwargs.get("office", random.randint(10, 100))
        self.n_school = kwargs.get("school", random.randint(5, 25))
        self.n_hospital = kwargs.get("hospital", random.randint(5, 20))
        self.n_public = kwargs.get("public_places", random.randint(10, 20))
        self.n_small_house = kwargs.get("smallhouse", self.initial_pop * 0.7 / 5)
        self.n_apartments = kwargs.get("apartment", self.initial_pop * 0.3 / 20)
        self._locs = defaultdict(lambda: defaultdict(tuple))
        self._occupancy = dict()

    @property
    def city_map(self):
        return self._locs

    def _update_locs(self, loc):
        """
        Arguments:
            loc {tuple} -- (`loc_type`, `loc_id`, `loc_coords`)
        """
        self._locs[loc[0]].update({loc[1]: tuple(itertools.chain(*loc[2]))})

    @staticmethod
    def _get_area(coord):
        return (coord[2] - coord[0]) * (coord[3] - coord[1])

    def _get_open_coords(self):
        flat_pos = np.arange(self.grid_size * self.grid_size)[self.grid.flatten() == 0]
        xs = flat_pos // self.grid_size
        ys = flat_pos % self.grid_size
        return list(zip(xs, ys))

    def _spawn_buildings(self,):

        for i in range(self.n_public):
            public_place = PublicPlaces()
            loc, self.grid = public_place.spawn(self.grid)
            self.rgb_grid[self.grid == public_place.fill] = public_place.color
            self._update_locs(("PublicPlace", f"pbl_{i}", loc))

        for i in range(self.n_office):
            office = Office()
            loc, self.grid = office.spawn(self.grid)
            self.rgb_grid[self.grid == office.fill] = office.color
            self._update_locs(("Office", f"office_{i}", loc))

        for i in range(self.n_school):
            school = School()
            loc, self.grid = school.spawn(self.grid)
            self.rgb_grid[self.grid == school.fill] = school.color
            self._update_locs(("School", f"school_{i}", loc))

        for i in range(self.n_hospital):
            hospital = Hospital()
            loc, self.grid = hospital.spawn(self.grid)
            self.rgb_grid[self.grid == hospital.fill] = hospital.color
            self._update_locs(("Hospital", f"hospital_{i}", loc))

        for i in range(self.n_apartments):
            apt = Apartment()
            loc, self.grid = apt.spawn(self.grid)
            self.rgb_grid[self.grid == apt.fill] = apt.color
            self._update_locs(("Apartment", f"apt_{i}", loc))

        for i in range(self.n_small_house):
            smh = SmallHouse()
            loc, self.grid = smh.spawn(self.grid)
            self.rgb_grid[self.grid == smh.fill] = smh.color
            self._update_locs(("SmallHouse", f"smh_{i}", loc))

        self._occupancy["Apartment"] = dict(
            zip(
                list(self.city_map["Apartment"].keys()),
                [City._get_area(x) for x in list(self.city_map["Apartment"].values())],
            )
        )
        self._occupancy["SmallHouse"] = dict(
            zip(
                list(self.city_map["SmallHouse"].keys()),
                [City._get_area(x) for x in list(self.city_map["SmallHouse"].values())],
            )
        )

    def draw_city(self, figsize=(8, 8)):
        plt.figure(figsize=figsize)
        plt.imshow(self.rgb_grid)
        plt.show()

    def spawn_person(self):
        occupation = np.random.choice(
            ["DeskJob", "FieldJob", "Student", "Retired"], p=[0.15, 0.5, 0.25, 0.1]
        )

        if occupation == "DeskJob":
            home_type = np.random.choice(["Apartment", "SmallHouse"], p=(0.8, 0.2))
        elif occupation == "Student":
            home_type = np.random.choice(["Apartment", "SmallHouse"], p=(0.5, 0.5))
        elif occupation == "FieldJob":
            home_type = np.random.choice(["Apartment", "SmallHouse"], p=(0.3, 0.7))
        elif occupation == "Retired":
            home_type = random.choice(["Apartment", "SmallHouse"])
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
            locate(self.city_map["Apartment"][home], center=True)
            if home_type == "Apartment"
            else locate(self.city_map["SmallHouse"][home], center=True)
        )

        if occupation == "DeskJob":
            offices = self.city_map["Office"]
            work = "office_" + str(
                np.argmin([City._get_area(x) for x in list(offices.values())])
            )
            work = [locate(self.city_map["Office"][work])]
        elif occupation == "Student":
            schools = self.city_map["School"]
            work = "school_" + str(
                np.argmin([City._get_area(x) for x in list(schools.values())])
            )
            work = [locate(self.city_map["School"][work])]
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
    def __init__(self, work, home, **kwargs):
        # self.age = int(np.clip(5 * np.random.randn() + 30, 0, 90))
        # self._time = 0
        # self.occupation = np.random.choice(['DeskJob', 'FieldJob', 'Student'], p = [0.15, 0.5, 0.35])
        # self.is_compromised = np.random.choice([0, 1], p=[0.2, 0.8])
        # self.immunity = np.clip(np.random.randn(1) + 1, 0, 1)
        # self.precaution = np.random.randn()
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
                                for x in list(self.city_map["PublicPlace"].values())
                            ]
                        )
                        self.pos = locate(
                            self.city_map["PublicPlace"]["pbl_" + str(pbl)],
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
                        visit = random.choice(list(self.city_map["PublicPlace"].keys()))
                        self.pos = locate(self.city_map["PublicPlace"][visit])
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
                        visit = random.choice(list(self.city_map["PublicPlace"].keys()))
                        self.pos = locate(self.city_map["PublicPlace"][visit])
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
