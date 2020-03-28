# COVID-19 Simulation

This repository is intended at simulating the spread of the SARS-Cov-2 in a synthetic environment to test out different strategic interventions e.g. social distancing, through the use of reinforcement learning.

### Installation
```
git clone https://github.com/bishwarup307/covid19-simulation.git
cd covid19-simulation
python setup.py build_ext --inplace
```

### Completed:
1. City model
``` python
from environment import City
city = City(grid_size = 200)
```
2. Spawn entities
```python
city = City(200)
city.spawn('Hospital')
city.spawn('Public_place')
city.spawn('School')
```

3. Spawn people
```python
p0 = city.spawn_person(0)
p0.about(details = True)

>> Name: Wendy
>> Age: 53
>> Occupation: FieldJob
>> Work Location: [(92, 114), (50, 120), (75, 124), (79, 133), (7, 101), (31, 36), (158, 45)]
>> Stays at: (112, 40)
>> Immunity: 0.9
>> Hygiene: 0.4216730019073207
>> Infected: False
```

### WIP:

1. Real-time browser streaming of events (requires a `redis-server` running at backend):

```python
city = City(grid_size = 200, stream = {'cache': 'redis'})
```
This opens up a browser window and automatically streams all the events henceforth dynamically

2. Implement an environment `step`: 
Each and every person in the city takes a step to complete an environment step. It's implemented at hour level. Basically it returns a renewed location guided by some set of rules (e.g. for `DeskJob` workers the location remains about same during 9 to 5 etc.).

``` python
p0 = city.spawn_person(0)
print(p0.occupation)
print(p0.home)
print(p0.work)

>> DeskJob
>> (90, 165)
>> [(92, 36)]
```

```python
for time in range(9, 21):
    print(p0.step(time))

>> (92, 36)
>> (91, 37)
>> (92, 36)
>> (91, 37)
>> (92, 38)
>> (93, 37)
>> (92, 36)
>> (91, 37)
>> (92, 38)
>> (109, 165)
>> (110, 164)
>> (111, 163)
```

```python
city.city_map['public_place']['pbl_3']
>> (107, 165, 114, 167)
```