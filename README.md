# Multi-UAV Task Allocation

## Introduction
The repository is developed to tackle the dynamic task allocation problem in the Suppression of Enemy Air Defenses missions (SEAD) for heterogeneous multi-UAV systems. To perform the SEAD mission in dynamic environments, a method for decentralized dynamic task allocation based on a Decentralized Parallel Genetic Algorithm (DPGA) is employed. The method parallelizes a genetic algorithm across the UAV swarm and periodically achieves information exchange through UAV-to-UAV (U2U) communication for conflict resolution and further optimization. Based on the received information and the strategies for dynamic task allocation, each UAV can generate appropriate solutions according to the current environment and be able to confront situations involving ad-hoc addition of targets and UAV failures. Moreover, a path-following method is employed to control them to execute the assigned tasks.The ultimate objective of the repository is to implement real flight operations in outdoor environments. Validation of the developed system is achieved through the repositories located at [Multi-UAV_System_UAVprogram
](https://github.com/jerryfungi/Multi-UAV_System_UAVprogram) and [GroundControlStation_of_Multi-UAV_Systems
](https://github.com/jerryfungi/GroundControlStation_of_Multi-UAV_Systems.git).

## OS
Due to the dubins package, the program is only worked on Linux. <br>
There are two ways to operate the repository on Windows:
1. Modify the dubins package and path setting according to the **[link](https://blog.csdn.net/qq_28266955/article/details/80332909)**.
2. Download Ubuntu on Windows and set up the required environments to run the scripts.

## Dependencies
Install the necessary dependencies by running.
```bash
pip install -r requirements.txt
```

## Usage
### DPGA for dynamic SEAD missions
```python
from decentralized_GA_SEAD import *

targets_sites = [[3100, 2600], [500, 2400]]
uav_id = [1, 2, 3]
uav_type = [1, 2, 3]
cruise_speed = [70, 80, 90]
turning_radii = [200, 250, 300]
initial_states = [[700, 1200, -np.pi], [1500, 700, np.pi / 2], [3600, 1000, np.pi / 3]]
base_locations = [[2500, 4500, np.pi / 2] for _ in range(3)]
dynamic_SEAD_mission = DynamicSEADMissionSimulator(targets_sites, uav_id, uav_type, cruise_speed, turning_radii, initial_states, base_locations)
dynamic_SEAD_mission.start_simulation(realtime_plot=True, unknown_targets=[[600, 2850]], uav_failure=[False, False, 55])
```
#### Result

### GA for static SEAD missions
```python
from GA_SEAD_process import *

targets_sites = [[500, 1500], [2000, 4500], [3000, 1500]]
uavs = [[1, 2, 3],  # UAV ID
        [1, 2, 3],  # UAV type
        [70, 80, 90],  # Cruise speed
        [200, 250, 300],  # Minimum turning radii
        [[700, 1200, -np.pi], [1500, 700, np.pi / 2], [3600, 1000, np.pi / 3]],  # initial states of UAVs
        [[2500, 4500, np.pi / 2] for _ in range(3)],  # Base positions of UAVs
        [],  # ignore
        [],  # ignore
        [],  # tasks completed
        []]  # new targets

population_size = 100
iteration = 100
mission = GA_SEAD(targets_sites, population_size)
solution, fitness, ga_population, convergence = mission.run_GA(iteration, uavs)
mission.plot_result(solution, convergence)
```
#### Result

