# Multi-UAV Task Allocation
The repository is developed to tackle the dynamic task allocation problem in the Suppression of Enemy Air Defenses missions (SEAD) for heterogeneous multi-UAV systems. To perform the SEAD mission in dynamic environments, a method for decentralized dynamic task allocation based on a Decentralized Parallel Genetic Algorithm (DPGA) is employed. The method parallelizes a genetic algorithm across the UAV swarm and periodically achieves information exchange through UAV-to-UAV (U2U) communication for conflict resolution and further optimization. Based on the received information and the strategies for dynamic task allocation, each UAV can generate appropriate solutions according to the current environment and be able to confront situations involving ad-hoc addition of targets and UAV failures. Moreover, a path-following method is employed to control them to execute the assigned tasks. <br>
The ultimate objective of the repository is to implement real flight operations in outdoor environments. Validation of the developed system is achieved through the repositories located at https://github.com/jerryfungi/Multi-UAV_System_UAVprogram and https://github.com/jerryfungi/GroundControlStation_of_Multi-UAV_Systems.git.

## Usage
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
