# Multi-UAV Task Allocation in SEAD missions
Task allocation alogrithms for multiple UAVs of SEAD missions and VRP.

## Introduction
The repository is developed to tackle the dynamic task allocation problem in the Suppression of Enemy Air Defenses missions (SEAD) for heterogeneous multi-UAV systems. To perform the SEAD mission in dynamic environments, a method for decentralized dynamic task allocation based on a Decentralized Parallel Genetic Algorithm (DPGA) is employed. The method parallelizes a genetic algorithm across the UAV swarm and periodically achieves information exchange through UAV-to-UAV (U2U) communication for conflict resolution and further optimization. Based on the received information and the strategies for dynamic task allocation, each UAV can generate appropriate solutions according to the current environment and be able to confront situations involving ad-hoc addition of targets and UAV failures. Moreover, a path-following method is employed to control them to execute the assigned tasks.The ultimate objective of the repository is to implement real flight operations in outdoor environments. Validation of the developed system is achieved through the repositories located at [Multi-UAV_System_UAVprogram
](https://github.com/jerryfungi/Multi-UAV_System_UAVprogram) and [GroundControlStation_of_Multi-UAV_Systems
](https://github.com/jerryfungi/GroundControlStation_of_Multi-UAV_Systems.git).

## OS
Due to the dubins package, the program is only worked on Linux. <br>
There are two ways to operate the repository on Windows:
*  Modify the dubins package and path setting according to the **[link](https://blog.csdn.net/qq_28266955/article/details/80332909)**.
*  Download Ubuntu on Windows and set up the required environments to run the scripts.

## Usage
* Clone this repo
```bash
git clone https://github.com/jerryfungi/Multi-UAV_Task_Allocation_SEADmission.git
```
* Install the necessary dependencies by running.
```bash
pip install -r requirements.txt
```
* Execute the python scripts

### DPGA for dynamic SEAD missions
* Python script: decentralized_GA_SEAD.py
* Method: Decentralized Parallel Genetic Algorithm


### GA for static SEAD missions
* Python script: GA_SEAD_process.py
* Method: Genetic Algorithm



### GA for VRP
* Python script: GA_VRP.py
* Method: Genetic Algorithm


### PSO for VRP
* Python script: PSO_VRP.py
* Method: Particle Swarm Opimization

