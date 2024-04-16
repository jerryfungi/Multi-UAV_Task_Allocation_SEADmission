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

## DPGA for dynamic SEAD missions
<video src="https://github.com/jerryfungi/Multi-UAV_Task_Allocation_SEADmission/assets/112320576/c3778494-ec80-4bc5-82ff-ece5e33b580d" autoplay=True width=70%>
</video>

<img src="https://github.com/jerryfungi/Multi-UAV_Task_Allocation_SEADmission/assets/112320576/425810ae-d80f-469f-b353-63d0f6d3e1c9" width=60%>
</img>

<img src="https://github.com/jerryfungi/Multi-UAV_Task_Allocation_SEADmission/assets/112320576/82675790-8392-4f3c-964b-fed58bfc5366" width=70%>
</img>

* Execution
```bash
python3 decentralized_GA_SEAD.py
```

## GA for static SEAD missions
<img src="https://github.com/jerryfungi/Multi-UAV_Task_Allocation_SEADmission/assets/112320576/0385ba58-b2ff-4945-befe-bc676a604565" width=80%>
</img>

* Execution
```bash
python3 GA_SEAD_process.py
```

## GA for VRP
<video src="https://github.com/jerryfungi/Multi-UAV_Task_Allocation_SEADmission/assets/112320576/0c97d951-4cc8-4ce7-a1a8-d0c368ed9d39"  autoplay=True width=70%>
</video>

* Execution
```bash
python3 GA_VRP.py
```

## PSO for VRP
<img src="https://github.com/jerryfungi/Multi-UAV_Task_Allocation_SEADmission/assets/112320576/395883f9-563f-454b-9f44-f318169b5f14" width=75%>
</img>

* Execution
```bash
python3 PSO_VRP.py
```

## Reference
Considering that the proposed thesis has not been published yet, the related research is presented below for reference.
* C. Xia, L. Yongtai, Y. Liyuan, and Q. Lijie, "Cooperative task assignment and track planning for multi-UAV attack mobile targets," Journal of Intelligent & Robotic Systems, vol. 100, pp. 1383-1400, 2020.
* G. Xu, T. Long, Z. Wang, and L. Liu, "Target-bundled genetic algorithm for multi-unmanned aerial vehicle cooperative task assignment considering precedence constraints," Proceedings of the Institution of Mechanical Engineers, Part G: Journal of Aerospace Engineering, vol. 234, no. 3, pp. 760-773, 2020.
* Z. Jia, J. Yu, X. Ai, X. Xu, and D. Yang, "Cooperative multiple task assignment problem with stochastic velocities and time windows for heterogeneous unmanned aerial vehicles using a genetic algorithm," Aerospace Science and Technology, vol. 76, pp. 112-125, 2018.
* R. Patel, E. Rudnick-Cohen, S. Azarm, M. Otte, H. Xu, and J. W. Herrmann, "Decentralized task allocation in multi-agent systems using a decentralized genetic algorithm," in 2020 IEEE International Conference on Robotics and Automation (ICRA), 2020: IEEE, pp. 3770-3776. 
* Z. Qin and Y. Yi, "Particle Swarm Optimization Algorithm with Real Number Encoding for Vehicle Routing Problem," 2011 International Conference of Information Technology, Computer Engineering and Management Sciences, Nanjing, China, 2011, pp. 118-121, doi: 10.1109/ICM.2011.360.
