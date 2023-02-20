import random
import time
import math
import numpy as np
import copy
from matplotlib import pyplot as plt
import dubins


class PSO_SEAD(object):
    def __init__(self, targets):
        self.targets = targets
        # global message
        self.uav_id = []
        self.uav_type = []
        self.uav_velocity = []
        self.uav_Rmin = []
        self.uav_position = []
        self.depots = []
        self.uavType_for_missions = []
        self.tasks_status = [3 for _ in range(len(self.targets))]
        self.cost_matrix = []
        self.discrete_heading = [_ for _ in range(0, 36)]
        # PSO parameters
        self.particle_num = 100
        self.w = 0.75
        self.c1 = 2
        self.c2 = 2
        self.lambda_1 = 0
        self.lambda_2 = 1e4
        self.Pbest = [[0 for _ in range(self.particle_num)], [0 for _ in range(self.particle_num)]]
        self.Gbest = [0, 0]

    def fitness_evaluate(self, particles):
        uav_num = len(self.uav_id)
        particle_len = len(particles[0][0])
        for index, particle in enumerate(particles):
            sorted_target = sorted(range(len(particle[0])), key=lambda u: particle[0][u])  # target sequence
            sorted_pos = [math.floor(pos) for pos in sorted(particle[0])]  # assign agent
            task_type_count = [1 for _ in range(len(self.targets))]
            cost = [0 for _ in range(uav_num)]
            pre_site, pre_heading = [0 for _ in range(uav_num)], [0 for _ in range(uav_num)]
            task_sequence_time = [[] for _ in range(uav_num)]  # time
            time_list = []
            for j in range(particle_len):
                assign_uav = sorted_pos[j]-1
                assign_target = sorted_target[j] // 3 + 1
                assign_heading = int(particle[1][j])
                cost[assign_uav] += self.cost_matrix[assign_uav][pre_site[assign_uav]][pre_heading[assign_uav]][assign_target][assign_heading]
                task_sequence_time[assign_uav].append([assign_target, task_type_count[assign_target-1],
                                                       cost[assign_uav] / self.uav_velocity[assign_uav]])
                pre_site[assign_uav], pre_heading[assign_uav] = assign_target, assign_heading
                task_type_count[assign_target-1] += 1
            for j in range(uav_num):
                cost[j] += self.cost_matrix[j][pre_site[j]][pre_heading[j]][0][0]
            for sequence in task_sequence_time:
                time_list.extend(sequence)
            time_list.sort()
            # time sequence penalty
            penalty, j = 0, 0
            for task_num in self.tasks_status:
                if task_num >= 2:
                    for k in range(1, task_num):
                        penalty += max(0, time_list[j + k - 1][2] - time_list[j + k][2])
                j += task_num
            fitness_value = 1 / (np.max(np.divide(cost, self.uav_velocity)) + self.lambda_1 * np.sum(cost)
                                 + self.lambda_2 * penalty)
            self.pbest_update(fitness_value, index, particle[:2])
            self.gbest_update(fitness_value, particle[:2])

    def initiate_particles(self):
        def generate_particle():
            uav_num = len(self.uav_id)
            particle = [[] for _ in range(4)]
            for i in range(sum(self.tasks_status)):
                particle[0].append(random.uniform(1, uav_num+1))  # position [1, m]
                # particle[1].append(round(random.uniform(-(self.uav_num+1), self.uav_num+1), 5))
                particle[2].append(random.uniform(-1, 1))  # velocity [-m, m]
                particle[1].append(random.uniform(0, 36))  # position(heading) [0, 2pi]
                particle[3].append(random.uniform(-1, 1))
            return particle
        return [generate_particle() for _ in range(self.particle_num)]

    def pbest_update(self, new_dot_fitness, dot_num, particle_pos):
        if self.Pbest[0][dot_num] < new_dot_fitness:
            self.Pbest[0][dot_num] = new_dot_fitness
            self.Pbest[1][dot_num] = particle_pos

    def gbest_update(self, new_max_fitness, particle_pos):
        if self.Gbest[0] < new_max_fitness:
            self.Gbest[0] = new_max_fitness
            self.Gbest[1] = particle_pos

    def particle_improve(self, particles):
        uav_num = len(self.uav_id)
        for i, particle in enumerate(particles):
            particle[2] = list(np.clip(self.w * np.array(particle[2]) +
                                       self.c1 * random.random() * np.subtract(self.Pbest[1][i][0], particle[0]) +
                                       self.c2 * random.random() * np.subtract(self.Gbest[1][0], particle[0]), -1, 1))
            particle[3] = list(np.clip(self.w * np.array(particle[3]) +
                                       self.c1 * random.random() * np.subtract(self.Pbest[1][i][0], particle[1]) +
                                       self.c2 * random.random() * np.subtract(self.Gbest[1][0], particle[1]), -1, 1))
            particle[0] = list(np.clip(np.add(particle[0], particle[2]), 1, uav_num+1-1e-5))
            particle[1] = list(np.clip(np.add(particle[1], particle[3]), 0, 36))

    def information_setting(self, uavs, population):
        regenerate, build_graph, empty = False, False, False
        terminated_tasks, new_target = uavs[8], uavs[9]
        print(uavs[7])
        if terminated_tasks:  # check terminated tasks
            terminated_tasks = sorted(terminated_tasks, key=lambda u: u[1])
            print(terminated_tasks)
            for task in terminated_tasks:
                if self.tasks_status[task[0]-1] == 3-task[1]+1:
                    self.tasks_status[task[0]-1] -= 1
            regenerate = True
            for chromosome in uavs[7]:
                for task in terminated_tasks:
                    del_site = chromosome[1].index(task[0])
                    for row in chromosome:
                        del row[del_site]
                chromosome[0] = [_ for _ in range(1, len(chromosome[0])+1)]
        if new_target:  # check new targets
            self.targets.apend(new_target)
            self.tasks_status.append(3)
            regenerate = True
        if not set(self.uav_id) == set(uavs[0]):  # check agents
            regenerate = True
            build_graph = True
        if population:
            population.extend([elite for elite in uavs[7]])
        if sum(self.tasks_status) == 0:
            empty = True
        # clear the information
        self.uav_id = uavs[0]
        self.uav_type = uavs[1]
        self.uav_velocity = uavs[2]
        self.uav_Rmin = uavs[3]
        self.uav_position = uavs[4]
        self.depots = uavs[5]
        self.uavType_for_missions = [[] for _ in range(3)]
        # cost graph --------------------------------------------------------------------------------------------
        if build_graph:
            self.cost_matrix = [[[[[0 for a in range(len(self.discrete_heading))] for b in range(len(self.targets) + 1)]
                                  for c in range(len(self.discrete_heading))] for d in range(len(self.targets) + 1)]
                                for u in range(len(self.uav_id))]
            for a in range(1, len(self.targets) + 1):
                for b in self.discrete_heading:
                    for c in range(a, len(self.targets) + 1):
                        for d in self.discrete_heading:
                            source_point = self.targets[a - 1] + [b * 10 * np.pi / 180]
                            end_point = self.targets[c - 1] + [d * 10 * np.pi / 180]
                            if source_point == end_point:
                                end_point[-1] += 1e5
                            for u in range(len(self.uav_id)):
                                distance = dubins.shortest_path(source_point, end_point, self.uav_Rmin[u]).path_length()
                                self.cost_matrix[u][a][b][c][d] = distance
                                self.cost_matrix[u][c][d][a][b] = distance
        # update real time information in graph
        for a in range(1, len(self.targets) + 1):
            for b in self.discrete_heading:
                end_point = self.targets[a - 1] + [b * 10 * np.pi / 180]
                for u in range(len(self.uav_id)):
                    distance = dubins.shortest_path(self.uav_position[u], end_point, self.uav_Rmin[u]).path_length()
                    self.cost_matrix[u][0][0][a][b] = distance
                    self.cost_matrix[u][a][b][0][0] = distance
        return regenerate, empty

    def run_PSO(self, iteration, uav_message, population=None):
        # start_t = time.time()
        a = []
        regenerate, empty = self.information_setting(uav_message, population)
        particles = self.initiate_particles()
        self.fitness_evaluate(particles)
        a.append(1/self.Gbest[0])
        iteration -= 1
        for iterate in range(iteration):
            self.particle_improve(particles)
            self.fitness_evaluate(particles)
            a.append(1/self.Gbest[0])
            # print(1/max(fitness))
        # print(time.time() - start_t)
        return self.Gbest, a

    def plot_result(self, best_solution, curve):
        print(f'best gene:{best_solution}')
        uav_num = len(self.uav_id)
        route_state = [[self.uav_position[_]] for _ in range(uav_num)]

        sorted_target = sorted(range(len(best_solution[0])), key=lambda u: best_solution[0][u])  # target sequence
        sorted_pos = [math.floor(pos) for pos in sorted(best_solution[0])]  # assign agent
        print(sorted(best_solution[0]))
        print(best_solution[0])
        cost = [0 for _ in range(uav_num)]
        pre_site = [site for site in self.uav_position]
        task_sequence_time = [[] for _ in range(uav_num)]  # time
        task_type_count = [1 for _ in range(len(self.targets))]

        for j in range(len(best_solution[0])):
            assign_uav = sorted_pos[j]-1
            assign_target = sorted_target[j] // 3 + 1
            assign_heading = best_solution[1][j]
            sp = pre_site[assign_uav]
            gp = self.targets[assign_target - 1] + [assign_heading]
            if sp == gp:
                gp[-1] += 1e-5
            dubins_path = dubins.shortest_path(sp, gp, self.uav_Rmin[assign_uav])
            cost[assign_uav] += dubins_path.path_length()
            task_sequence_time[assign_uav].append([assign_target, task_type_count[assign_target-1],
                                                   cost[assign_uav] / self.uav_velocity[assign_uav]])
            route_state[assign_uav].extend(dubins_path.sample_many(.1)[0])
            pre_site[assign_uav] = self.targets[assign_target - 1] + [assign_heading]
            task_type_count[assign_target - 1] += 1
        for j in range(uav_num):
            dubins_path = dubins.shortest_path(pre_site[j], self.depots[j], self.uav_Rmin[j])
            cost[j] += dubins_path.path_length()
            route_state[j].extend(dubins_path.sample_many(.1)[0])
        # arrange to target-based to check time sequence constraints
        time_list = []
        for time_sequence in task_sequence_time:
            time_list.extend(time_sequence)
        time_list.sort()
        print(time_list)
        penalty, j = 0, 0
        for task_num in self.tasks_status:
            if task_num >= 2:
                for k in range(1, task_num):
                    penalty += max(0, time_list[j + k - 1][2] - time_list[j + k][2])
            j += task_num
        # print(f'penalty: {penalty}')
        # print(max([t[2] for t in time_list]))
        # print(np.divide(dist, self.uav_velocity))
        # print(1/self.fitness_evaluate([best_solution])[0][0])
        color_style = ['tab:blue', 'tab:green', 'tab:orange', 'b']
        plt.subplot(121)
        for i in range(uav_num):
            plt.plot([x[0] for x in route_state[i]], [x[1] for x in route_state[i]], '-')
            plt.axis("equal")
            # for arrow in arrow_state[i]:
            #     plt.arrow(arrow[0], arrow[1], arrow[2] - arrow[0], arrow[3] - arrow[1], width=8, color=color_style[i])
        plt.plot([x[0] for x in self.uav_position], [x[1] for x in self.uav_position], 'ro')
        plt.plot([b[0] for b in self.targets], [b[1] for b in self.targets], 'bo')
        plt.subplot(122)
        plt.plot([b for b in range(1, len(curve)+1)], curve, '-')
        plt.show()


if __name__ == "__main__":
    # targets_sites = [[3900, 2400], [-1370, 4800], [-3700, 2520]]
    # targets_sites = [[3850, 650], [3900, 4700], [2750, 150], [1000, 2750], [4750, 4300], [5000, 5000]]
    targets_sites = [[500, 1500], [2000, 4500], [3000, 1500], [1500, 2500], [2500, 3500], [3500, 4000]]
    # targets_sites = [[random.randint(-5000, 5000), random.randint(0, 5000)] for _ in range(20)]
    # terminal = [[-500, 1200, np.pi / 2], [0, 1200, np.pi / 2], [500, 1200, np.pi / 2]]
    # terminal = [[-500, 1200, np.pi / 2], [0, 1200, np.pi / 2], [500, 1200, np.pi / 2], [1000, 1200, np.pi / 2]]
    # uavs = [[1, 2, 3, 4], [1, 2, 3, 2], [25, 15, 35, 40], [75, 50, 100, 100],
    #         [[-500, 0, np.pi / 2], [0, 0, np.pi / 2], [500, 0, np.pi / 2], [1000, 0, np.pi / 2]], terminal,
    #         [], [], [], []]
    # uavs = [[1, 2, 3], [1, 2, 3], [25, 15, 35], [75, 50, 100],
    #         [[-500, 0, np.pi / 2], [0, 0, np.pi / 2], [500, 0, np.pi / 2]], terminal,
    #         [], [], [], []]
    uavs = [[1, 2, 3], [1, 2, 3], [70, 80, 70], [200, 250, 200],
            [[0, 0, np.pi / 2], [500, 0, np.pi / 2], [1000, 0, np.pi / 2]],
            [[0, 0, -np.pi / 2], [500, 0, -np.pi / 2], [1000, 0, -np.pi / 2]],
            [], [], [], []]
    # uavs = [[1, 2, 3, 4, 5, 6], [2, 3, 2, 1, 2, 3], [235, 258, 225, 265, 220, 225], [80, 60, 70, 80, 90, 90],
    #         [[1000, 0, np.pi / 2], [1050, 0, np.pi / 2], [1100, 0, np.pi / 2], [1150, 0, np.pi / 2],
    #          [1200, 0, np.pi / 2], [1250, 0, np.pi / 2]],
    #         [[1000, 0, -np.pi / 2], [1050, 0, -np.pi / 2], [1100, 0, -np.pi / 2], [1150, 0, -np.pi / 2],
    #          [1200, 0, -np.pi / 2], [1250, 0, -np.pi / 2]],
    #         [], [], [], []]
    pso = PSO_SEAD(targets_sites)
    tf = time.time()
    [fitness_, solution], a = pso.run_PSO(100, uavs)
    print(time.time()-tf)
    print(1/fitness_)
    pso.plot_result(solution, a)
