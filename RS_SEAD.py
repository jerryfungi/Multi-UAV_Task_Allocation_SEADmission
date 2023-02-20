import random
import time
import math
import numpy as np
import copy
from matplotlib import pyplot as plt
import dubins


class RS_SEAD(object):
    def __init__(self, targets, parameters_multiplier=1):
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
        # GA parameters
        self.population_size = 300
        self.lambda_1 = 0
        self.lambda_2 = 1e4
        self.Nm = int(len(self.targets)*3/2)
        self.discrete_heading = [_ for _ in range(0, 36)]

    def fitness_evaluate(self, population):
        fitness_value = []
        uav_num = len(self.uav_id)
        chromosome_len = len(population[0][0])
        for chromosome in population:
            cost = np.zeros(uav_num)
            pre_site, pre_heading = [0 for _ in range(uav_num)], [0 for _ in range(uav_num)]
            task_sequence_time = [[] for _ in range(uav_num)]  # time
            time_list = []
            for j in range(chromosome_len):
                assign_uav = self.uav_id.index(chromosome[3][j])
                assign_target = chromosome[1][j]
                assign_heading = chromosome[4][j]
                cost[assign_uav] += self.cost_matrix[assign_uav][pre_site[assign_uav]][pre_heading[assign_uav]][assign_target][assign_heading]
                task_sequence_time[assign_uav].append([assign_target, chromosome[2][j],
                                                       cost[assign_uav] / self.uav_velocity[assign_uav]])
                pre_site[assign_uav], pre_heading[assign_uav] = assign_target, assign_heading
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
                        penalty += max(0, time_list[j+k-1][2] - time_list[j+k][2])
                j += task_num
            fitness_value.extend([1 / (np.max(np.divide(cost, self.uav_velocity))
                                       + self.lambda_1 * np.sum(cost)
                                       + self.lambda_2 * penalty)])
        return fitness_value

    def generate_population(self):
        def generate_chromosome():
            chromosome = np.zeros((5, sum(self.tasks_status)), dtype=int)
            for i in range(chromosome.shape[1]):
                chromosome[0][i] = i + 1  # order
                chromosome[1][i] = random.choice([n for n in range(1, len(self.targets) + 1)  # target id
                                                  if np.count_nonzero(chromosome[1] == n) < self.tasks_status[n - 1]])
            # turn to target-based
            zipped_gene = [list(g) for g in zip(chromosome[0], chromosome[1], chromosome[2],
                                                chromosome[3], chromosome[4])]
            target_based_gene = np.array(sorted(zipped_gene, key=lambda u: u[1]))
            mission_type_list = []
            for tasks in self.tasks_status:
                mission_type_list.extend([n+1 for n in range(3-tasks, 3)])
            for i in range(target_based_gene.shape[0]):
                target_based_gene[i][2] = mission_type_list[i]  # mission type
                target_based_gene[i][3] = random.choice(
                    self.uavType_for_missions[target_based_gene[i][2] - 1])  # uav id
                target_based_gene[i][4] = random.choice(self.discrete_heading)  # heading angle
            # back to order-based
            chromosome = [[] for _ in range(5)]
            order_based_gene = (sorted(target_based_gene, key=lambda u: u[0]))
            for i in range(5):
                chromosome[i] = [g[i] for g in order_based_gene]
            return chromosome
        return [generate_chromosome() for _ in range(self.population_size)]

    def information_setting(self, uavs, population):
        regenerate, build_graph, empty = False, False, False
        terminated_tasks, new_target = uavs[8], uavs[9]
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
        # classify capable UAVs to the missions
        # [surveillance[1,3],attack[1,2,3],munition[2]], [surveillance[s,att],attack[att,a],verification[s,att]]
        for i, agent in enumerate(self.uav_type):
            if agent == 1:  # surveillance
                self.uavType_for_missions[0].append(self.uav_id[i])
                self.uavType_for_missions[2].append(self.uav_id[i])
            elif agent == 2:  # attack, combat
                self.uavType_for_missions[0].append(self.uav_id[i])
                self.uavType_for_missions[1].append(self.uav_id[i])
                self.uavType_for_missions[2].append(self.uav_id[i])
            elif agent == 3:  # munition
                self.uavType_for_missions[1].append(self.uav_id[i])
        # cost graph --------------------------------------------------------------------------------------------
        if build_graph:
            self.cost_matrix = [[[[[0 for a in range(len(self.discrete_heading))] for b in range(len(self.targets)+1)]
                                 for c in range(len(self.discrete_heading))] for d in range(len(self.targets)+1)]
                                for u in range(len(self.uav_id))]
            for a in range(1, len(self.targets)+1):
                for b in self.discrete_heading:
                    for c in range(a, len(self.targets)+1):
                        for d in self.discrete_heading:
                            source_point = self.targets[a-1]+[b*10*np.pi/180]
                            end_point = self.targets[c-1]+[d*10*np.pi/180]
                            if source_point == end_point:
                                end_point[-1] += 1e5
                            for u in range(len(self.uav_id)):
                                distance = dubins.shortest_path(source_point, end_point, self.uav_Rmin[u]).path_length()
                                self.cost_matrix[u][a][b][c][d] = distance
                                self.cost_matrix[u][c][d][a][b] = distance
        # update real time information in graph
        for a in range(1, len(self.targets) + 1):
            for b in self.discrete_heading:
                end_point = self.targets[a-1] + [b*10*np.pi/180]
                for u in range(len(self.uav_id)):
                    distance = dubins.shortest_path(self.uav_position[u], end_point, self.uav_Rmin[u]).path_length()
                    self.cost_matrix[u][0][0][a][b] = distance
                    self.cost_matrix[u][a][b][0][0] = distance
        for u in range(len(self.uav_id)):
            distance = dubins.shortest_path(self.uav_position[u], self.depots[u], self.uav_Rmin[u]).path_length()
            self.cost_matrix[u][0][0][0][0] = distance
        self.lambda_1 = 1 / (sum(self.uav_velocity))
        return regenerate, empty

    def run_RS(self, iteration, uav_message, population=None):
        # start_t = time.time()
        a = []
        regenerate, empty = self.information_setting(uav_message, population)
        population = self.generate_population()
        fitness = self.fitness_evaluate(population)
        a.append(1/max(fitness))
        iteration -= 1
        for iterate in range(iteration):
            new_population = self.generate_population()
            new_fitness = self.fitness_evaluate(new_population)
            for i, chromosome in enumerate(new_population):
                if new_fitness[i] > min(fitness):
                    fitness[fitness.index(min(fitness))] = new_fitness[i]
                    population[i] = chromosome
            a.append(1/max(fitness))
            # print(1/max(fitness))
        # print(time.time() - start_t)
        return population[fitness.index(max(fitness))], max(fitness), population, a

    def plot_result(self, best_solution, curve):
        def dubins_plot(state_list, c, time_):
            distance = 0
            route_ = [[] for _ in range(2)]
            arrow_ = []
            for a in range(1, len(state_list)-1):
                state_list[a][2] *= np.pi / 180
            for a in range(len(state_list) - 1):
                sp = state_list[a]
                gp = state_list[a + 1] if state_list[a] != state_list[a + 1] \
                    else [state_list[a + 1][0], state_list[a + 1][1], state_list[a + 1][2] - 1e-5]
                dubins_path = dubins.shortest_path(sp, gp, self.uav_Rmin[c])
                path, _ = dubins_path.sample_many(.1)
                route_[0].extend([b[0] for b in path])
                route_[1].extend([b[1] for b in path])
                distance += dubins_path.path_length()
                try:
                    time_[a].append(distance / self.uav_velocity[c])
                except IndexError:
                    pass
            # arrow_.extend(
            #     [[route_[0][arr], route_[1][arr], route_[0][arr + 100], route_[1][arr + 100]]
            #      for arr in range(0, len(route_[0]), 7000)])
            # print(state_list)
            return distance, route_, arrow_

        print(f'best gene:{best_solution}')
        uav_num = len(self.uav_id)
        dist = np.zeros(uav_num)
        task_sequence_state = [[] for _ in range(uav_num)]
        task_route = [[] for _ in range(uav_num)]
        route_state = [[] for _ in range(uav_num)]
        arrow_state = [[] for _ in range(uav_num)]
        for j in range(len(best_solution[0])):
            assign_uav = self.uav_id.index(best_solution[3][j])
            assign_target = int(best_solution[1][j])
            assign_heading = best_solution[4][j]
            task_sequence_state[assign_uav].append([
                self.targets[assign_target - 1][0], self.targets[assign_target - 1][1],
                assign_heading])
            task_route[assign_uav].extend([[assign_target, best_solution[2][j]]])
        for j in range(uav_num):
            task_sequence_state[j] = [self.uav_position[j]] + task_sequence_state[j] + [self.depots[j]]
            dist[j], route_state[j], arrow_state[j] = dubins_plot(task_sequence_state[j], j, task_route[j])
        print(f'best route:{task_route[0]}')
        print(f'best route:{task_route[1]}')
        print(f'best route:{task_route[2]}')
        # arrange to target-based to check time sequence constraints
        time_list = []
        for time_sequence in task_route:
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
            # l = 2.0
            # plt.plot([sx, sx + l * np.cos(stheta)], [sy, sy + l * np.sin(stheta)], 'r-')
            # plt.plot([gx, gx + l * np.cos(gtheta)], [gy, gy + l * np.sin(gtheta)], 'r-')
            plt.plot(route_state[i][0], route_state[i][1], '-')
            # plt.plot(self.depots[i][0], self.depots[i][1], 'cs')
            plt.axis("equal")
            # for arrow in arrow_state[i]:
            #     plt.arrow(arrow[0], arrow[1], arrow[2] - arrow[0], arrow[3] - arrow[1], width=8, color=color_style[i])
        plt.plot([x[0] for x in self.uav_position], [x[1] for x in self.uav_position], 'ro')
        plt.plot([b[0] for b in self.targets], [b[1] for b in self.targets], 'bo')
        plt.subplot(122)
        plt.plot([b for b in range(1, len(curve)+1)], curve, '-')
        plt.show()


if __name__ == "__main__":
    # targets_sites = [[3850, 650], [3900, 4700], [500, 1500], [1000, 2750], [4450, 3600], [2800, 3900], [800, 3600]]
    # targets_sites = [[3100, 2200], [500, 3700], [2300, 2500], [2000, 3900]]
    targets_sites = [[4550, 650], [500, 1500], [1000, 2750], [4450, 3600], [4630, 4780], [800, 3600],
                     [3300, 2860], [2000, 2000], [3650, 1700], [2020, 3020]]
    # uavs = [[1, 2, 3], [2, 2, 2], [70, 80, 70], [200, 250, 200],
    #         [[0, 0, np.pi / 2], [50, 0, np.pi / 2], [100, 0, np.pi / 2]],
    #         [[0, 0, -np.pi / 2], [50, 0, -np.pi / 2], [100, 0, -np.pi / 2]],
    #         [], [], [], []]
    # uavs = [[1, 2, 3], [1, 2, 3], [70, 80, 90], [200, 250, 300],
    #         [[700, 1200, -np.pi], [1500, 700, np.pi / 2], [3600, 1000, np.pi / 3]],
    #         [[2500, 500, -np.pi / 2] for _ in range(3)],
    #         [], [], [], []]
    # uavs = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 2, 1, 2], [70, 80, 90, 60, 75, 80], [200, 250, 300, 180, 225, 260],
    #         [[500, 300, -60 * np.pi / 180], [1500, 700, 90 * np.pi / 180], [2800, 100, 135 * np.pi / 180],
    #          [3500, 120, 20 * np.pi / 180], [200, 4600, -45 * np.pi / 180], [4740, 2500, 115 * np.pi / 180]],
    #         [[5000, 1000, 0] for _ in range(6)],
    #         [], [], [], []]
    uavs = [[i for i in range(1, 13)], [2, 2, 3, 1, 1, 3, 1, 2, 2, 1, 2, 2],
            [70, 80, 90, 60, 100, 80, 75, 90, 85, 70, 65, 50],
            [200, 250, 300, 180, 300, 260, 225, 295, 250, 200, 170, 150],
            [[0, 3770, 0 * np.pi / 180], [1500, 700, 90 * np.pi / 180], [200, 900, 135 * np.pi / 180],
             [1800, 4500, -20 * np.pi / 180], [200, 2800, 45 * np.pi / 180], [4740, 3000, 140 * np.pi / 180],
             [350, 120, 70 * np.pi / 180], [3500, 4500, -75 * np.pi / 180], [5000, 2000, -115 * np.pi / 180],
             [2780, 5000, -55 * np.pi / 180], [400, 4400, 85 * np.pi / 180], [2040, 300, 65 * np.pi / 180]],
            [[3000, 0, -135 * np.pi / 180] for _ in range(12)],
            [], [], [], []]
    # uavs = [[1, 2, 3, 4, 5, 6], [2, 3, 2, 1, 2, 3], [235, 258, 225, 265, 220, 225], [80, 60, 70, 80, 90, 90],
    #         [[1000, 0, np.pi / 2], [1050, 0, np.pi / 2], [1100, 0, np.pi / 2], [1150, 0, np.pi / 2],
    #          [1200, 0, np.pi / 2], [1250, 0, np.pi / 2]],
    #         [[1000, 0, -np.pi / 2], [1050, 0, -np.pi / 2], [1100, 0, -np.pi / 2], [1150, 0, -np.pi / 2],
    #          [1200, 0, -np.pi / 2], [1250, 0, -np.pi / 2]],
    #         [], [], [], []]
    rs = RS_SEAD(targets_sites)
    tf = time.time()
    solution, fitness_, ga_population, a = rs.run_RS(300, uavs)
    print(time.time()-tf)
    print(1/fitness_)
    rs.plot_result(solution, a)
