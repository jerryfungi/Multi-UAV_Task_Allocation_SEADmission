import random
import time
import math
import numpy as np
import copy
from matplotlib import pyplot as plt
import dubins


class GA_SEAD(object):
    def __init__(self, targets, pop=300):
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
        # GA parameters
        self.population_size = pop
        self.crossover_num = 66
        self.mutation_num = 32
        self.elitism_num = 2
        self.lambda_1 = 0
        self.lambda_2 = 1e4
        self.Nm = 0

    def fitness_evaluate_calculate(self, population):
        fitness_value = []
        uav_num = len(self.uav_id)
        chromosome_len = len(population[0][0])
        for chromosome in population:
            cost = [0 for _ in range(uav_num)]
            pre_site = [site for site in self.uav_position]
            task_sequence_time = [[] for _ in range(uav_num)]  # time
            time_list = []
            for j in range(chromosome_len):
                assign_uav = self.uav_id.index(chromosome[3][j])
                assign_target = chromosome[1][j]
                assign_heading = chromosome[4][j]
                sp = pre_site[assign_uav]
                gp = self.targets[assign_target-1]+[assign_heading*10*np.pi/180]
                if sp == gp:
                    gp[-1] += 1e-5
                cost[assign_uav] += dubins.shortest_path(sp, gp, self.uav_Rmin[assign_uav]).path_length()
                task_sequence_time[assign_uav].append([assign_target, chromosome[2][j],
                                                       cost[assign_uav] / self.uav_velocity[assign_uav]])
                pre_site[assign_uav] = self.targets[assign_target-1]+[assign_heading*10*np.pi/180]
            for j in range(uav_num):
                cost[j] += dubins.shortest_path(pre_site[j], self.depots[j], self.uav_Rmin[j]).path_length()
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
        roulette_wheel = np.array(fitness_value) / np.sum(fitness_value)
        return fitness_value, roulette_wheel

    def fitness_evaluate(self, population):
        fitness_value = []
        uav_num = len(self.uav_id)
        for chromosome in population:
            cost = [0 for _ in range(uav_num)]
            task_sequence_time = [[] for _ in range(uav_num)]  # time
            time_list = []
            pre_site, pre_heading = [0 for _ in range(uav_num)], [0 for _ in range(uav_num)]
            for j in range(len(chromosome[0])):
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
            # fitness_value.append(1/max([t[2] for t in time_list]))
        roulette_wheel = np.array(fitness_value) / np.sum(fitness_value)
        return fitness_value, roulette_wheel

    def generate_population(self,):
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
            order_based_gene = np.array(sorted(target_based_gene, key=lambda u: u[0]))
            for i in range(5):
                chromosome[i] = [g[i] for g in order_based_gene]
            return chromosome
        return [generate_chromosome() for _ in range(self.population_size)]

    @staticmethod
    def selection(roulette_wheel, num):
        return np.random.choice(np.arange(len(roulette_wheel)), size=num, replace=False, p=roulette_wheel)

    def crossover_operator(self, wheel, population):
        def generate_crossover_offspring(parent_1, parent_2):
            target_based_gene, order_based_gene = [], []
            child_1, child_2 = [], []
            # turn to target-based
            for parents in [parent_1, parent_2]:
                zipped_gene = [list(g) for g in zip(parents[0], parents[1], parents[2], parents[3], parents[4])]
                target_based_gene.append(sorted(zipped_gene, key=lambda u: u[1]))
            # choose cut point
            cutpoint = random.sample(range(len(parent_1[0])), 2)
            cutpoint_1, cutpoint_2 = min(cutpoint), max(cutpoint)
            # 2 point crossover
            try:
                target_based_gene[0][cutpoint_1:cutpoint_2], target_based_gene[1][cutpoint_1:cutpoint_2] = \
                    [[b[:3] for b in target_based_gene[0][cutpoint_1:cutpoint_2]][i] +
                        [a[3:] for a in target_based_gene[1][cutpoint_1:cutpoint_2]][i] for i in range(cutpoint_2 - cutpoint_1)], \
                    [[b[:3] for b in target_based_gene[1][cutpoint_1:cutpoint_2]][i] +
                        [a[3:] for a in target_based_gene[0][cutpoint_1:cutpoint_2]][i] for i in range(cutpoint_2 - cutpoint_1)]
                # back to order-based
                for gene in target_based_gene:
                    order_based_gene.append(sorted(gene, key=lambda u: u[0]))
                for i in range(5):
                    child_1.append([g[i] for g in order_based_gene[0]])
                    child_2.append([g[i] for g in order_based_gene[1]])
                return [child_1, child_2]
            except IndexError:
                return [parent_1, parent_2]
        children = []
        for k in range(0, self.crossover_num, 2):
            p_1, p_2 = self.selection(wheel, 2)
            try:
                children.extend(generate_crossover_offspring(population[p_1], population[p_2]))
            except ValueError:
                children.extend([population[p_1], population[p_2]])
        return children

    def mutation_operator(self, wheel, population):
        def multi_point_mutation(chromosome):
            # choose mutate point
            mutpoint = np.random.choice(len(chromosome[0]), random.randint(1, self.Nm+1), replace=False)
            # mutate assign uav or heading angle
            new_gene = []
            for i in range(len(chromosome)):  # copy chromosome
                new_gene.append(chromosome[i][:])
            # new_gene = copy.deepcopy(chromosome)
            assign = random.choice([3, 4])
            # mutate assign uav
            if assign == 3:
                for point in mutpoint:
                    new_gene[assign][point] = random.choice(
                        [i for i in self.uavType_for_missions[new_gene[2][point] - 1]])
            # mutate assign heading
            else:
                for point in mutpoint:
                    new_gene[assign][point] = random.choice([i for i in self.discrete_heading if
                                                             i != chromosome[assign][point]])
            return new_gene

        def target_state_mutation(chromosome, target_index_array):
            # turn to target-based
            zipped_gene = [list(g) for g in zip(chromosome[0], chromosome[1], chromosome[2],
                                                chromosome[3], chromosome[4])]
            target_based_gene = (sorted(zipped_gene, key=lambda u: u[1]))
            target_sequence = [[index for (index, value) in enumerate(self.tasks_status) if value == task_num]
                               for task_num in range(1, 4)]
            for task_type in target_sequence:
                random.shuffle(task_type)
            target_sequence = target_sequence[0] + target_sequence[1] + target_sequence[2]
            mutate_target_based = [[] for _ in range(len(target_based_gene))]
            j = 0
            for sequence in target_sequence:
                mutate_target_based[j:j + self.tasks_status[sequence]] = \
                    [[b[:1] for b in target_based_gene[j:j + self.tasks_status[sequence]]][i] +
                     [a[1:] for a in target_based_gene[target_index_array[sequence]:target_index_array[sequence + 1]]]
                     [i] for i in range(self.tasks_status[sequence])]
                j += self.tasks_status[sequence]
            # back to order-based
            new_gene = [[] for _ in range(5)]
            try:
                order_based_gene = np.array(sorted(mutate_target_based, key=lambda u: u[0]))
                for i in range(5):
                    new_gene[i] = [g[i] for g in order_based_gene]
                return chromosome
            except:
                print(self.tasks_status)
                print(f"t:{target_based_gene}")
                print(f"m:{mutate_target_based}")
                print(chromosome)
                print(target_sequence)
                return chromosome

        def task_state_mutation(chromosome, task_amount_array, task_index_array):
            # turn to target-based
            zipped_gene = [list(g) for g in zip(chromosome[0], chromosome[1], chromosome[2],
                                                chromosome[3], chromosome[4])]
            task_based_gene = (sorted(zipped_gene, key=lambda u: u[2]))
            # choose mutate task
            muttask = random.choice([0, 1, 2])
            # shuffle the state
            task_sequence = list(range(task_amount_array[muttask]))
            random.shuffle(task_sequence)
            # task mutate
            mutate_task_based = []
            for i in range(len(task_based_gene)):  # copy chromosome
                mutate_task_based.append(task_based_gene[i][:])
            # mutate_task_based = copy.deepcopy(task_based_gene)
            for i, sequence in enumerate(task_sequence):
                mutate_task_based[task_index_array[muttask] + i][3:] = \
                    task_based_gene[task_index_array[muttask] + sequence][3:]
            # back to order-based
            new_gene = [[] for _ in range(5)]
            order_based_gene = np.array(sorted(mutate_task_based, key=lambda u: u[0]))
            for i in range(5):
                new_gene[i] = [g[i] for g in order_based_gene]
            # for task in [[1, 3], [2, 1], [3, 1]]:
            #     delete_task_index = new_gene[1].index(task[0])
            #     if new_gene[2][delete_task_index] == task[1]:
            #         pass
            #     else:
            #         print(f'task_state_mutation error')
            return new_gene

        def generate_mutation_offspring(chromosome):
            random_choose = random.choice([1, 2, 3])
            if random_choose == 1:
                mut_gene = multi_point_mutation(chromosome)
            elif random_choose == 2:
                mut_gene = target_state_mutation(chromosome, target_index)
            else:
                mut_gene = task_state_mutation(chromosome, task_amount, task_index)
            return mut_gene
        self.Nm = round(sum(self.tasks_status)/2)
        task_amount = [np.count_nonzero(np.array(self.tasks_status) >= 3 - a) for a in range(3)]
        task_index = [0, task_amount[0], task_amount[0]+task_amount[1]]
        target_index = [0]
        for k, times in enumerate(self.tasks_status):
            target_index.append(target_index[k] + times)
        return [generate_mutation_offspring(population[self.selection(wheel, 1)[0]]) for _ in range(self.mutation_num)]

    def elitism_operator(self, fitness, population):
        fitness_ranking = sorted(range(len(fitness)), key=lambda u: fitness[u], reverse=True)[
                          :self.elitism_num]
        return [population[_] for _ in fitness_ranking]

    def information_setting(self, uavs, population):
        lost_agent, build_graph, empty = False, False, False
        terminated_tasks, new_target = uavs[8], uavs[9]
        clear_task, new_task = [], []
        print(uavs[7])
        if terminated_tasks:  # check terminated tasks
            print(terminated_tasks)
            terminated_tasks = sorted(terminated_tasks, key=lambda u: u[1])
            for task in terminated_tasks:
                if self.tasks_status[task[0] - 1] == 3 - task[1] + 1:
                    self.tasks_status[task[0] - 1] -= 1
                    clear_task.append(task)
        if new_target:  # check new targets
            for target in new_target:
                if target not in self.targets:
                    self.targets.append(target)
                    self.tasks_status.append(3)
                    new_task.append(self.targets.index(target)+1)
            build_graph = True
        if sum(self.tasks_status) == 0:
            empty = True
        if not set(self.uav_id) == set(uavs[0]):  # check agents
            build_graph = True
            lost_agent = True
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
            self.cost_matrix = [[[[[0 for a in range(len(self.discrete_heading))] for b in range(len(self.targets) + 1)]
                                  for c in range(len(self.discrete_heading))] for d in range(len(self.targets) + 1)]
                                for u in range(len(self.uav_id))]
            for a in range(1, len(self.targets) + 1):
                for b in self.discrete_heading:
                    for c in range(1, len(self.targets) + 1):
                        for d in self.discrete_heading:
                            source_point = self.targets[a - 1] + [b * 10 * np.pi / 180]
                            end_point = self.targets[c - 1] + [d * 10 * np.pi / 180]
                            if source_point == end_point:
                                end_point[-1] += 1e5
                            for u in range(len(self.uav_id)):
                                distance = dubins.shortest_path(source_point, end_point, self.uav_Rmin[u]).path_length()
                                self.cost_matrix[u][a][b][c][d] = distance
        # update real time information in graph
        for a in range(1, len(self.targets) + 1):
            for b in self.discrete_heading:
                point = self.targets[a - 1] + [b * 10 * np.pi / 180]
                for u in range(len(self.uav_id)):
                    distance = dubins.shortest_path(self.uav_position[u], point, self.uav_Rmin[u]).path_length()
                    self.cost_matrix[u][0][0][a][b] = distance
                    distance = dubins.shortest_path(point, self.depots[u], self.uav_Rmin[u]).path_length()
                    self.cost_matrix[u][a][b][0][0] = distance
        for u in range(len(self.uav_id)):
            distance = dubins.shortest_path(self.uav_position[u], self.depots[u], self.uav_Rmin[u]).path_length()
            self.cost_matrix[u][0][0][0][0] = distance
        # modify population
        if population:
            population.extend([elite for elite in uavs[7]])
            # remove the gene of terminated tasks
            for chromosome in population:
                for task in clear_task:
                    try:
                        delete_task_index = chromosome[1].index(task[0])
                        if chromosome[2][delete_task_index] == task[1]:
                            for row in chromosome:
                                row.pop(delete_task_index)
                        else:
                            print(f'error del{chromosome}')
                    except ValueError:
                        pass
            # agent lost
            if new_target:
                for chromosome in population:
                    for target in new_task:
                        for task_type in range(1, 4):
                            chromosome[0].append(chromosome[0][-1]+1)
                            chromosome[1].append(target)
                            chromosome[2].append(task_type)
                            chromosome[3].append(random.choice(self.uavType_for_missions[task_type-1]))
                            chromosome[4].append(random.choice(self.discrete_heading))
            if lost_agent:
                for chromosome in population:
                    for index, task_type in enumerate(chromosome[2]):
                        if chromosome[3][index] not in self.uav_id:
                            chromosome[3][index] = random.choice(self.uavType_for_missions[task_type - 1])

        # ga parameters
        # self.population_size = round(100 / len(self.uav_id))
        self.crossover_num = round((self.population_size - self.elitism_num) * 0.67)
        self.mutation_num = self.population_size - self.crossover_num - self.elitism_num
        # self.lambda_1 = 1e-3/len(self.uav_id)
        self.lambda_1 = 1 / (sum(self.uav_velocity))
        return population, empty

    def run_GA(self, iteration, uav_message, population=None):
        # start_t = time.time()
        a = []
        population, empty = self.information_setting(uav_message, population)
        if not empty:
            if not population:
                population = self.generate_population()
                iteration -= 1
            fitness, wheel = self.fitness_evaluate(population)
            a.append(1/max(fitness))
            for iterate in range(iteration):
                new_population = []
                new_population.extend(self.elitism_operator(fitness, population))
                new_population.extend(self.crossover_operator(wheel, population))
                new_population.extend(self.mutation_operator(wheel, population))
                fitness, wheel = self.fitness_evaluate(new_population)
                population = new_population
                a.append(1/max(fitness))
                # print(1/max(fitness))
            # print(time.time() - start_t)
            return population[fitness.index(max(fitness))], max(fitness), population, a
        else:
            return [[] for _ in range(5)], 0, [], 0

    def run_GA_time_period_version(self, time_interval, uav_message, population=None):
        iteration = 0
        start_time = time.time()
        population, empty = self.information_setting(uav_message, population)
        if not empty:
            if not population:
                population = self.generate_population()
            fitness, wheel = self.fitness_evaluate(population)
            while time.time() - start_time <= time_interval:
                iteration += 1
                new_population = []
                new_population.extend(self.elitism_operator(fitness, population))
                new_population.extend(self.crossover_operator(wheel, population))
                new_population.extend(self.mutation_operator(wheel, population))
                fitness, wheel = self.fitness_evaluate(new_population)
                population = new_population
            return population[fitness.index(max(fitness))], max(fitness), population
        else:
            return [[] for _ in range(5)], 0, []

    def plot_result(self, best_solution, curve=None):
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
            arrow_.extend(
                [[route_[0][arr], route_[1][arr], route_[0][arr + 100], route_[1][arr + 100]]
                 for arr in range(0, len(route_[0]), 15000)])
            print(state_list)
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
            assign_target = best_solution[1][j]
            assign_heading = best_solution[4][j] * 10
            task_sequence_state[assign_uav].append([
                self.targets[assign_target - 1][0], self.targets[assign_target - 1][1],
                assign_heading])
            task_route[assign_uav].extend([[assign_target, best_solution[2][j]]])
        for j in range(uav_num):
            task_sequence_state[j] = [self.uav_position[j]] + task_sequence_state[j] + [self.depots[j]]
            dist[j], route_state[j], arrow_state[j] = dubins_plot(task_sequence_state[j], j, task_route[j])
        for route in task_route:
            print(f'best route:{route}')
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
        print(f'mission time: {max(np.divide(dist, self.uav_velocity))} (sec)')
        print(f'total distance: {sum(dist)} (m)')
        print(max(np.divide(dist, self.uav_velocity))+self.lambda_1*sum(dist)+self.lambda_2*penalty)
        print(f'penalty: {penalty}')
        # print(np.divide(dist, self.uav_velocity))
        print(1/self.fitness_evaluate_calculate([best_solution])[0][0])
        print(1 / self.fitness_evaluate([best_solution])[0][0])
        color_style = ['tab:blue', 'tab:green', 'tab:orange', '#DC143C', '#808080', '#030764', '#C875C4', '#008080',
                       '#DAA520', '#580F41', '#7BC8F6', '#06C2AC']
        if curve:
            plt.subplot(122)
            plt.plot([b for b in range(1, len(curve) + 1)], curve, '-')
            print(len(curve))
            plt.grid()
            # plt.ylabel('E ( $\mathregular{J_i}$ / $\mathregular{J_1}$ )', fontsize=12)
            plt.subplot(121)
        for i in range(uav_num):
            # l = 2.0
            # plt.plot([sx, sx + l * np.cos(stheta)], [sy, sy + l * np.sin(stheta)], 'r-')
            # plt.plot([gx, gx + l * np.cos(gtheta)], [gy, gy + l * np.sin(gtheta)], 'r-')
            plt.plot(route_state[i][0], route_state[i][1], '-', markersize=.01, color=color_style[i])
            plt.text(self.uav_position[i][0]-100, self.uav_position[i][1]-200, f'UAV {self.uav_id[i]}', fontsize='8')
            plt.axis("equal")
            for arrow in arrow_state[i]:
                plt.arrow(arrow[0], arrow[1], arrow[2] - arrow[0], arrow[3] - arrow[1], width=30, color=color_style[i])
        plt.plot([x[0] for x in self.uav_position], [x[1] for x in self.uav_position], 'k^', label='UAV start point',
                 markerfacecolor='none', markersize=8)
        plt.plot([b[0] for b in self.targets], [b[1] for b in self.targets], 'ms', label='Target position',
                 markerfacecolor='none', markersize=6)
        plt.plot(self.depots[0][0], self.depots[0][1], 'r*', markerfacecolor='none', markersize=10, label='Airport')
        for t in self.targets:
            plt.text(t[0]+100, t[1]+100, f'Target {self.targets.index(t)+1}', color='m', fontsize='8')
        plt.text(self.depots[0][0]-100, self.depots[0][1]-200, 'Airport', color='r', fontsize='8')
        plt.legend(loc='upper right', fontsize=8)
        plt.grid()
        plt.show()


if __name__ == "__main__":
    # targets_sites = [[3100, 2200], [500, 3700], [2300, 2500], [1100, 2100], [2800, 4100]]
    # targets_sites = [[3100, 2200], [500, 3700], [2300, 2500], [2000, 3900]]
    targets_sites = [[3100, 2600], [500, 2400], [1800, 2100], [1800, 3600]]
    # targets_sites = [[3850, 650], [3900, 4700], [500, 1500], [1000, 2750], [4450, 3600], [2800, 3900], [800, 3600]]
    # targets_sites = [[4550, 650], [500, 1500], [1000, 2750], [4450, 3600], [4630, 4780], [800, 3600],
    #                  [3300, 2860], [2000, 2000], [3650, 1700], [2020, 3020]]
    # targets_sites = [[500, 1500], [2000, 4500], [3000, 1500]]
    # targets_sites = [[random.randint(-5000, 5000), random.randint(0, 5000)] for _ in range(20)]
    # terminal = [[-500, 1200, np.pi / 2], [0, 1200, np.pi / 2], [500, 1200, np.pi / 2]]
    # terminal = [[-500, 1200, np.pi / 2], [0, 1200, np.pi / 2], [500, 1200, np.pi / 2], [1000, 1200, np.pi / 2]]
    # uavs = [[1, 2, 3, 4], [1, 2, 3, 2], [25, 15, 35, 40], [75, 50, 100, 100],
    #         [[-500, 0, np.pi / 2], [0, 0, np.pi / 2], [500, 0, np.pi / 2], [1000, 0, np.pi / 2]], terminal,
    #         [], [], [], []]
    # uavs = [[1, 2, 3], [1, 2, 3], [25, 15, 35], [75, 50, 100],
    #         [[-500, 0, np.pi / 2], [0, 0, np.pi / 2], [500, 0, np.pi / 2]], terminal,
    #         [], [], [], []]
    uavs = [[1, 2, 3], [2, 2, 2], [70, 80, 90], [200, 250, 300],
            [[700, 1200, -np.pi], [1500, 700, np.pi / 2], [3600, 1000, np.pi / 3]],
            [[2500, 4500, np.pi / 2] for _ in range(3)],
            [], [], [], []]
    # uavs = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 2, 1, 2], [70, 80, 90, 60, 100, 80], [200, 250, 300, 180, 300, 260],
    #         [[500, 300, -60*np.pi/180], [1500, 700, 90*np.pi/180], [200, 1100, 135*np.pi/180],
    #          [3500, 120, 20*np.pi/180], [200, 4600, -45*np.pi/180], [4740, 2500, 115*np.pi/180]],
    #         [[5000, 1000, 0] for _ in range(6)],
    #         [], [], [], []]
    # uavs = [[i for i in range(1, 13)], [2, 2, 3, 1, 1, 3, 1, 2, 2, 1, 2, 2],
    #         [70, 80, 90, 60, 100, 80, 75, 90, 85, 70, 65, 50],
    #         [200, 250, 300, 180, 300, 260, 225, 295, 250, 200, 170, 150],
    #         [[0, 3770, 0 * np.pi / 180], [1500, 700, 90 * np.pi / 180], [200, 900, 135 * np.pi / 180],
    #          [1800, 4500, -20 * np.pi / 180], [200, 2800, 45 * np.pi / 180], [4740, 3000, 140 * np.pi / 180],
    #          [350, 120, 70 * np.pi / 180], [3500, 4500, -75 * np.pi / 180], [5000, 2000, -115 * np.pi / 180],
    #          [2780, 5000, -55 * np.pi / 180], [400, 4400, 85 * np.pi / 180], [2040, 300, 65 * np.pi / 180]],
    #         [[3000, 0, -135 * np.pi / 180] for _ in range(12)],
    #         [], [], [], []]
    ga = GA_SEAD(targets_sites, 100)
    tf = time.time()
    solution, fitness_, ga_population, a = ga.run_GA(100, uavs)
    print(time.time()-tf)
    print(1/fitness_)
    # print(solution)
    # print('second--------------------------------')
    # uavs = [[1, 2, 3], [1, 2, 3], [70, 80, 90], [200, 250, 300],
    #         [[700, 1200, -np.pi], [1500, 700, np.pi / 2], [3600, 1000, np.pi / 3]],
    #         [[2500, 500, -np.pi / 2] for _ in range(3)],
    #         [], [], [[1, 1], [1, 2]], []]
    # solution, fitness_, ga_population, a = ga.run_GA(100, uavs, ga_population)
    # print(solution)
    # uavs = [[1, 2, 3], [1, 2, 3], [70, 80, 90], [200, 250, 300],
    #         [[700, 1200, -np.pi], [1500, 700, np.pi / 2], [3600, 1000, np.pi / 3]],
    #         [[2500, 500, -np.pi / 2] for _ in range(3)],
    #         [], [], [[2, 1]], []]
    # solution, fitness_, ga_population, a = ga.run_GA(100, uavs, ga_population)
    # print(solution)
    # ga.plot_result([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [1, 3, 1, 4, 3, 1, 2, 3, 4, 4, 2, 2],
    #                 [1, 1, 2, 1, 2, 3, 1, 3, 2, 3, 2, 3], [2, 1, 3, 1, 3, 2, 1, 2, 3, 2, 3, 2],
    #                 [5, 11, 24, 11, 3, 13, 25, 13, 18, 14, 24, 23]])
    # solution = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    #             [3, 3, 3, 7, 6, 7, 7, 2, 4, 4, 6, 2, 6, 4, 5, 1, 5, 2, 1, 5, 1],
    #             [1, 2, 3, 1, 1, 2, 3, 1, 1, 2, 2, 2, 3, 3, 1, 1, 2, 3, 2, 3, 3],
    #             [2, 3, 1, 5, 5, 2, 2, 5, 1, 3, 3, 3, 1, 2, 5, 6, 3, 5, 6, 1, 6],
    #             [11, 19, 10, 2, 35, 12, 22, 35, 6, 6, 3, 0, 33, 34, 30, 25, 24, 22, 28, 29, 33]]
    # solution =  [
    #     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    #     [2, 10, 5, 7, 10, 3, 8, 9, 4, 1, 3, 5, 2, 9, 7, 6, 10, 6, 6, 3, 8, 1, 2, 5, 1, 4, 9, 4, 7, 8],
    #     [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 3, 2, 3, 3, 2, 2, 3, 3, 3, 2, 3, 3, 3, 3],
    #     [7, 1, 10, 5, 3, 7, 12, 5, 9, 5, 12, 9, 2, 2, 2, 11, 1, 3, 1, 1, 3, 11, 7, 9, 11, 9, 10, 9, 2, 10],
    #     [9, 25, 29, 20, 30, 16, 7, 3, 12, 22, 7, 4, 29, 8, 7, 28, 21, 34, 7, 24, 28, 26, 31, 0, 29, 19, 16, 26, 32, 6]]
    ga.plot_result(solution)
    # uavs = [[1, 2, 3], [1, 2, 3], [70, 80, 70], [200, 250, 200],
    #         [[0, 0, np.pi / 2], [50, 0, np.pi / 2], [100, 0, np.pi / 2]],
    #         [[0, 0, -np.pi / 2], [50, 0, -np.pi / 2], [100, 0, -np.pi / 2]],
    #         [], [], [[2, 1]], [[0, 4000]]]
    # solution, fitness_, ga_population = ga.run_GA(100, uavs, ga_population)
    # print(1 / fitness_)
    # ga.plot_result(solution)
    # tf = time.time()
    # solution, fitness_value, ga_population = ga.run_GA(100, uavs, ga_population)
    # solution, fitness_value, ga_population = ga.run_GA_time_period_version(1.5, uavs, ga_population)
    # print(time.time() - tf)
    # print(1 / fitness_value)
    # ga.plot_result(solution)
    # solution, fitness_value, ga_population = ga.run_GA_time_period_version(1.5, uavs, ga_population)
    # print(1 / fitness_value)
    # ga.plot_result(solution, curve)
    # solution, fitness_value, ga_population = ga.run_GA_time_period_version(1.5, uavs, ga_population)
    # print(1 / fitness_value)
    # uavs_1 = [[1, 2, 3], [1, 2, 3], [25, 15, 35], [75, 50, 100],
    #         [[-500, 0, np.pi / 2], [0, 0, np.pi / 2], [500, 0, np.pi / 2]], terminal, [], [], [[4, 1]], []]
    # solution, fitness_value, ga_population = ga.run_GA_time_period_version(1, uavs_1, ga_population)
    # uavs_1 = [[1, 2, 3], [1, 2, 3], [25, 15, 35], [75, 50, 100],
    #           [[-500, 0, np.pi / 2], [0, 0, np.pi / 2], [500, 0, np.pi / 2]], terminal, [], [],
    #           [[3, 2], [4, 2], [3, 1], [3, 1]], []]
    # solution, fitness_value, ga_population = ga.run_GA_time_period_version(1, uavs_1, ga_population)
    # print(1 / fitness_value)
    # solution, fitness_value, ga_population = ga.run_GA_time_period_version(1, uavs, ga_population)
    # print(1 / fitness_value)



