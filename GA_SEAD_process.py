import random
import time
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import dubins


class GA_SEAD(object):
    def __init__(self, targets, population_size=300, crossover_prob=0.67, elitism_num=2, heading_discretization=10):
        self.targets = targets
        # Global message
        self.uav_id = []
        self.uav_type = []
        self.uav_velocity = []
        self.uav_turning_radius = []
        self.uav_state = []
        self.uav_base = []
        self.uav_num, self.target_num = 0, 0
        # GA parameters
        self.total_population_size = population_size
        self.population_size = population_size
        self.elitism_num = elitism_num
        self.crossover_num = round((self.population_size - self.elitism_num) * crossover_prob)
        self.mutation_num = self.population_size - self.crossover_num - self.elitism_num
        self.crossover_prob = crossover_prob
        self.mutation_operators_prob = [0.25, 0.25, 0.25, 0.25]
        self.crossover_operators_prob = [0.5, 0.5]
        # Coefficients of objective function
        self.lambda_1 = 0
        self.lambda_2 = 10
        # The precomputed matrix for optimization
        self.cost_graph = []
        self.uavType_for_missions = []
        self.tasks_status = [3 for _ in range(len(self.targets))]
        self.heading_discretization = heading_discretization
        self.discrete_integer_heading = [theta for theta in range(heading_discretization)]
        self.heading_multiplier = 2 * np.pi / heading_discretization
        self.remaining_targets = []
        self.task_amount_array = []
        self.task_index_array = []
        self.target_sequence = []
        self.target_index_array = []

    class Chromosome:
        def __init__(self, chromosome):
            self.chromosome = chromosome
            self.fitness_value = 0

        def copy_chromosome(self):
            chromosome_len, gene_len = len(self.chromosome), len(self.chromosome[0])
            duplicate_chromosome = [[0 for _ in range(gene_len)] for _ in range(chromosome_len)]
            for i in range(chromosome_len):
                for j in range(gene_len):
                    duplicate_chromosome[i][j] = self.chromosome[i][j]
            return duplicate_chromosome

    @staticmethod
    def order2target_bundle(chromosome):
        zipped_gene = [list(g) for g in zip(chromosome[0], chromosome[1], chromosome[2], chromosome[3], chromosome[4])]
        return sorted(sorted(zipped_gene, key=lambda u: u[2]), key=lambda u: u[1])

    @staticmethod
    def order2task_bundle(chromosome):
        zipped_gene = [list(g) for g in zip(chromosome[0], chromosome[1], chromosome[2], chromosome[3], chromosome[4])]
        return sorted(zipped_gene, key=lambda u: u[2])

    @staticmethod
    def turn2order_based(chromosome):
        order_based_gene = np.array(sorted(chromosome, key=lambda u: u[0]))
        return [[g[i] for g in order_based_gene] for i in range(5)]

    @staticmethod
    def get_roulette_wheel(population):
        fitness_list = np.array([c.fitness_value for c in population])
        return fitness_list / np.sum(fitness_list)

    def objectives_evaluation(self, chromosome):
        cost = [0 for _ in range(self.uav_num)]
        task_sequence_time, time_list = [[] for _ in range(self.uav_num)], []  # time
        pre_site, pre_heading = [0 for _ in range(self.uav_num)], [0 for _ in range(self.uav_num)]
        for j, _ in enumerate(chromosome.chromosome[0]):
            assign_uav = self.uav_id.index(chromosome.chromosome[3][j])
            assign_target = chromosome.chromosome[1][j]
            assign_heading = chromosome.chromosome[4][j]
            cost[assign_uav] += self.cost_graph[assign_uav][pre_site[assign_uav]][pre_heading[assign_uav]][assign_target][assign_heading]
            task_sequence_time[assign_uav].append([assign_target, chromosome.chromosome[2][j],
                                                   cost[assign_uav] / self.uav_velocity[assign_uav]])
            pre_site[assign_uav], pre_heading[assign_uav] = assign_target, assign_heading
        for j in range(self.uav_num):
            cost[j] += self.cost_graph[j][pre_site[j]][pre_heading[j]][0][0]
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
        # Calculate objective value
        mission_time = np.max(np.divide(cost, self.uav_velocity))
        total_distance = np.sum(cost)
        fitness = 1 / (mission_time + self.lambda_1 * total_distance + self.lambda_2 * penalty)
        return fitness, mission_time, total_distance, penalty

    def fitness_evaluation(self, population):
        for chromosome in population:
            fitness, _, _, _ = self.objectives_evaluation(chromosome)
            # Update fitness value
            chromosome.fitness_value = fitness

    def generate_population(self):
        def generate_chromosome():
            chromosome = np.zeros((5, sum(self.tasks_status)), dtype=int)
            for i in range(chromosome.shape[1]):
                chromosome[0][i] = i + 1  # order
                chromosome[1][i] = random.choice([n for n in range(1, self.target_num + 1)
                                                  if np.count_nonzero(chromosome[1] == n)
                                                  < self.tasks_status[n - 1]])  # target id
            # turn to target-based
            target_bundle_chromosome = self.order2target_bundle(chromosome)
            for i in range(len(target_bundle_chromosome)):
                target_bundle_chromosome[i][2] = mission_type_list[i]  # mission type
                target_bundle_chromosome[i][3] = random.choice(
                    self.uavType_for_missions[target_bundle_chromosome[i][2] - 1])  # uav id
                target_bundle_chromosome[i][4] = random.choice(self.discrete_integer_heading)  # heading angle
            return self.Chromosome(self.turn2order_based(target_bundle_chromosome))  # back to order-based
        mission_type_list = []
        for tasks in self.tasks_status:
            mission_type_list.extend([n + 1 for n in range(3 - tasks, 3)])
        return [generate_chromosome() for _ in range(self.population_size)]

    @staticmethod
    def selection(roulette_wheel, num):
        return np.random.choice(np.arange(len(roulette_wheel)), size=num, replace=False, p=roulette_wheel)

    def crossover_operator(self, wheel, population):
        def two_point_crossover(parent_1, parent_2):
            # turn to target-based
            target_based_gene = [self.order2target_bundle(parent_1.chromosome),
                                 self.order2target_bundle(parent_2.chromosome)]
            # choose cut point
            cut_point_1, cut_point_2 = sorted(random.sample(range(len(parent_1.chromosome[0])), 2))
            cut_len = cut_point_2 - cut_point_1
            target_based_gene[0][cut_point_1:cut_point_2], target_based_gene[1][cut_point_1:cut_point_2] = \
                [target_based_gene[0][cut_point_1 + i][:3] + target_based_gene[1][cut_point_1 + i][3:] for i in
                 range(cut_len)], \
                [target_based_gene[1][cut_point_1 + i][:3] + target_based_gene[0][cut_point_1 + i][3:] for i in
                 range(cut_len)]
            # back to order-based
            child_1 = self.turn2order_based(target_based_gene[0])
            child_2 = self.turn2order_based(target_based_gene[1])
            return [self.Chromosome(child_1), self.Chromosome(child_2)]

        def target_bundle_crossover(parent_1, parent_2):
            # turn to target-based
            target_based_gene = [self.order2target_bundle(parent_1.chromosome),
                                 self.order2target_bundle(parent_2.chromosome)]
            # select targets to exchange
            targets_exchanged = random.sample(self.remaining_targets, random.randint(1, len(self.remaining_targets)))
            for target in targets_exchanged:
                start_index = sum(self.tasks_status[:target - 1])
                for i in range(start_index, start_index + self.tasks_status[target - 1]):
                    target_based_gene[0][i], target_based_gene[1][i] = \
                        target_based_gene[0][i][:3] + target_based_gene[1][i][3:], \
                        target_based_gene[1][i][:3] + target_based_gene[0][i][3:]
            # back to order-based
            child_1 = self.turn2order_based(target_based_gene[0])
            child_2 = self.turn2order_based(target_based_gene[1])
            return [self.Chromosome(child_1), self.Chromosome(child_2)]

        children = []
        for k in range(0, self.crossover_num, 2):
            p_1, p_2 = self.selection(wheel, 2)
            children.extend(np.random.choice([two_point_crossover, target_bundle_crossover],
                                             p=self.crossover_operators_prob)(population[p_1], population[p_2]))
        return children

    def mutation_operator(self, wheel, population):
        def point_agent_mutation(chromosome):
            # choose a point to mutate
            mut_point = np.random.randint(0, len(chromosome.chromosome[0]))
            new_gene = chromosome.copy_chromosome()
            # mutate assign UAV
            new_gene[3][mut_point] = random.choice([i for i in self.uavType_for_missions[new_gene[2][mut_point] - 1]])
            return self.Chromosome(new_gene)

        def point_heading_mutation(chromosome):
            # choose a point to mutate
            mut_point = np.random.randint(0, len(chromosome.chromosome[0]))
            new_gene = chromosome.copy_chromosome()
            # mutate assign heading
            new_gene[4][mut_point] = random.choice([i for i in self.discrete_integer_heading
                                                    if i != chromosome.chromosome[4][mut_point]])
            return self.Chromosome(new_gene)

        def target_bundle_mutation(chromosome):
            target_based_gene = self.order2target_bundle(chromosome.chromosome)
            for task_type in self.target_sequence:
                random.shuffle(task_type)
            shuffle_sequence = self.target_sequence[0] + self.target_sequence[1] + self.target_sequence[2]
            mutate_target_based = [[] for _ in range(len(target_based_gene))]
            j = 0
            for sequence in shuffle_sequence:
                mutate_target_based[j:j + self.tasks_status[sequence]] = \
                    [[b[:1] for b in target_based_gene[j:j + self.tasks_status[sequence]]][i] +
                     [a[1:] for a in target_based_gene[self.target_index_array[sequence]:self.target_index_array[sequence + 1]]]
                     [i] for i in range(self.tasks_status[sequence])]
                j += self.tasks_status[sequence]
            return self.Chromosome(self.turn2order_based(mutate_target_based))

        def task_bundle_mutation(chromosome):
            # turn to target-based
            task_based_gene = self.order2task_bundle(chromosome.chromosome)
            # choose a task to mutate
            mut_task = np.random.randint(0, 3)
            # shuffle the state
            task_sequence = list(range(self.task_amount_array[mut_task]))
            random.shuffle(task_sequence)
            # copy
            chromosome_len, gene_len = len(task_based_gene), len(task_based_gene[0])
            mutate_task_based = [[0 for _ in range(gene_len)] for _ in range(chromosome_len)]
            for i in range(chromosome_len):
                for j in range(gene_len):
                    mutate_task_based[i][j] = task_based_gene[i][j]
            # task mutation
            for i, sequence in enumerate(task_sequence):
                mutate_task_based[self.task_index_array[mut_task] + i][3:] = \
                    task_based_gene[self.task_index_array[mut_task] + sequence][3:]
            return self.Chromosome(self.turn2order_based(mutate_task_based))

        mutation_operators = [point_agent_mutation, point_heading_mutation, target_bundle_mutation, task_bundle_mutation]
        return [np.random.choice(mutation_operators, p=self.mutation_operators_prob)
                (population[self.selection(wheel, 1)[0]]) for _ in range(self.mutation_num)]

    def elitism_operator(self, population):
        fitness_ranking = sorted(range(len(population)), key=lambda u: population[u].fitness_value, reverse=True)
        return [population[_] for _ in fitness_ranking[:self.elitism_num]]

    def information_setting(self, information, population, distributed=False):
        lost_agent, build_graph = False, False
        terminated_tasks, new_target = \
            sorted(information.tasks_completed, key=lambda u: u[1]), sorted(information.new_targets)
        clear_task, new_task = [], []
        if terminated_tasks:  # check terminated tasks
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
        if not set(self.uav_id) == set(information.uav_id):  # check agents
            build_graph = True
            lost_agent = True
        # Clear the information
        self.uav_id = information.uav_id
        self.uav_type = information.uav_type
        self.uav_velocity = information.cruising_speed
        self.uav_turning_radius = information.turning_radii
        self.uav_state = information.uav_states
        self.uav_base = information.base
        self.uavType_for_missions = [[] for _ in range(3)]
        self.uav_num, self.target_num = len(self.uav_id), len(self.targets)
        # Classify capable UAVs to the missions
        # [surveillance[1,3],attack[1,2,3],munition[2]], [surveillance[s,a],attack[a,m],verification[s,a]]
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
        # COST TABLE (graph) -------------------------------------------------------------------------------------
        if build_graph:
            self.cost_graph = [[[[[0 for a in range(self.heading_discretization)] for b in range(self.target_num + 1)]
                                 for c in range(self.heading_discretization)] for d in range(self.target_num + 1)]
                               for u in range(self.uav_num)]
            for a in range(1, self.target_num + 1):
                for b in self.discrete_integer_heading:
                    for c in range(1, self.target_num + 1):
                        for d in self.discrete_integer_heading:
                            source_node = self.targets[a - 1] + [self.heading_multiplier * b]
                            end_node = self.targets[c - 1] + [self.heading_multiplier * d]
                            if source_node == end_node:
                                end_node[-1] += 1e-5
                            for u in range(self.uav_num):
                                distance = dubins.shortest_path(source_node, end_node,
                                                                self.uav_turning_radius[u]).path_length()
                                self.cost_graph[u][a][b][c][d] = distance
        # Cost of UAVs to targets or back to base (update real time information in graph)
        for a in range(1, len(self.targets) + 1):
            for b in self.discrete_integer_heading:
                node = self.targets[a - 1] + [self.heading_multiplier * b]
                for u in range(len(self.uav_id)):
                    distance = dubins.shortest_path(self.uav_state[u], node,
                                                    self.uav_turning_radius[u]).path_length()
                    self.cost_graph[u][0][0][a][b] = distance
                    distance = dubins.shortest_path(node, self.uav_base[u], self.uav_turning_radius[u]).path_length()
                    self.cost_graph[u][a][b][0][0] = distance
        for u in range(len(self.uav_id)):
            distance = dubins.shortest_path(self.uav_state[u], self.uav_base[u],
                                            self.uav_turning_radius[u]).path_length()
            self.cost_graph[u][0][0][0][0] = distance

        # GA parameters update
        self.population_size = round(self.total_population_size / len(self.uav_id)) \
            if distributed else self.total_population_size
        self.crossover_num = round((self.population_size - self.elitism_num) * self.crossover_prob)
        self.mutation_num = self.population_size - self.crossover_num - self.elitism_num
        self.lambda_1 = 1 / (sum(self.uav_velocity))

        # Predefined matrix
        self.remaining_targets = [target_id for target_id in range(1, len(self.targets) + 1) if
                                  not self.tasks_status[target_id - 1] == 0]
        self.task_amount_array = [np.count_nonzero(np.array(self.tasks_status) >= 3 - t) for t in range(3)]
        self.task_index_array = [0, self.task_amount_array[0], self.task_amount_array[0] + self.task_amount_array[1]]
        self.target_sequence = [[index for (index, value) in enumerate(self.tasks_status) if value == task_num]
                                for task_num in range(1, 4)]
        self.target_index_array = [0]
        for k, times in enumerate(self.tasks_status):
            self.target_index_array.append(self.target_index_array[k] + times)

        # Modify population
        if population:
            for elite in information.elite_chromosomes:
                for task in clear_task:
                    for site in range(len(elite[0])):
                        if elite[1][site] == task[0] and elite[2][site] == task[1]:
                            for row in elite:
                                row.pop(site)
                            elite[0] = [sequence for sequence in range(1, len(elite[0])+1)]
                            break

            if lost_agent:
                for elite in information.elite_chromosomes:
                    for index, task_type in enumerate(elite[2]):
                        if elite[3][index] not in self.uav_id:
                            elite[3][index] = random.choice(self.uavType_for_missions[task_type - 1])

            if new_target:
                for elite in information.elite_chromosomes:
                    for target in new_task:
                        insert_index = sorted(np.random.choice(len(elite[0])+1, 3))
                        insert_index = [insert_index[i] + i for i in range(3)]
                        task_type = 1
                        for point in insert_index:
                            elite[1].insert(point, target)
                            elite[2].insert(point, task_type)
                            elite[3].insert(point, random.choice(self.uavType_for_missions[task_type-1]))
                            elite[4].insert(point, random.choice(self.discrete_integer_heading))
                            task_type += 1
                    elite[0] = [sequence for sequence in range(1, len(elite[1]) + 1)]

            if new_target or clear_task or lost_agent:
                population = self.generate_population()
            population.extend([self.Chromosome(elite) for elite in information.elite_chromosomes
                               if len(elite[0]) == sum(self.tasks_status)])
        return population

    def run_GA(self, iteration, uav_message, population=None, distributed=False):
        fitness_convergence = []
        population = self.information_setting(uav_message, population, distributed)
        residual_tasks = sum(self.tasks_status)
        if residual_tasks != 0:
            self.crossover_operators_prob = [0, 1] if residual_tasks <= 1 else [0.5, 0.5]
            if not population:
                try:
                    population = self.generate_population()
                except IndexError:
                    return [[] for _ in range(5)], 1e5, [], 0
                iteration -= 1
            self.fitness_evaluation(population)
            wheel = self.get_roulette_wheel(population)
            fitness_convergence.append(1 / max([_.fitness_value for _ in population]))
            for _ in tqdm(range(iteration)):
                new_population = []
                new_population.extend(self.elitism_operator(population))
                new_population.extend(self.crossover_operator(wheel, population))
                new_population.extend(self.mutation_operator(wheel, population))
                self.fitness_evaluation(new_population)
                wheel = self.get_roulette_wheel(new_population)
                population = new_population
                fitness_convergence.append(1 / max([_.fitness_value for _ in population]))
            return self.find_best_solution(population), population, fitness_convergence
        else:
            return [[] for _ in range(5)], 0, [], 0

    def run_GA_time_period_version(self, time_interval, uav_message, population=None, update=True, distributed=False):
        iteration = 0
        start_time = time.time()
        if update:
            population = self.information_setting(uav_message, population, distributed)
        residual_tasks = sum(self.tasks_status)
        if residual_tasks != 0:
            self.crossover_operators_prob = [0, 1] if residual_tasks <= 1 else [0.5, 0.5]
            if not population:
                population = self.generate_population()
            self.fitness_evaluation(population)
            wheel = self.get_roulette_wheel(population)
            while time.time() - start_time <= time_interval:
                iteration += 1
                new_population = []
                new_population.extend(self.elitism_operator(population))
                new_population.extend(self.crossover_operator(wheel, population))
                new_population.extend(self.mutation_operator(wheel, population))
                self.fitness_evaluation(new_population)
                wheel = self.get_roulette_wheel(new_population)
                population = new_population
            return self.find_best_solution(population), population
        else:
            chromosome = self.Chromosome([[] for _ in range(5)])
            self.fitness_evaluation([chromosome])
            return chromosome, []

    def run_RS(self, iteration, uav_message, population=None):
        a = []
        population = self.generate_population()
        self.fitness_evaluation(population)
        fitness = [_.fitness_value for _ in population]
        a.append(1/max(fitness))
        iteration -= 1
        for _ in range(iteration):
            new_population = self.generate_population()
            self.fitness_evaluation(new_population)
            new_fitness = [_.fitness_value for _ in population]
            for i, chromosome in enumerate(new_population):
                if new_fitness[i] > min(fitness):
                    fitness[fitness.index(min(fitness))] = new_fitness[i]
                    population[i] = chromosome
            a.append(1/max(fitness))
        return self.find_best_solution(population), population, a

    @staticmethod
    def find_best_solution(population):
        fitness = 0
        index = 0
        for i, chromosome in enumerate(population):
            if chromosome.fitness_value > fitness:
                fitness = chromosome.fitness_value
                index = i
        return population[index]

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
                dubins_path = dubins.shortest_path(sp, gp, self.uav_turning_radius[c])
                path, _ = dubins_path.sample_many(.1)
                route_[0].extend([b[0] for b in path])
                route_[1].extend([b[1] for b in path])
                distance += dubins_path.path_length()
                try:
                    time_[a].append(distance / self.uav_velocity[c])
                except IndexError:
                    pass
            try:
                arrow_.extend(
                    [[route_[0][arr], route_[1][arr], route_[0][arr + 100], route_[1][arr + 100]]
                     for arr in range(0, len(route_[0]), 15000)])
            except IndexError:
                pass
            return distance, route_, arrow_

        fitness, mission_time, total_distance, penalty = self.objectives_evaluation(best_solution)
        best_solution = best_solution.chromosome
        print(f'Chromosome: \n{np.array(best_solution)}')
        print("==============================================================================")
        uav_num = len(self.uav_id)
        dist = np.zeros(uav_num)
        task_sequence_state = [[] for _ in range(uav_num)]
        task_route = [[] for _ in range(uav_num)]
        route_state = [[] for _ in range(uav_num)]
        arrow_state = [[] for _ in range(uav_num)]
        for j in range(len(best_solution[0])):
            assign_uav = self.uav_id.index(best_solution[3][j])
            assign_target = best_solution[1][j]
            assign_heading = best_solution[4][j] * self.heading_multiplier * 180 / np.pi
            task_sequence_state[assign_uav].append([
                self.targets[assign_target - 1][0], self.targets[assign_target - 1][1],
                assign_heading])
            task_route[assign_uav].extend([[assign_target, best_solution[2][j]]])
        for j in range(uav_num):
            task_sequence_state[j] = [self.uav_state[j]] + task_sequence_state[j] + [self.uav_base[j]]
            dist[j], route_state[j], arrow_state[j] = dubins_plot(task_sequence_state[j], j, task_route[j])

        task_type = ["classify", "attack", "verify"]
        uav_type = ["Surveillance UAV", "Attack UAV", "Munition UAV"]
        print("Tasks assigned: ")
        for j in range(len(task_route)):
            print(f'\nUAV{self.uav_id[j]} ({uav_type[self.uav_type[j] - 1]}): ')
            for k in range(len(task_route[j])):
                print(f'Target{task_route[j][k][0]} {task_type[task_route[j][k][1] - 1]} task')

        print("==============================================================================")
        print("Results: ")
        print(f'Mission time: {np.round(mission_time, 3)} (sec)')
        print(f'Total distance: {np.round(total_distance, 3)} (m)')
        print(f'Cost value: {np.round(1 / fitness, 3)}')
        print(f'Penalty for task sequence constraints: {np.round(penalty, 3)}')
        print("==============================================================================")

        color_style = ['tab:blue', 'tab:green', 'tab:orange', '#DC143C', '#808080', '#030764', '#06C2AC', '#008080',
                       '#DAA520', '#580F41', '#7BC8F6', '#C875C4']
        font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 8}
        font0 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
        font1 = {'family': 'Times New Roman', 'weight': 'normal', 'color': 'm', 'size': 8}
        font2 = {'family': 'Times New Roman', 'weight': 'normal', 'color': 'r', 'size': 8}
        if curve:
            plt.subplot(122)
            plt.plot([b for b in range(1, len(curve) + 1)], curve, '-')
            plt.grid()
            plt.title("Convergence", font0)
            plt.xlabel("Iteration", font0)
            plt.ylabel("Cost", font0)
            plt.subplot(121)
        else:
            fig, ax = plt.subplots()
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels]
        for i in range(uav_num):
            plt.plot(route_state[i][0], route_state[i][1], '-', linewidth=0.8, color=color_style[i], label=f'UAV {self.uav_id[i]}')
            plt.text(self.uav_state[i][0]-100, self.uav_state[i][1]-200, f'UAV {self.uav_id[i]}', font)
            plt.axis("equal")
            for arrow in arrow_state[i]:
                plt.arrow(arrow[0], arrow[1], arrow[2] - arrow[0], arrow[3] - arrow[1], width=16, color=color_style[i])
        plt.plot([x[0] for x in self.uav_state], [x[1] for x in self.uav_state], 'k^', markerfacecolor='none', markersize=8)
        plt.plot([b[0] for b in self.targets], [b[1] for b in self.targets], 'ms', label='Target position',
                 markerfacecolor='none', markersize=6)
        plt.plot([x[0] for x in self.uav_base], [x[1] for x in self.uav_base], 'r*', markerfacecolor='none', markersize=10, label='Base')
        for t in self.targets:
            plt.text(t[0]+100, t[1]+100, f'Target {self.targets.index(t)+1}', font1)
        for b in self.uav_base:
            plt.text(b[0]-100, b[1]-200, f'Base', font2)
        plt.legend(loc='upper right', prop=font)
        plt.title("Routes", font0)
        plt.xlabel('East, m', font0)
        plt.ylabel('North, m', font0)
        plt.show()


class InformationOfUAVs(object):
    def __init__(self, uav_id, uav_type, uav_states, cruising_velocities, minimum_tuning_radii, base_configurations,
                 uav_best_solution=None, new_targets=None, tasks_completed=None):
        self.uav_id = uav_id
        self.uav_type = uav_type
        self.uav_states = uav_states
        self.cruising_speed = cruising_velocities
        self.turning_radii = minimum_tuning_radii
        self.base = base_configurations
        self.new_targets = new_targets if new_targets else []
        self.tasks_completed = tasks_completed if tasks_completed else []
        self.elite_chromosomes = uav_best_solution if uav_best_solution else []


if __name__ == "__main__":
    # targets
    targets = [[3100, 2200], [500, 3700], [2300, 2500], [2000, 3900], [4450, 3600], [4630, 4780], [1400, 4500]]
    # UAVs
    UAV_ID = [1, 2, 3, 4, 5, 6]
    UAV_type = [1, 2, 3, 1, 3, 2]  # 1: surveillance, 2: attack, 3: munition
    cruising_speed = [70, 80, 90, 60, 100, 80]  # (m/s)
    minimum_turning_radii = [200, 250, 300, 180, 300, 260]
    UAV_state = [[1000, 300, -np.pi], [1500, 700, np.pi / 2], [3000, 0, np.pi / 3],
                 [1800, 400, -20 * np.pi / 180], [2200, 280, 45 * np.pi / 180],
                 [4740, 300, 140 * np.pi / 180]]  # [East(m), North(m), heading angle(rad)]
    base_configuration = [[0, 0, -np.pi / 2], [0, 0, -np.pi / 2], [1000, 6000, np.pi / 2],
                          [1000, 6000, np.pi / 2], [4000, 5500, np.pi / 3],
                          [4000, 5500, np.pi / 3]]  # [East(m), North(m), runway direction(rad)]
    uav_info = InformationOfUAVs(UAV_ID, UAV_type, UAV_state, cruising_speed, minimum_turning_radii, base_configuration)

    population_size = 300
    iteration = 100
    ga = GA_SEAD(targets, population_size)
    solution, ga_population, convergence = ga.run_GA(iteration, uav_info)
    ga.plot_result(solution, convergence)
