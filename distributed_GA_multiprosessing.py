import random
import time
import math
import numpy as np
import threading
import multiprocessing as mp
import queue
from matplotlib import pyplot as plthbb


class DistributedGA(mp.Process):
    def __init__(self, uav_sites, targets_sites, uav_specification, solution_sharing, thread_id):
        mp.Process.__init__(self)
        # configuration
        self.uav_sites = uav_sites
        self.targets_sites = targets_sites
        self.uav_num = len(uav_sites)
        self.targets_num = len(targets_sites)
        self.uav_velocity = uav_specification
        # self.uav_Rmin = uav_specification[1]
        # initial mapping
        # self.cost_matrix = np.zeros((self.uav_num, self.targets_num+1, self.targets_num+1))
        # for x in range(self.targets_num):
        #     self.cost_matrix[:, 0, x+1], self.cost_matrix[:, x+1, 0] = \
        #         [self.time_cost(self.uav_sites[z], self.targets_sites[x]) for z in range(self.uav_num)], \
        #         [self.time_cost(self.uav_sites[z], self.targets_sites[x]) for z in range(self.uav_num)]
        #     for y in range(self.targets_num):
        #         self.cost_matrix[:, x+1, y+1] = self.time_cost(self.targets_sites[x], targets_sites[y])
        self.node = [[] for _ in range(self.uav_num)]
        self.cost_matrix = [[] for _ in range(self.uav_num)]
        for i in range(self.uav_num):
            self.node[i].extend([self.uav_sites[i]])
            self.node[i].extend(self.targets_sites)
        for i in range(self.uav_num):
            self.cost_matrix[i] = [(lambda x: [math.sqrt(math.pow(self.node[i][x][0] - self.node[i][y][0], 2) +
                                                          math.pow(self.node[i][x][1] - self.node[i][y][1], 2)) for y
                                               in range(self.targets_num + 1)])(j) for j in range(self.targets_num + 1)]
        # communication
        self.thread_id = thread_id
        self.broadcast_list = [agents for agents in range(self.uav_num) if not agents == self.thread_id]
        self.subpopulation = solution_sharing
        # GA parameters
        # self.pop_size = math.ceil(300/self.uav_num)
        self.pop_size = 100
        self.crossover_prob = 1
        self.mutation_prob = 0.6
        self.elitism_num = 2 if self.pop_size % 2 == 0 else 1
        self.iteration = 200

    def time_cost(self, node1, node2):
        return np.sqrt((node2[0] - node1[0]) ** 2 + (node2[1] - node1[1]) ** 2) / self.uav_velocity

    def fitness_evaluate(self, population):
        # def fitness_function(total_cost):
        #     return 1/(max(total_cost) + 0.01 * sum(total_cost))
        # fitness = np.zeros(len(population))
        # for i, chromosome in enumerate(population):
        #     decision_variables = np.zeros((self.uav_num, self.targets_num+1, self.targets_num+1), dtype=int)
        #     pre = np.zeros(self.uav_num, dtype=int)
        #     for j in range(self.targets_num):
        #         decision_variables[chromosome[1][j]-1, pre[chromosome[1][j]-1], chromosome[0][j]-1] = 1
        #         pre[chromosome[1][j]-1] = chromosome[0][j]-1
        #     decision_variables[:, pre[:], 0] = 1
        #     cost = [np.sum(cv) for cv in np.multiply(self.cost_matrix, decision_variables)]
        #     fitness[i] = fitness_function(cost)
        # roulette_wheel = fitness / np.sum(fitness)
        # return fitness, roulette_wheel
        fitness_value = []
        for k in range(len(population)):
            route = [[0] for _ in range(self.uav_num)]
            dist = [0 for _ in range(self.uav_num)]
            for j in range(self.uav_num):
                pre_pos = 0
                for i in range(len(population[k][0])):
                    if population[k][1][i] == j + 1:
                        target = population[k][0][i]
                        route[j].extend([target])
                        dist[j] += self.cost_matrix[j][pre_pos][target]
                        pre_pos = target
                route[j].extend([0])
                dist[j] += self.cost_matrix[j][pre_pos][0]
            fitness_value.extend([1 / (max(dist) + 0.01 * sum(dist))])
        return fitness_value

    def roulette_wheel(self, fitness):
        return np.array(fitness) / np.sum(fitness)

    def generate_population(self):
        def generate_chromosome():
            chromosome = [[i+1 for i in range(self.targets_num)],
                          [random.randint(1, self.uav_num) for _ in range(self.targets_num)]]
            random.shuffle(chromosome[0])
            return chromosome
        return [generate_chromosome() for _ in range(self.pop_size)]

    def selection(self, population, roulette_wheel):
        return np.random.choice(np.arange(len(population)), 2, replace=False, p=roulette_wheel)

    def crossover(self, parent_1, parent_2):
        if random.random() < self.crossover_prob:
            cutpoint = random.sample(range(1, self.targets_num - 1), 2)
            cutpoint.sort()
            remain_1, remain_2 = [[], []], [[], []]
            child_1, child_2 = [[], []], [[], []]
            # order crossover
            for i in range(self.targets_num):
                if parent_2[0][i] not in parent_1[0][cutpoint[0]:cutpoint[1]]:
                    remain_1[0].append(parent_2[0][i])
                    remain_1[1].append(parent_2[1][i])
                if parent_1[0][i] not in parent_2[0][cutpoint[0]:cutpoint[1]]:
                    remain_2[0].append(parent_1[0][i])
                    remain_2[1].append(parent_1[1][i])
            for i in range(2):
                child_1[i].extend(
                    remain_1[i][:cutpoint[0]] + parent_1[i][cutpoint[0]:cutpoint[1]] + remain_1[i][cutpoint[0]:])
                child_2[i].extend(
                    remain_2[i][:cutpoint[0]] + parent_2[i][cutpoint[0]:cutpoint[1]] + remain_2[i][cutpoint[0]:])
            return child_1, child_2
        else:
            return parent_1, parent_2

    def mutation(self, chromosome):
        if random.random() < self.mutation_prob:
            gene = [[], []]
            mutpoint = random.randint(0, self.targets_num - 1)
            assign = random.choice([i for i in range(1, self.uav_num + 1) if not i == chromosome[1][mutpoint]])
            gene[0].extend(chromosome[0][:])
            gene[1].extend(chromosome[1][:mutpoint] + [assign] + chromosome[1][mutpoint + 1:])
            return gene
        else:
            return chromosome

    def substitute(self, population, fitness, new_pop, new_fit):
        for i in range(len(new_pop)):
            if new_fit[i] > min(fitness):
                population[np.argmin(fitness)] = new_pop[i]
                fitness[np.argmin(fitness)] = new_fit[i]

    def elitism(self, population, fitness):
        eli_id = sorted(range(self.pop_size), key=lambda v: fitness[v], reverse=True)[:self.elitism_num]
        return [population[_] for _ in eli_id]

    def chromosome_transmit(self, population, fitness):
        for port in self.broadcast_list:
            self.subpopulation[port].put(population[np.argmax(fitness)])

    def chromosome_receive(self, population):
        subpopulation = []
        while self.subpopulation[self.thread_id].qsize() > 0:
            try:
                new_chromosome = self.subpopulation[self.thread_id].get(timeout=1e-5)
                subpopulation.append(new_chromosome)
            except queue.Empty:
                pass
        if not subpopulation == []:
            # print(subpopulation)
            # weak_individual = sorted(range(len(subpopulation)), key=lambda v: fitness[v], reverse=True)
            # for i in range(len(subpopulation)):
            #     population[weak_individual[i]] = subpopulation[i]
            population.extend(subpopulation)

    def run(self):
        population = self.generate_population()
        fitness = self.fitness_evaluate(population)
        performance = [np.max(fitness)]
        for n in range(1, self.iteration+1):
            new_population = []
            wheel = self.roulette_wheel(fitness)
            if n % 30 == 0:
                self.chromosome_transmit(population, fitness)
            for m in range(0, self.pop_size, 2):
                parents = self.selection(population, wheel)
                offspring = self.crossover(population[parents[0]], population[parents[1]])
                offspring_1 = self.mutation(offspring[0])
                offspring_2 = self.mutation(offspring[1])
                new_population.extend([offspring_1, offspring_2])
            self.chromosome_receive(new_population)
            new_fitness = self.fitness_evaluate(new_population)
            self.substitute(population, fitness, new_population, new_fitness)
            new_population.clear()
            new_fitness.clear()
            performance.append(np.max(fitness))
        print(1/performance[-1])


def run_process():
    target_sites = [[-65, 15], [27, 66], [-51, 58], [-19, 34], [77, 25], [50, 50], [-23, 91], [77, 77], [0, 39],
                    [71, 95], [-25, 10], [0, 91], [30, 30], [15, 15], [-100, 24], [38, 75]]
    # uavs_sites = [[0,0],[50,25],[-70,87],[23,10],[21,2]]
    uavs_sites = [[0, 0] for _ in range(4)]
    print(f'cpu:{mp.cpu_count()}')
    uav_configuration = 1
    broadcast = [mp.Queue() for _ in range(len(uavs_sites))]
    GA_process = [DistributedGA(uavs_sites, target_sites, uav_configuration, broadcast, a) for a in
                  range(len(uavs_sites))]
    start = time.time()
    for process in GA_process:
        process.start()
    for process in GA_process:
        process.join()
    print(f'Done!!'
          f'past time:{time.time() - start}sec')


def distributed_ga(uav_sites, targets_sites, uav_specification, solution_sharing, thread_id):
    start_t = time.time()
    # configuration
    uav_num = len(uav_sites)
    targets_num = len(targets_sites)
    uav_velocity = uav_specification
    node = [[] for _ in range(uav_num)]
    cost_matrix = [[] for _ in range(uav_num)]
    for _ in range(uav_num):
        node[_].extend([uav_sites[_]])
        node[_].extend(targets_sites)
    for _ in range(uav_num):
        cost_matrix[_] = [(lambda x: [math.sqrt(math.pow(node[_][x][0] - node[_][y][0], 2) +
                                                     math.pow(node[_][x][1] - node[_][y][1], 2)) for y in
                                      range(targets_num + 1)])(d) for d in range(targets_num + 1)]
    # communication
    thread_id = thread_id
    broadcast_list = [agents for agents in range(uav_num) if not agents == thread_id]
    subpopulation = solution_sharing
    # GA parameters
    pop_size = 80
    crossover_prob = 1
    mutation_prob = 0.5
    iteration = 300

    def fitness_evaluate(pop):
        fitness_value = []
        for k in range(len(pop)):
            dist = [0 for _ in range(uav_num)]
            for j in range(uav_num):
                pre_pos = 0
                for i in range(len(pop[k][0])):
                    if pop[k][1][i] == j + 1:
                        target = pop[k][0][i]
                        dist[j] += cost_matrix[j][pre_pos][target]
                        pre_pos = target
                dist[j] += cost_matrix[j][pre_pos][0]
            fitness_value.extend([1 / (max(dist) + 0.01 * sum(dist))])
        return fitness_value

    def roulette_wheel(fit):
        return np.array(fit) / np.sum(fit)

    def generate_population():
        def generate_chromosome():
            chromosome = [[i+1 for i in range(targets_num)],
                          [random.randint(1, uav_num) for _ in range(targets_num)]]
            random.shuffle(chromosome[0])
            return chromosome
        return [generate_chromosome() for _ in range(pop_size)]

    def selection(pop, r_wheel):
        return np.random.choice(np.arange(len(pop)), 2, replace=False, p=r_wheel)

    def crossover(parent_1, parent_2):
        if random.random() < crossover_prob:
            cutpoint = random.sample(range(1, targets_num - 1), 2)
            cutpoint.sort()
            remain_1, remain_2 = [[], []], [[], []]
            child_1, child_2 = [[], []], [[], []]
            # order crossover
            for i in range(targets_num):
                if parent_2[0][i] not in parent_1[0][cutpoint[0]:cutpoint[1]]:
                    remain_1[0].append(parent_2[0][i])
                    remain_1[1].append(parent_2[1][i])
                if parent_1[0][i] not in parent_2[0][cutpoint[0]:cutpoint[1]]:
                    remain_2[0].append(parent_1[0][i])
                    remain_2[1].append(parent_1[1][i])
            for i in range(2):
                child_1[i].extend(
                    remain_1[i][:cutpoint[0]] + parent_1[i][cutpoint[0]:cutpoint[1]] + remain_1[i][cutpoint[0]:])
                child_2[i].extend(
                    remain_2[i][:cutpoint[0]] + parent_2[i][cutpoint[0]:cutpoint[1]] + remain_2[i][cutpoint[0]:])
            return child_1, child_2
        else:
            return parent_1, parent_2

    def mutation(chromosome):
        if random.random() < mutation_prob:
            gene = [[], []]
            mutpoint = random.randint(0, targets_num - 1)
            assign = random.choice([i for i in range(1, uav_num + 1) if not i == chromosome[1][mutpoint]])
            gene[0].extend(chromosome[0][:])
            gene[1].extend(chromosome[1][:mutpoint] + [assign] + chromosome[1][mutpoint + 1:])
            return gene
        else:
            return chromosome

    def substitute(pop, fit, new_pop, new_fit):
        for i in range(len(new_pop)):
            if new_fit[i] > min(fit):
                pop[np.argmin(fit)] = new_pop[i]
                fit[np.argmin(fit)] = new_fit[i]

    def chromosome_transmit(pop, fit):
        for port in broadcast_list:
            subpopulation[port].put(pop[np.argmax(fit)])

    def chromosome_receive(pop):
        subpop = []
        while not subpopulation[thread_id].empty():
            try:
                new_chromosome = subpopulation[thread_id].get(timeout=1)
                subpop.append(new_chromosome)
            except queue.Empty:
                pass
        if not subpop == []:
            pop.extend(subpop)

    population = generate_population()
    fitness = fitness_evaluate(population)
    performance = [max(fitness)]
    for n in range(1, iteration + 1):
        new_population = []
        wheel = roulette_wheel(fitness)
        if n % 50 == 0:
            chromosome_transmit(population, fitness)
        for m in range(0, pop_size, 2):
            parents = selection(population, wheel)
            offspring = crossover(population[parents[0]], population[parents[1]])
            offspring_1 = mutation(offspring[0])
            offspring_2 = mutation(offspring[1])
            new_population.extend([offspring_1, offspring_2])
        chromosome_receive(new_population)
        new_fitness = fitness_evaluate(new_population)
        substitute(population, fitness, new_population, new_fitness)
        performance.append(np.max(fitness))
    print(f'Process_{thread_id} past time: {round(time.time()-start_t, 5)}, ')
    print(f'Process_{thread_id} chromosome:{population[fitness.index(max(fitness))]}')
    print(1 / performance[-1])


if __name__ == '__main__':
    target_sites = [[-65, 15], [27, 66], [-51, 58], [-19, 34], [77, 25], [50, 50], [-23, 91], [77, 77], [0, 39],
                    [71, 95], [-25, 10], [0, 91], [30, 30], [15, 15], [-100, 24], [38, 75]]
    # uavs_sites = [[0,0],[50,25],[-70,87],[23,10],[21,2]]
    uavs_sites = [[0, 0] for _ in range(4)]
    print(f'cpu:{mp.cpu_count()}')
    uav_configuration = 1
    broadcast = [mp.Queue() for _ in range(len(uavs_sites))]
    multi_process = [mp.Process(target=distributed_ga, args=(uavs_sites, target_sites, uav_configuration, broadcast, a))
                     for a in range(len(uavs_sites))]
    start = time.time()
    for p in multi_process:
        p.start()
    for p in multi_process:
        p.join()
    print(f'Done!!'
          f'past time: {round(time.time() - start, 3)} sec')
