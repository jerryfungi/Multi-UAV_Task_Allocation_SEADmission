import random
from random import randrange
import time
import math
from matplotlib import pyplot as plt
import numpy as np
import multiprocessing as mp


class GA_vrp(object):

    def __init__(self, targets_sites, uavs_sites):
        # input data
        self.targets_sites = targets_sites
        self.uavs_sites = uavs_sites
        # GA parameters
        multiplier = 3
        self.pop_size = 100*multiplier
        self.pop_cros = 66*multiplier
        self.pop_mut = 32*multiplier
        self.pop_eli = 2*multiplier
        self.prob_cros = 1
        self.prob_mut = 0.7
        self.iteration = 170
        # mapping
        self.uav_num = len(uavs_sites)
        self.target_num = len(targets_sites)
        self.node = [[] for _ in range(self.uav_num)]
        self.distance_mat = [[] for _ in range(self.uav_num)]
        for i in range(self.uav_num):
            self.node[i].extend([self.uavs_sites[i]])
            self.node[i].extend(self.targets_sites)
        for i in range(self.uav_num):
            self.distance_mat[i] = [(lambda x: [math.sqrt(math.pow(self.node[i][x][0]-self.node[i][y][0],2) +
                                                   math.pow(self.node[i][x][1]-self.node[i][y][1],2)) for
                                                y in range(self.target_num+1)])(j) for j in range(self.target_num+1)]
        # constraints
        self.uav_velocity = 1
        # performance
        self.training_curve = []

    def fitness(self, population):
        fitness_value, roulette_wheel = [], []
        for k in range(len(population)):
            route = [[0] for _ in range(self.uav_num)]
            dist = [0 for _ in range(self.uav_num)]
            for j in range(self.uav_num):
                pre_pos = 0
                for i in range(len(population[k][0])):
                    if population[k][1][i] == j+1:
                        target = population[k][0][i]
                        route[j].extend([target])
                        dist[j] += self.distance_mat[j][pre_pos][target]
                        pre_pos = target
                route[j].extend([0])
                dist[j] += self.distance_mat[j][pre_pos][0]
            fitness_value.extend([1/(max(dist)+0.01*sum(dist))])
        sum_fit = sum(fitness_value)
        for i in range(len(fitness_value)):
            roulette_wheel.append(fitness_value[i] / sum_fit)
        return fitness_value, max(fitness_value), roulette_wheel

    def initiate_population(self):
        def generate_chromosome():
            chromosome = [[],[]]
            for i in range(self.target_num):
                chromosome[0].append(i+1) # target id
                chromosome[1].append(random.randint(1, self.uav_num)) # uav id
            random.shuffle(chromosome[0])
            return chromosome
        return [generate_chromosome() for i in range(self.pop_size)]

    def selection(self, roulette_wheel):
        p1, p2 = random.choices(range(self.pop_size), weights=roulette_wheel, k=2)
        return p1, p2

    def crossover(self, parent_1, parent_2):
        cutpoint = random.sample(range(self.target_num), 2)
        cutpoint_1, cutpoint_2 = min(cutpoint), max(cutpoint)
        remain_1, remain_2 = [[], []], [[], []]
        child_1, child_2 = [[], []], [[], []]
        # order crossover
        for i in range(self.target_num):
            if parent_2[0][i] not in parent_1[0][cutpoint_1:cutpoint_2]:
                remain_1[0].append(parent_2[0][i])
                remain_1[1].append(parent_2[1][i])
            if parent_1[0][i] not in parent_2[0][cutpoint_1:cutpoint_2]:
                remain_2[0].append(parent_1[0][i])
                remain_2[1].append(parent_1[1][i])
        for i in range(2):
            child_1[i].extend(remain_1[i][:cutpoint_1] + parent_1[i][cutpoint_1:cutpoint_2] + remain_1[i][cutpoint_1:])
            child_2[i].extend(remain_2[i][:cutpoint_1] + parent_2[i][cutpoint_1:cutpoint_2] + remain_2[i][cutpoint_1:])
        return child_1, child_2

    def mutation(self, chromosome):
        gene = [[], []]
        if random.random() <= 0.5 and not self.uav_num == 1:  # mutate agent
            mutpoint = random.randint(0, self.target_num - 1)
            assign = random.choice([i for i in range(1, self.uav_num + 1) if i not in [chromosome[1][mutpoint]]])
            gene[0].extend(chromosome[0][:])
            gene[1].extend(chromosome[1][:mutpoint] + [assign] + chromosome[1][mutpoint + 1:])
        else:                                                 # inverse mutation
            cutpoint = random.sample(range(1, self.target_num - 1), 2)
            cutpoint.sort()
            for i, row in enumerate(chromosome):
                inverse_gene = row[cutpoint[0]:cutpoint[1]]
                inverse_gene.reverse()
                gene[i].extend(row[:cutpoint[0]] + inverse_gene + row[cutpoint[1]:])
        return gene

    def elitism(self, fitness_value_):
        elitism = []
        fitness_value = fitness_value_[:]
        for i in range(self.pop_eli):
            elitism.append(fitness_value.index(max(fitness_value)))
            fitness_value[fitness_value.index(max(fitness_value))] = 0
        return elitism

    def adaptive_setting(self, Nit):
        Ncr = round((self.pop_size-self.pop_eli)*math.exp(-Nit/self.iteration))
        return Ncr

    def GA_main(self):
        start = time.time()
        population = self.initiate_population()
        fitness = self.fitness(population)
        self.training_curve.append(1/fitness[1])
        itr = [0]
        for i in range(self.iteration):
            itr.append(i+1)
            # cross_num = self.adaptive_setting(itr[-1])
            new_population = []
            elitism = self.elitism(fitness[0])
            for j in range(self.pop_eli):
                new_population.extend([population[elitism[j]]])
            for j in range(0, self.pop_cros, 2):
                parent = self.selection(fitness[2])
                offspring = self.crossover(population[parent[0]], population[parent[1]])
                new_population.extend([offspring[0],offspring[1]])
            for j in range(self.pop_size-len(new_population)):
                chromosome = population[self.selection(fitness[2])[0]]
                new_population.extend([self.mutation(chromosome)])
            fitness = self.fitness(new_population)
            self.training_curve.append(1/fitness[1])
            population = new_population
            self.plot_animation(population[fitness[0].index(max(fitness[0]))], itr)
        print(f'past time = {time.time() - start}')
        self.plot_best(population[fitness[0].index(max(fitness[0]))])

    def GA_prob_version(self):
        a = time.time()
        population = self.initiate_population()
        fitness = self.fitness(population)
        self.training_curve.append(1 / fitness[1])
        for i in range(self.iteration):
            new_population = []
            for j in range(int(self.pop_size / 2)):
                parent = self.selection(fitness[2])
                offspring = self.crossover(population[parent[0]], population[parent[1]])
                new_population.extend([offspring[0],offspring[1]])
            for j in range(self.pop_size):
                if random.random() < self.prob_mut:
                    new_population[j] = self.mutation(new_population[j])
            new_fitness = self.fitness(new_population)
            for j in range(self.pop_size):
                if new_fitness[0][j] > min(fitness[0]):
                    population[fitness[0].index(min(fitness[0]))] = new_population[j]
                    fitness[0][fitness[0].index(min(fitness[0]))] = new_fitness[0][j]
            self.training_curve.append(1 / max(fitness[0]))
        print(time.time()-a)
        print(self.training_curve[-1])
        self.plot_best(population[fitness[0].index(max(fitness[0]))])

    def plot_best(self, best_solution):
        print(f"best gene = {best_solution}")
        route = [[0] for _ in range(self.uav_num)]
        dist = [0 for _ in range(self.uav_num)]
        x = [[uavs_sites[_][0]] for _ in range(self.uav_num)]
        y = [[uavs_sites[_][1]] for _ in range(self.uav_num)]
        for j in range(self.uav_num):
            pre_pos = 0
            for i in range(len(best_solution[0])):
                if best_solution[1][i] == j + 1:
                    route[j].extend([best_solution[0][i]])
                    dist[j] += self.distance_mat[j][pre_pos][best_solution[0][i]]
                    pre_pos = best_solution[0][i]
            dist[j] += self.distance_mat[j][pre_pos][0]
            route[j].extend([0])
            for k in range(1, len(route[j])-1):
                x[j].extend([self.targets_sites[route[j][k]-1][0]])
                y[j].extend([self.targets_sites[route[j][k]-1][1]])
            x[j].extend([uavs_sites[j][0]])
            y[j].extend([uavs_sites[j][1]])
        print(f"route: {route}")
        plt.subplot(121)
        for i in range(len(x)):
            plt.plot(x[i], y[i], marker="o")
        plt.xlabel("X-Axis")
        plt.ylabel("Y-Axis")
        plt.title("route")
        plt.subplot(122)
        plt.plot(range(self.iteration+1), self.training_curve)
        plt.title("cost = {:.3f}".format(max(dist)+0.01*sum(dist)))
        plt.show()

    def plot_animation(self, best_solution, iteration):
        plt.clf()
        # print(f"best gene = {best_solution}")
        route = [[0] for _ in range(self.uav_num)]
        dist = [0 for _ in range(self.uav_num)]
        x = [[self.uavs_sites[_][0]] for _ in range(self.uav_num)]
        y = [[self.uavs_sites[_][1]] for _ in range(self.uav_num)]
        for j in range(self.uav_num):
            pre_pos = 0
            for i in range(len(best_solution[0])):
                if best_solution[1][i] == j + 1:
                    route[j].extend([best_solution[0][i]])
                    dist[j] += self.distance_mat[j][pre_pos][best_solution[0][i]]
                    pre_pos = best_solution[0][i]
            dist[j] += self.distance_mat[j][pre_pos][0]
            route[j].extend([0])
            for k in range(1, len(route[j])-1):
                x[j].extend([self.targets_sites[route[j][k]-1][0]])
                y[j].extend([self.targets_sites[route[j][k]-1][1]])
            x[j].extend([self.uavs_sites[j][0]])
            y[j].extend([self.uavs_sites[j][1]])
        # print(f"route: {route}")
        plt.subplot(121)
        for i in range(len(x)):
            plt.plot(x[i], y[i], marker="^", markersize=5)
        plt.xlabel("X-Axis")
        plt.ylabel("Y-Axis")
        plt.title("route")
        plt.subplot(122)
        plt.plot(iteration, self.training_curve)
        plt.title("cost = {:.3f}".format(max(dist)+0.01*sum(dist)))
        plt.pause(1e-10)


if __name__ == '__main__':
    targets_sites = [[-65,15],[27,66],[-51,58],[-19,34],[77,25],[50,50],[-23,91],[77,77],[0,39],
                     [71,95],[-25,10],[0,91],[30,30],[15,15],[-100,24],[38,75]]
    # targets_sites = [[random.randint(-100, 100), random.randint(50, 100)] for _ in range(30)]
    # uavs_sites = [[0,0],[50,25],[-70,87],[23,10],[21,2]]
    # uavs_sites = [[0, 0] for _ in range(4)]
    uavs_sites = [[0, 0], [75, 0], [-100, 0]]
    gavrp = GA_vrp(targets_sites, uavs_sites)
    gavrp.GA_main()
