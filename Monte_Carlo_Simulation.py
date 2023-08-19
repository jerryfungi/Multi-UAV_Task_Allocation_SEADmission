import random
import time
import math
import numpy as np
import copy
from matplotlib import pyplot as plt
import dubins
import multiprocessing as mp
from GA_SEAD_process import *
from PSO_SEAD import *


def DGA(targets, uav_msg, queue, index, iteration):
    share_list = [i for i in range(len(uav_msg[0])) if not i == index]
    dga = GA_SEAD(targets, round(300/len(uav_msg[0])))
    # dga = GA_SEAD(targets, 300)
    population = None
    fitness_list = []
    for i in range(int(iteration/100)):
        ans, fitness, population, a = dga.run_GA(100, uav_msg, population)
        fitness_list.extend(a)
        for port in share_list:
            queue[port].put(ans)
        subpop = []
        while not len(subpop) == len(uav_msg[0])-1:
            subpop.append(queue[index].get())
        population.extend(subpop)
    _, ti, di, pe = dga.chromosome_objectives_evaluate(ans)
    queue[-1].put([fitness_list, ans, [ti, di, pe]])
    # fitness, _ = dga.fitness_evaluate(population)
    # fitness_list.append(1/max(fitness))
    # print(f'best solution: {population[fitness.index(max(fitness))]}')
    # print(f'objectives: {1/max(fitness)}')
    # dga.plot_result(population[fitness.index(max(fitness))], fitness_list)


if __name__ == "__main__":
    monte_carlo = 50
    generation = 300
    history_cost = 1e5
    best_chromosome = []
    convergence = [[0 for g in range(generation)] for _ in range(3)]
    object_value = [[] for _ in range(3)]
    mission_time = [[] for _ in range(3)]
    total_distance = [[] for _ in range(3)]
    penalty = [[] for _ in range(3)]
    process_time = [[] for _ in range(3)]

    targets_sites = [[3100, 2200], [500, 3700], [2300, 2500], [2000, 3900], [4450, 3600], [4630, 4780], [1400, 4500],
                     [3300, 3415], [1640, 1700], [4230, 1700], [500, 2200], [3000, 4500], [5000, 2810]]
    uavs = [[i for i in range(1, 12)], [1, 2, 3, 1, 3, 2, 1, 2, 3, 1, 2],
            [70, 80, 90, 60, 100, 80, 75, 90, 85, 70, 65],
            [200, 250, 300, 180, 300, 260, 225, 295, 250, 200, 170],
            [[1000, 300, -np.pi], [1500, 700, np.pi / 2], [3000, 0, np.pi / 3], [1800, 400, -20 * np.pi / 180],
             [2200, 280, 45 * np.pi / 180], [4740, 300, 140 * np.pi / 180], [4000, 100, 70 * np.pi / 180],
             [3500, 450, -75 * np.pi / 180], [5000, 900, -115 * np.pi / 180], [2780, 500, -55 * np.pi / 180],
             [4000, 600, 85 * np.pi / 180]],
            [[0, 0, -np.pi / 2] for _ in range(11)]]
    targets_sites = targets_sites[:4]
    uavs = [[row[j] for j in range(3)] for row in uavs] + [[], [], [], []]
    # targets_sites = targets_sites[:7]
    # uavs = [[row[j] for j in range(6)] for row in uavs] + [[], [], [[4, 1], [5, 1]], []]
    # targets_sites = targets_sites[:]
    # uavs = uavs + [[], [], [[4, 1], [5, 1], [9, 1], [9, 2], [8, 1], [8, 2], [13, 1]], []]

    for j in range(monte_carlo):
        broadcast = [mp.Queue() for _ in range(len(uavs[0])+1)]
        multi_process = [mp.Process(target=DGA, args=(targets_sites, uavs, broadcast, a, generation))
                         for a in range(len(uavs[0]))]
        start = time.time()
        # s = time.time()
        for p in multi_process:
            p.start()
        # for p in multi_process:
        #     p.join()
        # print(time.time() - s)
        f = [[[], [], []] for _ in range(len(uavs[0]))]
        for k in f:
            if not k[0]:
                solution = broadcast[-1].get()
                k[0].extend(solution[0])
                k[1].extend(solution[1])
                k[2].extend(solution[2])
        process_time[2].append(time.time()-start)
        print(time.time()-start)
        # print(len(f))
        # print(f)
        curve = [min([a[0][g] for a in f]) for g in range(generation)]
        convergence[2] = list(np.add(convergence[2], curve))
        total_fit = [a[0][-1] for a in f]
        object_value[2].append(min(total_fit))
        mission_time[2].append(min([t[2][0] for t in f]))
        total_distance[2].append(min([t[2][1] for t in f]))
        penalty[2].append(min([t[2][2] for t in f]))
        best_chromosome = f[total_fit.index(min(total_fit))][1] if min(total_fit) < history_cost else best_chromosome
        history_cost = min(total_fit) if min(total_fit) < history_cost else history_cost

    print(f"best_solution = {best_chromosome}")

    # print('cga')
    cga = GA_SEAD(targets_sites, 300)
    for j in range(monte_carlo):
        start = time.time()
        solution, fitness_value, ga_population, b = cga.run_GA(generation, uavs)
        process_time[1].append(time.time() - start)
        convergence[1] = list(np.add(convergence[1], b))
        object_value[1].append(1/fitness_value)
        _, t, d, p = cga.chromosome_objectives_evaluate(solution)
        mission_time[1].append(t)
        total_distance[1].append(d)
        penalty[1].append(p)
        # print(f'CGA: {1/fitness_value}')

    # print('pso')
    # pso = PSO_SEAD(targets_sites)
    # for j in range(monte_carlo):
    #     gbest, c = pso.run_PSO(generation, uavs)
    #     convergence[1] = list(np.add(convergence[1], c))
    #     object_value[1].append(1/gbest[0])
    #     # print(f'PSO: {1/fitness_value}')

    # print('rs')
    for j in range(monte_carlo):
        start = time.time()
        solution, fitness_value, ga_population, d = cga.run_RS(generation, uavs)
        process_time[0].append(time.time() - start)
        convergence[0] = list(np.add(convergence[0], d))
        object_value[0].append(1/fitness_value)
        _, t, d, p = cga.chromosome_objectives_evaluate(solution)
        mission_time[0].append(t)
        total_distance[0].append(d)
        penalty[0].append(p)
        # print(f'RS: {1/fitness_value}')

    print(f"convergence = {convergence} \n"
          f"object_value = {object_value} \n"
          f"mission_time = {mission_time} \n"
          f"total_distance = {total_distance} \n"
          f"penalty = {penalty} \n"
          f"process_time = {process_time}")

