import random
import time
import math
import numpy as np
import copy
from matplotlib import pyplot as plt
import dubins
import multiprocessing as mp
from GA_SEAD_process import *
from RS_SEAD import *
from PSO_SEAD import *


def DGA(targets, uav_msg, queue, index, iteration):
    share_list = [i for i in range(len(uav_msg[0])) if not i == index]
    dga = GA_SEAD(targets, round(300/len(uav_msg[0])))
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
    queue[-1].put([fitness_list, ans])
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
    # case 1
    targets_sites = [[3100, 2200], [500, 3700], [2300, 2500], [2000, 3900]]
    uavs = [[1, 2, 3], [1, 2, 3], [70, 80, 90], [200, 250, 300],
            [[1000, 300, -np.pi], [1500, 700, np.pi / 2], [3000, 0, np.pi / 3]],
            [[0, 0, -np.pi / 2], [0, 0, -np.pi / 2], [0, 0, -np.pi / 2]],
            [], [], [], []]
    # case 2
    # targets_sites = [[3850, 650], [3900, 4700], [500, 1500], [1000, 2750], [4450, 3600], [2800, 3900], [800, 3600]]
    # uavs = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 2, 1, 2], [70, 80, 90, 60, 100, 80], [200, 250, 300, 180, 300, 260],
    #         [[500, 300, -60*np.pi/180], [1500, 700, 90*np.pi/180], [200, 1100, 135*np.pi/180],
    #          [3500, 120, 20*np.pi/180], [200, 4600, -45*np.pi/180], [4740, 2500, 115*np.pi/180]],
    #         [[5000, 1000, 0] for _ in range(6)],
    #         [], [], [], []]
    # case 3
    # targets_sites = [[4550, 650], [500, 1500], [1000, 2750], [4450, 3600], [4630, 4780], [800, 3600],
    #                  [3300, 2860], [2000, 2000], [3650, 1700], [2020, 3020]]
    # uavs = [[i for i in range(1, 13)], [2, 2, 3, 1, 1, 3, 1, 2, 2, 1, 2, 2],
    #         [70, 80, 90, 60, 100, 80, 75, 90, 85, 70, 65, 50],
    #         [200, 250, 300, 180, 300, 260, 225, 295, 250, 200, 170, 150],
    #         [[0, 3770, 0 * np.pi / 180], [1500, 700, 90 * np.pi / 180], [200, 900, 135 * np.pi / 180],
    #          [1800, 4500, -20 * np.pi / 180], [200, 2800, 45 * np.pi / 180], [4740, 3000, 140 * np.pi / 180],
    #          [350, 120, 70 * np.pi / 180], [3500, 4500, -75 * np.pi / 180], [5000, 2000, -115 * np.pi / 180],
    #          [2780, 5000, -55 * np.pi / 180], [400, 4400, 85 * np.pi / 180], [2040, 300, 65 * np.pi / 180]],
    #         [[3000, 0, -135 * np.pi / 180] for _ in range(12)],
    #         [], [], [], []]
    for j in range(monte_carlo):
        broadcast = [mp.Queue() for _ in range(len(uavs[0])+1)]
        multi_process = [mp.Process(target=DGA, args=(targets_sites, uavs, broadcast, a, generation))
                         for a in range(len(uavs[0]))]
        # s = time.time()
        for p in multi_process:
            p.start()
        # for p in multi_process:
        #     p.join()
        # print(time.time() - s)
        f = [[[], []] for _ in range(len(uavs[0]))]
        for k in f:
            if not k[0]:
                solution = broadcast[-1].get()
                k[0].extend(solution[0])
                k[1].extend(solution[1])
        print(len(f))
        print(f)
        curve = [min([a[0][g] for a in f]) for g in range(generation)]
        convergence[2] = list(np.add(convergence[2], curve))
        total_fit = [a[0][-1] for a in f]
        object_value[2].append(min(total_fit))
        best_chromosome = f[total_fit.index(min(total_fit))][1] if min(total_fit) < history_cost else best_chromosome
        history_cost = min(total_fit) if min(total_fit) < history_cost else history_cost

    print('cga')
    cga = GA_SEAD(targets_sites, 300)
    for j in range(monte_carlo):
        solution, fitness_value, ga_population, b = cga.run_GA(generation, uavs)
        convergence[1] = list(np.add(convergence[1], b))
        object_value[1].append(1/fitness_value)
        # print(f'CGA: {1/fitness_value}')

    # print('pso')
    # pso = PSO_SEAD(targets_sites)
    # for j in range(monte_carlo):
    #     gbest, c = pso.run_PSO(generation, uavs)
    #     convergence[1] = list(np.add(convergence[1], c))
    #     object_value[1].append(1/gbest[0])
    #     # print(f'PSO: {1/fitness_value}')

    print('rs')
    rs = RS_SEAD(targets_sites)
    for j in range(monte_carlo):
        solution, fitness_value, ga_population, d = rs.run_RS(generation, uavs)
        convergence[0] = list(np.add(convergence[0], d))
        object_value[0].append(1/fitness_value)
        # print(f'RS: {1/fitness_value}')

    print('finish')
    plt.figure(1)
    legend = ['RS', 'GA', 'DGA']
    for k in range(3):
        convergence[k] = list(np.divide(convergence[k], monte_carlo))
        convergence[k] = list(np.divide(convergence[k], convergence[k][0]))
        plt.plot([g for g in range(1, generation+1)], convergence[k], '-', label=legend[k])
    plt.xlabel('i generation', fontsize=12)
    plt.ylabel('E ( $\mathregular{J_i}$ / $\mathregular{J_1}$ )', fontsize=12)
    plt.legend()
    plt.show()
    print('convergence curve')
    print(convergence)

    plt.figure(2)
    plt.boxplot(object_value, labels=legend)
    plt.show()
    print('objective list')
    print(object_value)

    plt.figure(3)
    cga.plot_result(best_chromosome)
    print(f'optimal objective: {history_cost}')
