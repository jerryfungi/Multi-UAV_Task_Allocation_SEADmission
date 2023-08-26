import multiprocessing as mp
from GA_SEAD_process import *
from PSO_SEAD import *

' Parameters '
monte_carlo_runs = 5
generation = 300
population_size = 300

def DGA(targets, uav_msg, queue, index, iteration, exchange_interval=100):
    share_list = [i for i in range(len(uav_msg[0])) if not i == index]
    dga = GA_SEAD(targets, population_size)
    population, ans = None, None
    fitness_list = []
    for i in range(int(iteration/exchange_interval)):
        ans, fitness, population, a = dga.run_GA(exchange_interval, uav_msg, population, distributed=True)
        fitness_list.extend(a)
        for port in share_list:
            queue[port].put(ans)
        subpop = []
        while not len(subpop) == len(uav_msg[0])-1:
            subpop.append(queue[index].get())
        population.extend(subpop)
    _, ti, di, pe = dga.chromosome_objectives_evaluate(ans)
    queue[-1].put([fitness_list, ans, [ti, di, pe]])


if __name__ == "__main__":
    history_cost = 1e5
    best_chromosome = []
    convergence = [[0 for g in range(generation)] for _ in range(3)]
    object_value = [[] for _ in range(3)]
    mission_time = [[] for _ in range(3)]
    total_distance = [[] for _ in range(3)]
    penalty = [[] for _ in range(3)]
    process_time = [[] for _ in range(3)]

    ' Simulation conditions '
    targets = [[3100, 2200], [500, 3700], [2300, 2500], [2000, 3900], [4450, 3600], [4630, 4780], [1400, 4500],
               [3300, 3415], [1640, 1700], [4230, 1700], [500, 2200], [3000, 4500], [5000, 2810]]
    uavs_info = [[i for i in range(1, 12)], [1, 2, 3, 1, 3, 2, 1, 2, 3, 1, 2],
                 [70, 80, 90, 60, 100, 80, 75, 90, 85, 70, 65],
                 [200, 250, 300, 180, 300, 260, 225, 295, 250, 200, 170],
                 [[1000, 300, -np.pi], [1500, 700, np.pi / 2], [3000, 0, np.pi / 3], [1800, 400, -20 * np.pi / 180],
                  [2200, 280, 45 * np.pi / 180], [4740, 300, 140 * np.pi / 180], [4000, 100, 70 * np.pi / 180],
                  [3500, 450, -75 * np.pi / 180], [5000, 900, -115 * np.pi / 180], [2780, 500, -55 * np.pi / 180],
                  [4000, 600, 85 * np.pi / 180]],
                 [[0, 0, -np.pi / 2] for _ in range(11)]]
    conditions = {"small scale": [targets[:4],
                                  [[row[j] for j in range(3)] for row in uavs_info] + [[], [], [], []]],
                  "median scale": [targets[:7],
                                   [[row[j] for j in range(6)] for row in uavs_info] + [[], [], [[4, 1], [5, 1]], []]],
                  "large scale": [targets[:],
                                  uavs_info + [[], [], [[4, 1], [5, 1], [9, 1], [9, 2], [8, 1], [8, 2], [13, 1]], []]]}

    targets_sites = conditions["small scale"][0]
    uavs = conditions["small scale"][1]

    print('Start simulations')
    print('Decentralized parallel genetic algorithm operating ------')
    for j in range(monte_carlo_runs):
        broadcast = [mp.Queue() for _ in range(len(uavs[0])+1)]
        multi_process = [mp.Process(target=DGA, args=(targets_sites, uavs, broadcast, a, generation))
                         for a in range(len(uavs[0]))]
        start = time.time()
        for p in multi_process:
            p.start()
        f = [[[], [], []] for _ in range(len(uavs[0]))]
        for k in f:
            if not k[0]:
                solution = broadcast[-1].get()
                k[0].extend(solution[0])
                k[1].extend(solution[1])
                k[2].extend(solution[2])
        process_time[2].append(time.time()-start)
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

    print('Genetic algorithm operating ------')
    cga = GA_SEAD(targets_sites, population_size)
    for j in range(monte_carlo_runs):
        start = time.time()
        solution, fitness_value, ga_population, b = cga.run_GA(generation, uavs)
        process_time[1].append(time.time() - start)
        convergence[1] = list(np.add(convergence[1], b))
        object_value[1].append(1/fitness_value)
        _, t, d, p = cga.chromosome_objectives_evaluate(solution)
        mission_time[1].append(t)
        total_distance[1].append(d)
        penalty[1].append(p)

    print('Random search method operating ------')
    for j in range(monte_carlo_runs):
        start = time.time()
        solution, fitness_value, ga_population, d = cga.run_RS(generation, uavs)
        process_time[0].append(time.time() - start)
        convergence[0] = list(np.add(convergence[0], d))
        object_value[0].append(1/fitness_value)
        _, t, d, p = cga.chromosome_objectives_evaluate(solution)
        mission_time[0].append(t)
        total_distance[0].append(d)
        penalty[0].append(p)

    print(f"\nResults of {monte_carlo_runs} Monte Carlo simulations: \n"
          f"convergence = {convergence} \n"
          f"object_value = {object_value} \n"
          f"mission_time = {mission_time} \n"
          f"total_distance = {total_distance} \n"
          f"penalty = {penalty} \n"
          f"process_time = {process_time}"
          f"-------------------------------------------")

    print(f"Process time: \n"
          f"RS: {np.average(process_time[0])} \n"
          f"GA: {np.average(process_time[1])} \n"
          f"DPGA: {np.average(process_time[2])} \n"
          f"-------------------------------------------")

    print(f"Times violating the task precedence constraints: \n"
          f"RS: {np.count_nonzero(penalty[0])} \n"
          f"GA: {np.count_nonzero(penalty[1])} \n"
          f"DPGA: {np.count_nonzero(penalty[2])} \n"
          f"-------------------------------------------")

    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
    fig, ax = plt.subplots()
    legend = ['RS', 'GA', 'DPGA']

    for k in range(3):
        convergence[k] = list(np.divide(convergence[k], monte_carlo_runs))
        convergence[k] = list(np.divide(convergence[k], convergence[k][0]))
        convergence[k] = [v ** (1 / 2) for v in convergence[k]]
        plt.plot([g for g in range(1, generation + 1)], convergence[k], '-', label=legend[k], linewidth=0.8)
    plt.legend(prop=font2)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.xlabel('i generation', font)
    plt.ylabel('E ( $\mathregular{J_i}$ / $\mathregular{J_1}$ )', font)
    plt.show()

    total_distance = [[a / np.sum(uavs[2]) for a in total_distance[b]] for b in range(len(total_distance))]
    fig, ax = plt.subplots(1, 3)
    colors = ['pink', 'lightblue', 'lightgreen']
    for axes in ax:
        labels = axes.get_xticklabels() + axes.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
    box1 = ax[0].boxplot(object_value, patch_artist=True, showmeans=True,
                         boxprops={"facecolor": colors[0],
                                   "edgecolor": "k",
                                   "linewidth": 0.5},
                         medianprops={"color": "k", "linewidth": 0.5},
                         meanprops={'marker': '+',
                                    'markerfacecolor': 'k',
                                    'markeredgecolor': 'k',
                                    'markersize': 5}, labels=legend, whiskerprops={"color": "k", "linewidth": 0.7},
                         showfliers=False)
    ax[0].grid(axis='y', ls='--', alpha=0.8)
    ax[0].set_title('Objectives', font)
    box2 = ax[1].boxplot(mission_time, patch_artist=True, showmeans=True,
                         boxprops={"facecolor": colors[1],
                                   "edgecolor": "k",
                                   "linewidth": 0.5},
                         medianprops={"color": "k", "linewidth": 0.5},
                         meanprops={'marker': '+',
                                    'markerfacecolor': 'k',
                                    'markeredgecolor': 'k',
                                    'markersize': 5}, labels=legend,
                         whiskerprops={"color": "k", "linewidth": 0.7},
                         showfliers=False)
    ax[1].grid(axis='y', ls='--', alpha=0.8)
    ax[1].set_title('Mission time', font)
    box3 = ax[2].boxplot(total_distance, patch_artist=True, showmeans=True,
                         boxprops={"facecolor": colors[2],
                                   "edgecolor": "k",
                                   "linewidth": 0.5},
                         medianprops={"color": "k", "linewidth": 0.5},
                         meanprops={'marker': '+',
                                    'markerfacecolor': 'k',
                                    'markeredgecolor': 'k',
                                    'markersize': 5}, labels=legend, whiskerprops={"color": "k", "linewidth": 0.7},
                         showfliers=False)
    ax[2].grid(axis='y', ls='--', alpha=0.8)
    ax[2].set_title('Total distance', font)

    plt.tight_layout()
    plt.show()

    cga.plot_result(best_chromosome)
