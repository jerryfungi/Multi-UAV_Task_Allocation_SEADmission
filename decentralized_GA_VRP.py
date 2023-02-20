import random
import time
import math
import numpy as np
import threading
import queue
from dubins_model import *
from matplotlib import pyplot as plt
import multiprocessing as mp
import copy


class UAV(object):
    def __init__(self, uav_id, uav_site, uav_specification):
        self.uav_id = uav_id
        self.x0 = uav_site[0]
        self.y0 = uav_site[1]
        self.theta0 = uav_site[2]
        self.velocity = uav_specification[0]
        self.Rmin = uav_specification[1]
        self.omega_max = self.velocity / self.Rmin


# thread 1
def GA_thread(uav, targets_sites, ga2control_queue, control2ga_queue):
    # configuration
    uav_site = [uav.x0, uav.y0]
    targets_sites = targets_sites
    targets_num = len(targets_sites)
    # initial mapping

    def distance_cost(node1, node2):
        return np.sqrt((node2[0] - node1[0]) ** 2 + (node2[1] - node1[1]) ** 2)

    cost_matrix = [
        (lambda x: [distance_cost(targets_sites[x], targets_sites[y]) for y in range(targets_num)])(z)
        for z in range(targets_num)]
    uav2targets = [[distance_cost(targets_sites[_], uav_site) for _ in range(targets_num)]]
    targets2origin = [[distance_cost(targets_sites[_], uav_site) for _ in range(targets_num)]]
    # other uav information
    all_agents_id = [uav.uav_id]
    all_agents_information = [[uav.velocity, uav.Rmin, uav_site[0], uav_site[1]]]
    # communication
    ga2control_port = ga2control_queue
    control2ga_port = control2ga_queue
    # GA parameters
    pop_size = 100
    crossover_prob = 1
    mutation_prob = 0.7
    crossover_num = 66
    mutation_num = 32
    elitism_num = 4
    task_status_list = [1 for _ in range(targets_num)]

    def fitness_evaluate(pop):
        def fitness_function(total_cost):
            return 1 / (max(total_cost) + 0.01 * sum(total_cost))

        fitness_value = []
        for index, chromosome in enumerate(pop):
            cost, pre_location = [0 for _ in range(len(all_agents_id))], [0 for _ in range(len(all_agents_id))]
            for n in range(len(chromosome[0])):
                a = all_agents_id.index(chromosome[1][n])
                target = chromosome[0][n]
                if not pre_location[a] == 0:
                    cost[a] += cost_matrix[pre_location[a]-1][target-1] / all_agents_information[a][0]
                else:
                    cost[a] += uav2targets[a][target-1] / all_agents_information[a][0]
                pre_location[a] = target
            cost = np.add(cost, [targets2origin[m][pre_location[m]-1] / all_agents_information[m][0]
                                 for m in range(len(cost))])
            fitness_value.append(fitness_function(cost))
        return fitness_value

    def generate_population():
        def generate_chromosome():
            chromosome = [[i + 1 for i in range(targets_num)],
                          [random.choice(all_agents_id) for _ in range(targets_num)]]
            random.shuffle(chromosome[0])
            return chromosome

        return [generate_chromosome() for _ in range(pop_size)]

    def roulette_wheel(fit):
        return np.array(fit) / np.sum(fit)

    def selection(roulette__wheel, num):
        return np.random.choice(np.arange(len(roulette__wheel)), size=num, replace=False, p=roulette__wheel)

    def crossover(parent_1, parent_2):
        targets_number = sum(task_status_list)
        child_1, child_2 = [[], []], [[], []]
        if random.random() <= crossover_prob:
            try:
                cutpoint = random.sample(range(targets_number), 2)
                cutpoint.sort()
                remain_1, remain_2 = [[], []], [[], []]
                # order crossover
                for i in range(targets_number):
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
                return [child_1, child_2]
            except:
                return [parent_1, parent_2]
        else:
            return [parent_1, parent_2]

    def mutation(chromosome):
        gene = [[], []]
        if random.random() <= mutation_prob:
            if random.random() <= 0.5 and not len(all_agents_id) == 1:  # mutate agent
                try:
                    mutpoint = random.randint(0, sum(task_status_list) - 1)
                    assign = random.choice([i for i in all_agents_id if i not in [chromosome[1][mutpoint]]])
                    gene[0].extend(chromosome[0][:])
                    gene[1].extend(chromosome[1][:mutpoint] + [assign] + chromosome[1][mutpoint + 1:])
                    return gene
                except:
                    return chromosome
            else:  # inverse mutation
                try:
                    cutpoint = random.sample(range(sum(task_status_list)), 2)
                    cutpoint.sort()
                    for i, row in enumerate(chromosome):
                        inverse_gene = row[cutpoint[0]:cutpoint[1]]
                        inverse_gene.reverse()
                        gene[i].extend(row[:cutpoint[0]] + inverse_gene + row[cutpoint[1]:])
                    return gene
                except:
                    return chromosome
        else:
            return chromosome

    def elitism(pop, fit):
        ranking = sorted(range(len(pop)), key=lambda f: fit[f], reverse=True)[:elitism_num]
        return [pop[_] for _ in ranking]

    def substitute(pop, fit, new_pop, new_fit):
        for i in range(len(new_pop)):
            if new_fit[i] > min(fit):
                pop[np.argmin(fit)] = new_pop[i]
                fit[np.argmin(fit)] = new_fit[i]

    def uav_information_adjust(msg, pop):
        new_agent_set, task_executed_set = [], []
        task_complete = False
        # ga.all_agents_id = [ga.uav.uav_id]
        # ga.all_agents_information = [[ga.uav.velocity, ga.uav.Rmin, ga.uav_site[0], ga.uav_site[1]]]
        for packet in msg:
            if packet[0] not in all_agents_id:  # find new agent
                all_agents_id.append(packet[0])
                all_agents_information.append(packet[1:5])
                uav2targets.append([])
                new_agent_set.append(packet[0])
                targets2origin.append([distance_cost(targets_sites[_], uav_site) for _ in range(targets_num)])
            else:  # update current location
                uav_index = all_agents_id.index(packet[0])
                all_agents_information[uav_index][2:] = packet[3:5]
            if not packet[-1] == task_status_list:
                task_complete = True
        for agent in range(len(uav2targets)):  # adjust uav to targets cost
            uav2targets[agent] = [distance_cost(targets_sites[_], all_agents_information[agent][2:])
                                  for _ in range(targets_num)]
        # update complete task
        for packet in msg:
            for i, task in enumerate(packet[-1]):
                if task == 0:
                    task_status_list[i] = 0
        if task_complete:
            for chromosome in pop:
                for i, gene in enumerate(chromosome[0]):
                    if task_status_list[gene-1] == 0:
                        del chromosome[0][i], chromosome[1][i]
            for packet in msg:
                for i, gene in enumerate(packet[5][0]):
                    if task_status_list[gene-1] == 0:
                        del packet[5][0][i], packet[5][1][i]
                pop.append(packet[5])
        else:
            for packet in msg:
                pop.append(packet[5])
        if not new_agent_set == []:
            for chromosome in pop:
                chromosome[1] = [random.choice(all_agents_id) for _ in range(len(chromosome[0]))]

    def ga_chromosome2task_execution(pop, fit):
        ga2control_port.put(pop[fit.index(max(fit))])

    def others_information2ga_thread(pop, fit):
        try:
            subpop = control2ga_port.get()
            uav_information_adjust(subpop, pop)
            fit = fitness_evaluate(pop)
        except queue.Empty:
            pass
        return pop, fit

    # initial setting
    broadcast_interval = 1
    previous_time = time.time()
    y = 0
    # start algorithm
    population = generate_population()
    fitness = fitness_evaluate(population)
    while True:
        y += 1
        wheel = roulette_wheel(fitness)
        new_population = []
        new_population.extend(elitism(population, fitness))
        for cross in range(0, pop_size-len(new_population), 2):
            parents = selection(wheel, 2)
            offspring = crossover(population[parents[0]], population[parents[1]])
            new_population.extend([mutation(offspring[0])])
            new_population.extend([mutation(offspring[1])])
        new_fitness = fitness_evaluate(new_population)
        population = new_population
        # if max(new_fitness) > max(fitness):
        #     print(f'{uav.uav_id}__cost:{1 / max(new_fitness)}')
        fitness = new_fitness
        if (time.time() - previous_time) >= broadcast_interval:
            # uav = [[]for _ in range(len(all_agents_id))]
            # g = population[fitness.index(max(fitness))]
            # for i in range(len(g[0])):
            #     uav[g[1][i]-1].append(g[0][i]-1)
            # x = [[all_agents_information[_][2]]for _ in range(len(all_agents_id))]
            # y = [[all_agents_information[_][3]]for _ in range(len(all_agents_id))]
            # for w, uu in enumerate(uav):
            #     for t in uu:
            #         x[w].append(targets_sites[t][0])
            #         y[w].append(targets_sites[t][1])
            #     plt.plot(x[w], y[w], '-')
            #     plt.plot()
            # for task in targets_sites:
            #     plt.plot(task[0], task[1], 'ro')
            # plt.show()

            ga_chromosome2task_execution(population, fitness)
            population, fitness = others_information2ga_thread(population, fitness)
            previous_time = time.time()
            print(f'y={y}')
            y = 0


def TaskSequenceExecution(uav, targets_sites, communication, ga2control_queue, control2ga_queue, u2g):
    # configuration
    uav = uav
    uav_site = [uav.x0, uav.y0]
    targets_sites = targets_sites
    targets_sites.append([uav.x0, uav.y0])
    # communication queue (broadcast and receive port between agents)
    communication = communication
    # communication between ga thread and task execution thread
    ga2control_port = ga2control_queue
    control2ga_port = control2ga_queue
    # u2g for results
    gcs = u2g
    task_status_list = [1 for _ in range(len(targets_sites)-1)]

    def decode_chromosome(chromosome):
        task_sequence = []
        for i, gene in enumerate(chromosome[1]):
            if gene == uav.uav_id:
                task_sequence.append(targets_sites[chromosome[0][i] - 1])
        task_sequence.append([uav.x0, uav.y0])
        return task_sequence

    def pack_chromosome(chromosome):
        # print(task_status_list)
        return [uav.uav_id, uav.velocity, uav.Rmin, uav_site[0], uav_site[1],
                chromosome, task_status_list]

    # build uav
    u = 0
    t = 0
    xn = uav.x0
    yn = uav.y0
    theta = uav.theta0
    list_for_u = [0]
    list_for_t = [0]
    task_allocation = []
    c = 0.1
    actual_x = [uav.x0]
    actual_y = [uav.y0]
    armed = False
    disarmed = False
    complete = False
    waypoint_radius = 50
    # communication setting
    broadcast_list = [i for i in range(len(communication)) if not i+1 == uav.uav_id]
    plot_time = 0
    # print(f'{uav.uav_id} broadcast_list: {broadcast_list}')
    previous_time = 0
    while True:
        # ---------------------- communication part ------------------------- #
        try:
            if not ga2control_port.empty():
                # get current best solution (from ga thread)-----------
                current_best_chromosome = ga2control_port.get(timeout=1)
                # print(f'uav{uav.uav_id}::best_solution:{ga_solution}')
                # if got current best solution from ga thread----------
                task_allocation = decode_chromosome(current_best_chromosome)
                print(f'UAV_{uav.uav_id}  current_best: {current_best_chromosome}')
                # print(f'uav{uav.uav_id}_tasks:{task_allocation}')
                broadcast_packet = pack_chromosome(current_best_chromosome)
                # broadcast packet-------------------------------------
                for q in broadcast_list:
                    communication[q].put(broadcast_packet)
                # get other uav information (xbee)---------------------
                receive_from_v2v = []
                while not len(receive_from_v2v) == len(communication)-1:
                    try:
                        v2v_message = communication[uav.uav_id - 1].get(timeout=1)
                        receive_from_v2v.append(v2v_message)
                        # update terminal task--------------------------
                        for task, check in enumerate(v2v_message[-1]):
                            if check == 0:
                                task_status_list[task] = 0
                    except queue.Empty:
                        pass
                # give it to ga thread-----------------------------------
                control2ga_port.put(receive_from_v2v + [broadcast_packet])
        except queue.Empty:
            pass
        # ---------------------- control part ---------------------------- #
        if not task_allocation == [] and not armed:  # first command
            armed = True
            print(f'UAV_{uav.uav_id} arm !!')
            previous_time = time.time()  # mission start time

        if not armed or complete:
            continue
        else:
            # plot animation
            if time.time() - plot_time >= 0.5:
                gcs.put([uav.uav_id, xn, yn])
                plot_time = time.time()
            # print(task_allocation)
            assign = 0
            while distance_between_points([xn, yn], task_allocation[assign]) <= waypoint_radius:
                if task_allocation[assign] == [uav.x0, uav.y0]:
                    actual_x.append(task_allocation[assign][0])
                    actual_y.append(task_allocation[assign][1])
                    disarmed = True
                    task_allocation.clear()
                    break
                task_status_list[targets_sites.index(task_allocation[0])] = 0
                # print(f'target: {task_allocation[0]} completed by uav{uav.uav_id}')
                assign += 1
            if disarmed:
                # gcs.put([uav.uav_id, actual_x, actual_y, list_for_t])
                print(f'UAV_{uav.uav_id} mission complete !!!!!!!!!!!!!!')
                armed = False
                complete = True
            elif distance_between_points([xn, yn], task_allocation[assign]) > waypoint_radius:
                angle_between_two_points = angle_between((xn, yn), task_allocation[assign][:2])
                dt = (time.time() - previous_time)
                previous_time = time.time()
                xn, yn, thetan = step(uav, xn, yn, theta, u, dt)
                relative_angle = angle_between_two_points - thetan
                error_of_heading = relative_angle if abs(relative_angle) <= 2 * pi - abs(relative_angle) \
                    else -(relative_angle / abs(relative_angle)) * (2 * pi - abs(relative_angle))
                if c >= error_of_heading >= -c:
                    u = 0
                    theta = thetan
                elif error_of_heading < -c:
                    u = -1
                    theta = thetan

                elif error_of_heading > c:
                    u = 1
                    theta = thetan
                else:
                    u = 1
                    theta = thetan

                actual_x.append(xn)
                actual_y.append(yn)
                t += dt
                list_for_u.append(u)
                list_for_t.append(t)

                uav_site = [xn, yn]  # update current position


if __name__ == '__main__':
    uav_position = [[1500, 0, np.pi/2], [-1500, 0, np.pi/2], [0, 0, np.pi/2]]
    # uav_position = [[0, 0, np.pi / 2]]
    uav_configuration = [[25, 75], [15, 50], [35, 100]]
    # uav_configuration = [[1, 2]]
    uav_num = len(uav_configuration)
    targets = [[-65,15],[27,66],[-51,58],[77,77],[0,39],
                     [71,95],[-25,10],[0,91],[30,30],[-100,24],[38,75]]
    # targets = [[random.randint(-100, 100), random.randint(20, 100)] for _ in range(20)]
    targets = [list(np.array(task)*20) for task in targets]
    broadcast = [mp.Queue() for _ in range(uav_num)]
    control2ga_thread = [mp.Queue() for _ in range(uav_num)]
    ga2control_thread = [mp.Queue() for _ in range(uav_num)]
    GCS = mp.Queue()
    # build uav
    UAVs = [UAV(a + 1, uav_position[a], uav_configuration[a]) for a in range(uav_num)]
    # GA
    GA_threads = [mp.Process(target=GA_thread, args=(UAVs[a], targets, ga2control_thread[a], control2ga_thread[a]))
                  for a in range(uav_num)]
    # control and communication
    TaskSequenceExecution_threads = [mp.Process(target=TaskSequenceExecution,
                                                args=(UAVs[a], targets, broadcast,
                                                      ga2control_thread[a], control2ga_thread[a], GCS))
                                     for a in range(uav_num)]
    for a in range(uav_num):
        GA_threads[a].start()
        TaskSequenceExecution_threads[a].start()
    for t in targets:
        plt.plot(t[0], t[1], 'ko', markerfacecolor='none', markersize=8)
    for u in uav_position:
        plt.plot(u[0], u[1], 'r^', markerfacecolor='none', markersize=8)
    color_style = ['tab:blue', 'tab:green', 'tab:orange']
    while True:
        try:
            surveillance = GCS.get(timeout=1e-5)
            # print(surveillance)
            plt.plot(surveillance[1], surveillance[2], 'o', color=color_style[surveillance[0]-1], markersize=1)
            plt.pause(1e-10)
        except queue.Empty:
            pass
    # plot results
    # for data in result:
    #     plt.plot(data[1], data[2], '-')
    # for t in targets:
    #     plt.plot(t[0], t[1], 'bo')
    # for u in uav_position:
    #     plt.plot(u[0], u[1], 'ro')
    # plt.show()
