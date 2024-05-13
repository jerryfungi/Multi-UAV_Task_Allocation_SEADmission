import random
import time
import math
import numpy as np
import copy
from matplotlib import pyplot as plt
from dubins__ import *
import dubins


class GA_task_allocation(object):

    def __init__(self, target_sites, uav_sites, terminal_sites, mission_amount, uav_specification):
        # input data
        self.target_sites = target_sites
        self.uav_sites = uav_sites
        self.terminal_sites = terminal_sites
        self.target_num = len(target_sites)
        self.uav_num = len(uav_sites)
        self.mission_num = mission_amount
        # GA parameters
        self.population_size = 100
        self.crossover_num = 66
        self.mutation_num = 32
        self.elitism_num = 2
        self.iteration = 100
        self.velocity = uav_specification[0]
        self.Rmax = uav_specification[1]
        self.discrete_heading = [_ for _ in range(0, 360, 10)]
        self.uavType_for_missions = [[] for _ in range(3)]
        self.error_time = 1
        self.Nm = self.target_num * self.mission_num / 2
        # [surveillance[1,3],attack[1,2,3],ammunition[2]]
        # [surveillance[s,att],attack[att,a],verification[s,att]]
        for i, agent in enumerate(uav_specification[2]):
            if agent == 1:  # surveillance
                self.uavType_for_missions[0].append(i + 1)
                self.uavType_for_missions[2].append(i + 1)
            elif agent == 2:  # attack
                self.uavType_for_missions[0].append(i + 1)
                self.uavType_for_missions[1].append(i + 1)
                self.uavType_for_missions[2].append(i + 1)
            elif agent == 3:  # ammunition
                self.uavType_for_missions[1].append(i + 1)
        self.local_planner = [Dubins(r, .5) for r in self.Rmax]

    def dubins_(self, state_list, time_list, uav_index):
        distance = 0
        for state in state_list[1:-1]:  # degree to rad
            state[2] *= np.pi / 180
        for i in range(len(state_list) - 1):
            sp = state_list[i]
            gp = state_list[i + 1] if state_list[i] != state_list[i + 1] \
                else [state_list[i + 1][0], state_list[i + 1][1], abs(state_list[i + 1][2] - 1e-3)]
            dubins_path_length = dubins.shortest_path(sp, gp, self.Rmax[uav_index]).path_length()
            distance += dubins_path_length
            try:
                time_list[i].append(distance / self.velocity[uav_index])
            except:
                pass
        return distance / self.velocity[uav_index]

    def fitness(self, population):
        fitness_value = []
        for i in range(self.population_size):
            cost = np.zeros(self.uav_num)
            task_sequence_state = [[] for _ in range(self.uav_num)]  # target site
            task_sequence_time = [[] for _ in range(self.uav_num)]  # time
            time_list = []
            for j in range(self.target_num * self.mission_num):
                task_sequence_state[population[i][3][j] - 1].append([
                    self.target_sites[population[i][1][j] - 1][0], self.target_sites[population[i][1][j] - 1][1],
                    population[i][4][j]])
                task_sequence_time[population[i][3][j] - 1].append([population[i][1][j], population[i][2][j]])
            for j in range(self.uav_num):
                task_sequence_state[j] = [self.uav_sites[j]] + task_sequence_state[j][:] + [self.terminal_sites[j]]
                cost[j] = self.dubins_(task_sequence_state[j], task_sequence_time[j], j)
            for sequence in task_sequence_time:
                time_list.extend(sequence)
            time_list.sort()
            # time sequence penalty
            penalty = 0
            for j in range(0, len(time_list), self.mission_num):
                for k in range(1, self.mission_num):
                    penalty += max(0, time_list[j + k - 1][2] - time_list[j + k][2] + self.error_time)
            fitness_value.extend([1 / (np.max(cost) + 0.01 * np.sum(cost) + 100 * penalty)])
        roulette_wheel = np.array(fitness_value) / np.sum(fitness_value)
        return fitness_value, list(roulette_wheel)

    def initiate_population(self):
        def generate__chromosome():
            chromosome = np.zeros((5, self.target_num * self.mission_num), dtype=int)

            for i in range(chromosome.shape[1]):
                chromosome[0][i] = i + 1  # order
                chromosome[1][i] = random.choice([i for i in range(1, self.target_num + 1)  # target id
                                                  if np.count_nonzero(chromosome[1] == i) < self.mission_num])
            # turn to target-based
            zipped_gene = [list(g) for g in zip(chromosome[0], chromosome[1], chromosome[2],
                                                chromosome[3], chromosome[4])]
            target_based_gene = np.array(sorted(zipped_gene, key=lambda u: u[1]))
            for i in range(target_based_gene.shape[0]):
                target_based_gene[i][2] = (i % self.mission_num) + 1  # mission type
                target_based_gene[i][3] = random.choice(
                    self.uavType_for_missions[target_based_gene[i][2] - 1])  # uav id
                target_based_gene[i][4] = random.choice(self.discrete_heading)  # heading angle
            # back to order-based
            chromosome = [[] for _ in range(5)]
            order_based_gene = np.array(sorted(target_based_gene, key=lambda u: u[0]))
            for i in range(5):
                chromosome[i] = [g[i] for g in order_based_gene]
            return chromosome

        return [generate__chromosome() for _ in range(self.population_size)]

    def selection(self, roulette_wheel, num):
        return np.random.choice(np.arange(len(roulette_wheel)), size=num, replace=False, p=roulette_wheel)

    def crossover(self, parent_1, parent_2):
        # turn to target-based
        target_based_gene, order_based_gene = [], []
        for parents in [parent_1, parent_2]:
            zipped_gene = [list(g) for g in zip(parents[0], parents[1], parents[2], parents[3], parents[4])]
            target_based_gene.append(sorted(zipped_gene, key=lambda u: u[1]))
        # choose cut point
        cutpoint = random.sample(range(self.target_num), 2)
        cutpoint_1, cutpoint_2 = min(cutpoint), max(cutpoint)
        child_1, child_2 = [], []
        # 2 point crossover
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
        return child_1, child_2

    def mutation(self, chromosome):
        def point_mutation():
            # choose mutate point
            mutpoint = random.randint(0, len(chromosome[0]) - 1)
            # mutate assign uav or heading angle
            new_gene = [[] for _ in range(5)]
            for i in range(len(chromosome)):  # copy chromosome
                new_gene[i] = chromosome[i][:]
            if random.random() < 0.5:
                new_gene[3][mutpoint] = random.choice(
                    [i for i in self.uavType_for_missions[new_gene[2][mutpoint] - 1] if
                     i != chromosome[3][mutpoint]])
            else:
                new_gene[4][mutpoint] = random.choice([i for i in self.discrete_heading if
                                                       i != chromosome[4][mutpoint]])
            return new_gene

        def multiple_point_mutation():
            # choose mutate point
            mutpoint = np.random.choice(len(chromosome[0]), np.random.randint(1, self.Nm + 1), replace=False)
            # mutate assign uav or heading angle
            new_gene = [[] for _ in range(5)]
            for i in range(len(chromosome)):  # copy chromosome
                new_gene[i] = chromosome[i][:]
            if random.random() < 0.5:
                for point in mutpoint:
                    new_gene[3][point] = random.choice([i for i in self.uavType_for_missions[new_gene[2][point] - 1] if
                                                        i != chromosome[3][point]])
            else:
                for point in mutpoint:
                    new_gene[4][point] = random.choice([i for i in self.discrete_heading if
                                                        i != chromosome[4][point]])
            return new_gene

        def target_state_mutation():
            # turn to target-based
            zipped_gene = [list(g) for g in zip(chromosome[0], chromosome[1], chromosome[2],
                                                chromosome[3], chromosome[4])]
            target_based_gene = (sorted(zipped_gene, key=lambda u: u[1]))
            # shuffle the state
            target_sequence = list(range(self.target_num))
            random.shuffle(target_sequence)
            mutate_target_based = [[] for _ in range(self.target_num * self.mission_num)]
            for n in range(self.target_num):
                mutate_target_based[self.mission_num * n:self.mission_num * (n + 1)] = \
                    [[b[:1] for b in target_based_gene[self.mission_num * n:self.mission_num * (n + 1)]][i] +
                     [a[1:] for a in target_based_gene[self.mission_num * target_sequence[n]:
                                                       self.mission_num * (target_sequence[n] + 1)]][i]
                     for i in range(self.mission_num)]
            # back to order-based
            new_gene = [[] for _ in range(5)]
            order_based_gene = (sorted(mutate_target_based, key=lambda u: u[0]))
            for i in range(5):
                new_gene[i] = [g[i] for g in order_based_gene]
            return new_gene

        def task_state_mutation():
            # turn to target-based
            zipped_gene = [list(g) for g in zip(chromosome[0], chromosome[1], chromosome[2],
                                                chromosome[3], chromosome[4])]
            task_based_gene = (sorted(zipped_gene, key=lambda u: u[2]))
            # shuffle the state
            task_sequence = list(range(self.target_num))
            random.shuffle(task_sequence)
            # choose mutate task
            muttask = random.randint(0, self.mission_num-1)
            # task mutate
            mutate_task_based = [[] for _ in range(len(task_based_gene))]
            for i in range(len(task_based_gene)):  # copy chromosome
                mutate_task_based[i] = task_based_gene[i][:]
            for i, sequence in enumerate(task_sequence):
                mutate_task_based[muttask*self.target_num+sequence][3:] = \
                    task_based_gene[muttask*self.target_num+i][3:]
            # back to order-based
            new_gene = [[] for _ in range(5)]
            order_based_gene = (sorted(mutate_task_based, key=lambda u: u[0]))
            for i in range(5):
                new_gene[i] = [g[i] for g in order_based_gene]
            return new_gene
        random_choose = random.choice([1, 2, 3])
        if random_choose == 1:
            mut_gene = multiple_point_mutation()
        elif random_choose == 2:
            mut_gene = target_state_mutation()
        else:
            mut_gene = task_state_mutation()
        return mut_gene

    def elitism(self, fitness_value):
        fitness_ranking = sorted(range(len(fitness_value)), key=lambda u: fitness_value[u], reverse=True)
        elitism_id = fitness_ranking[:self.elitism_num]
        return elitism_id

    def adaptive_setting(self, Nit):
        Ncr = round((self.population_size - self.elitism_num) * math.exp(-Nit / self.iteration))
        Nmu = self.population_size - self.elitism_num - Ncr
        return Ncr, Nmu

    def GA_SEAD(self):
        start = time.time()
        fitness_curve = []
        population = self.initiate_population()
        fitness, wheel = self.fitness(population)
        fitness_curve.append(1 / max(fitness))
        for i in range(self.iteration):
            print(i)
            new_population = []
            # crossover_num, mutation_num = self.adaptive_setting(i+1)
            # print(time.time())
            elitism_gene = self.elitism(fitness)
            new_population.extend([population[k] for k in elitism_gene])
            for j in range(0, self.crossover_num, 2):
                parent_1, parent_2 = self.selection(wheel, 2)
                children = self.crossover(population[parent_1], population[parent_2])
                new_population.extend([children[0], children[1]])
                # if crossover_num % 2 == 1:
                #     parent_1, parent_2 = [self.selection(wheel) for k in range(2)]
                #     child_1, child_2 = self.crossover(population[parent_1], population[parent_2])
                #     new_population.extend([child_1])
                # for child in children:
                #     if random.random() <= 0.5:
                #         new_population.append(self.mutation(child))
                #     else:
                #         new_population.extend([child])
            # print(time.time())
            for j in range(self.population_size - len(new_population)):
                mutate_gene = self.selection(wheel, 1)[0]
                new_population.append(self.mutation(population[mutate_gene]))
            # print(time.time())
            fitness, wheel = self.fitness(new_population)
            # print(time.time())
            # print(55555555555555555555555555555555555555555555555555555555555)
            fitness_curve.append(1 / max(fitness))
            population = new_population
        print(f'consume time:{time.time() - start}')
        self.plot_result(population[fitness.index(max(fitness))], fitness_curve)

    def plot_result(self, best_solution, performance):
        def dubins_plot(state_list, j, time_):
            distance = 0
            route_state = [[] for _ in range(2)]
            arrow = []
            for i in range(len(state_list) - 2):
                state_list[i + 1][2] *= np.pi / 180
            for i in range(len(state_list) - 1):
                sp = state_list[i]
                gp = state_list[i + 1] if state_list[i] != state_list[i + 1] \
                    else [state_list[i + 1][0], state_list[i + 1][1], abs(state_list[i + 1][2] - 1e-3)]
                dubins_path = dubins.shortest_path(sp, gp, self.Rmax[j])
                path, _ = dubins_path.sample_many(.1)
                route_state[0].extend([a[0] for a in path])
                route_state[1].extend([a[1] for a in path])
                distance += dubins_path.path_length()
                try:
                    time_[i].append(distance / self.velocity[j])
                except:
                    pass
            arrow.extend(
                [[route_state[0][arr], route_state[1][arr], route_state[0][arr + 100], route_state[1][arr + 100]]
                 for arr in range(0, len(route_state[0]), 7000)])
            return distance, route_state, arrow

        print(f'best gene:{best_solution}')
        dist = np.zeros(self.uav_num)
        task_sequence_state = [[] for _ in range(self.uav_num)]
        task_route = [[] for _ in range(self.uav_num)]
        route_state = [[] for _ in range(self.uav_num)]
        arrow_state = [[] for _ in range(self.uav_num)]
        for j in range(self.target_num * self.mission_num):
            task_sequence_state[best_solution[3][j] - 1].append([
                self.target_sites[best_solution[1][j] - 1][0], self.target_sites[best_solution[1][j] - 1][1],
                best_solution[4][j]])
            task_route[best_solution[3][j] - 1].extend([[best_solution[1][j], best_solution[2][j]]])
        for j in range(self.uav_num):
            task_sequence_state[j] = [self.uav_sites[j]] + task_sequence_state[j] + [self.terminal_sites[j]]
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
        plt.subplot(121)
        color_style = ['tab:blue', 'tab:green', 'tab:orange']
        for i in range(self.uav_num):
            # l = 2.0
            # plt.plot([sx, sx + l * np.cos(stheta)], [sy, sy + l * np.sin(stheta)], 'r-')
            # plt.plot([gx, gx + l * np.cos(gtheta)], [gy, gy + l * np.sin(gtheta)], 'r-')
            plt.plot(route_state[i][0], route_state[i][1], '-', color=color_style[i])
            plt.plot(self.uav_sites[i][0], self.uav_sites[i][1], 'ro')
            plt.plot(self.terminal_sites[i][0], self.terminal_sites[i][1], 'cs')
            plt.axis("equal")
            for arrow in arrow_state[i]:
                plt.arrow(arrow[0], arrow[1], arrow[2] - arrow[0], arrow[3] - arrow[1], width=8, color=color_style[i])
        plt.plot([b[0] for b in self.target_sites], [b[1] for b in self.target_sites], 'bo')
        plt.subplot(122)
        plt.plot(range(self.iteration + 1), performance)
        plt.title('cost = {:.3f}\n total distance = {:.0f}\n time = {:.3f}'.format(performance[-1], np.sum(dist),
                                                                                   np.max(dist / self.velocity)))
        plt.show()


if __name__ == '__main__':
    # targets = [[random.randint(-1000, 1000), random.randint(-1000, 1000)] for m in range(5)]
    targets = [[0, 400], [200, 800], [-700, 350], [-900, 650], [860, 840]]
    # uav = [[random.randint(-1000, 1000),
    #         random.randint(-1000, 1000),
    #         random.randint(0, 360)] for n in range(3)]
    uav = [[-500, 0, np.pi / 2], [0, 0, np.pi / 2], [500, 0, np.pi / 2]]
    terminal = [[-500, 1200, np.pi / 2], [0, 1200, np.pi / 2], [500, 1200, np.pi / 2]]
    # specification = [[25, 75, 2], [15, 50, 2], [35, 100, 2]]
    specification = [[25, 15, 35], [75, 50, 100], [1, 2, 3]]
    ga_sead_ = GA_task_allocation(targets, uav, terminal, 3, specification)
    ga_sead_.GA_SEAD()
