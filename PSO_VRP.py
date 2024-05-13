import random
import time
import math
import copy
from matplotlib import pyplot as plt


class PSO_vrp(object):
    def __init__(self, targets_sites, uavs_sites):
        # input data
        self.targets_sites = targets_sites
        self.uavs_sites = uavs_sites
        # PSO parameters
        self.particle_num = 14
        self.w = 0.729
        self.c1 = 2.05
        self.c2 = 2.05
        self.iteration = 200
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
                                                   math.pow(self.node[i][x][1]-self.node[i][y][1],2))
                                         for y in range(self.target_num+1)])(j) for j in range(self.target_num+1)]
        # constraints
        self.uav_velocity = 1
        # performance
        self.Pbest = [[0 for _ in range(self.particle_num)], [0 for _ in range(self.particle_num)]]
        self.Gbest = [0, 0, 0]
        self.training_curve = []

    def fitness(self, particles, route):
        fitness_value = []
        for k in range(len(route)):
            dist = [0 for _ in range(self.uav_num)]
            for j in range(self.uav_num):
                pre_pos = 0
                for i in range(1, len(route[k][j])):
                    target = route[k][j][i]
                    dist[j] += self.distance_mat[j][pre_pos][target]
                    pre_pos = target
            fitness = 1/(max(dist)+0.01*sum(dist))
            fitness_value.extend([fitness])
            self.pbest_update(fitness, k, particles[k][0])
        best_index = self.Pbest[0].index(max(self.Pbest[0]))
        self.gbest_update(max(fitness_value), particles[best_index][0], route[best_index])

    def decode_solution(self, particles):
        route = [[[0] for i in range(self.uav_num)] for j in range(self.particle_num)]
        for i in range(self.particle_num):
            sorted_target = sorted(range(len(particles[i][0])), key=lambda u: particles[i][0][u])  # target sequence
            sorted_pos = [math.floor(pos) for pos in sorted(particles[i][0])]  # assign agent
            for j in range(self.target_num):
                route[i][sorted_pos[j]-1].append(sorted_target[j]+1)
            for j in range(self.uav_num):
                route[i][j].append(0)
        return route

    def initiate_particles(self):
        def generate_particle():
            particle = [[], []]
            for i in range(self.target_num):
                particle[0].append(round(random.uniform(1, self.uav_num+1), 5)) # position [1, m]
                # particle[1].append(round(random.uniform(-(self.uav_num+1), self.uav_num+1), 5)) # velocity [-m, m]
                particle[1].append(round(random.uniform(-1, 1), 5))
            return particle
        return [generate_particle() for _ in range(self.particle_num)]

    def pbest_update(self, new_dot_fitness, dot_num, particle_pos):
        if self.Pbest[0][dot_num] < new_dot_fitness:
            self.Pbest[0][dot_num] = copy.deepcopy(new_dot_fitness)
            self.Pbest[1][dot_num] = copy.deepcopy(particle_pos)

    def gbest_update(self, new_max_fitness, particle_pos, best_route):
        if self.Gbest[0] < new_max_fitness:
            self.Gbest[0] = copy.deepcopy(new_max_fitness)
            self.Gbest[1] = copy.deepcopy(particle_pos)
            self.Gbest[2] = copy.deepcopy(best_route)

    def particle_improve(self, particles, route, current_iteration):
        def limit_v(V, vmax, vmin):
            if V > vmax :
                V = vmax
            elif V < vmin :
                V = vmin
            return V

        def limit_x(X, xmax, xmin):
            if X >= xmax:
                X = xmax-1e-5
            elif X < xmin:
                X = xmin
            return X
        for j in range(self.particle_num):
            new_particle = [[], []]
            for i in range(self.target_num):
                # w = self.w - self.w*self.iteration/current_iteration
                new_particle[1].append(limit_v(round(self.w * particles[j][1][i] + self.c1 * random.random() * (self.Pbest[1][j][i] - particles[j][0][i]) \
                                      + self.c2 * random.random() * (self.Gbest[1][i] - particles[j][0][i]), 5), 1, -1))
                new_particle[0].append(limit_x(round(particles[j][0][i] + new_particle[1][i], 5), self.uav_num+1, 1))
            particles[j] = new_particle

    def two_opt_method(self, particle_pos, route):
        def route_cost(single_route, uav_id):
            dist, pre_pos = 0, 0
            for n in range(1, len(single_route)):
                target = single_route[n]
                dist += self.distance_mat[uav_id][pre_pos][target]
                pre_pos = target
            return dist
        for k in range(self.uav_num):
            # compare_cost = route_cost(route[k], k)
            best = route[k]
            for i in range(1, len(route[k])-2):
                for j in range(i+1, len(route[k])):
                    new_route = route[k][:]
                    new_route[i:j] = route[k][j - 1:i - 1:-1]
                    # new_route_cost = route_cost(new_route, k)
                    if route_cost(new_route, k) < route_cost(best, k):
                        best = new_route[:]
            route[k] = best
        sequence = []
        sorted_pos = sorted(particle_pos)
        for i in range(self.uav_num):
            sequence += route[i][1:len(route[i])-1]
        for i in range(self.target_num):
            particle_pos[sequence[i] - 1] = sorted_pos[i]

    def PSO_main(self):
        start = time.time()
        particles = self.initiate_particles()
        route = self.decode_solution(particles)
        self.fitness(particles, route)
        self.training_curve.append(1/self.Gbest[0])
        for i in range(self.iteration):
            for j in range(self.particle_num):
                self.two_opt_method(particles[j][0], route[j])
            self.fitness(particles, route)
            self.particle_improve(particles, route, i+1)
            route = self.decode_solution(particles)
            self.fitness(particles, route)
            self.training_curve.append(1/self.Gbest[0])
        print(f'past time = {time.time()-start}')
        self.plot_best(self.Gbest)

    def plot_best(self, best_solution):
        print(f"best particle = {best_solution[1]}")
        x = [[self.uavs_sites[_][0]] for _ in range(self.uav_num)]
        y = [[self.uavs_sites[_][1]] for _ in range(self.uav_num)]
        for j in range(self.uav_num):
            for i in range(1, len(best_solution[2][j])-1):
                x[j].extend([self.targets_sites[best_solution[2][j][i]-1][0]])
                y[j].extend([self.targets_sites[best_solution[2][j][i]-1][1]])
            x[j].extend([self.uavs_sites[j][0]])
            y[j].extend([self.uavs_sites[j][1]])
        print(f"route: {best_solution[2]}")
        plt.figure(figsize=(8, 4))
        mngr = plt.get_current_fig_manager()
        mngr.window.geometry("+1000+300")
        plt.subplot(121)
        for i in range(len(x)):
            plt.plot(x[i], y[i], marker="o")
        plt.xlabel("X-Axis", size=10)
        plt.ylabel("Y-Axis", size=10)
        plt.title("Trajectories", size=10)
        plt.subplot(122)
        plt.plot(range(self.iteration+1), self.training_curve)
        plt.title("Cost = {:.3f}".format(1/best_solution[0]), size=10)
        plt.show()


if __name__ == '__main__':
    targets_sites = [[-65,15],[27,66],[-51,58],[-19,34],[77,25],[50,50],[-23,91],[77,77],[0,39],
                     [71,95],[-25,10],[0,91],[30,30],[15,15],[-100,24],[38,75]]
    uavs_sites = [[0,0],[0,0],[0,0],[0,0]]
    gavrp = PSO_vrp(targets_sites, uavs_sites)
    gavrp.PSO_main()
