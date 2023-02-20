import random
import time
import math
import numpy as np
import threading
import queue
from matplotlib import pyplot as plt
import multiprocessing as mp
import dubins
from dubins_model import *
from GA_SEAD_process import *
import copy


class UAV(object):
    def __init__(self, uav_id, uav_type, uav_velocity, uav_Rmin, initial_position, depot):
        self.id = uav_id
        self.type = uav_type
        self.velocity = uav_velocity
        self.Rmin = uav_Rmin
        self.omega_max = self.velocity / self.Rmin
        self.x0 = initial_position[0]
        self.y0 = initial_position[1]
        self.theta0 = initial_position[2]
        self.depot = depot


def GA_thread(targets_sites, ga2control_queue, control2ga_queue):
    iteration = 100
    time_interval = 1.5
    ga_population = None
    sead_mission = GA_SEAD(targets_sites, 300)
    while True:
        uavs = control2ga_queue.get()  # get other uav information from task execution thread
        solution, fitness_value, ga_population = sead_mission.run_GA_time_period_version(time_interval, uavs,
                                                                                         ga_population)
        ga2control_queue.put([fitness_value, solution])  # put the best solution to task execution thread


def TaskExecution_thread(uav, targets_sites, u2u_communication, ga2control_queue, control2ga_queue, u2g):
    # build uav
    u = 0
    t = 0
    x_n = uav.x0
    y_n = uav.y0
    theta_n = uav.theta0
    list_for_u = [0]
    list_for_t = [0]
    actual_x = [uav.x0]
    actual_y = [uav.y0]

    # communication setting
    broadcast_list = [i for i in range(len(u2u_communication)) if not i + 1 == uav.id]
    broadcast_timestamp = 8
    receive_confirm = False
    broadcast_confirm = True
    terminated_tasks, new_targets = [], []
    packets, path, target = [], [], None
    fitness, best_solution = 0, []

    # path following setting
    recede_horizon = 1
    path_window = 50
    desire_point_index = 0
    waypoint_radius = 25
    previous_time = 0
    waypoint_confirm = False
    task_confirm = False
    plot_index = 0
    start_time = time.time()

    failure = False
    new_t = False

    def pack_broadcast_packet(fit, chromosome):
        return [uav.id, uav.type, uav.velocity, uav.Rmin, [x_n, y_n, theta_n], uav.depot, fit, chromosome,
                terminated_tasks, new_targets]

    def repack_packets2ga_thread(msg):
        repack_packets = [[] for _ in range(10)]
        for uav_msg in msg:
            for i in range(8):
                repack_packets[i].append(uav_msg[i])
            for i in range(2):
                if uav_msg[8+i]:
                    repack_packets[8+i].extend(uav_msg[8+i])
        return repack_packets

    def generate_path(chromosome):
        path_route, task_sequence_state = [], []
        if chromosome:
            for p in range(len(chromosome[0])):
                if chromosome[3][p] == uav.id:
                    assign_target = chromosome[1][p]
                    assign_heading = chromosome[4][p] * 10
                    task_sequence_state.append([targets_sites[assign_target - 1][0],
                                                targets_sites[assign_target - 1][1], assign_heading,
                                                assign_target, chromosome[2][p]])
            task_sequence_state.append(uav.depot)
            for state in task_sequence_state[:-1]:
                state[2] *= np.pi / 180
            dubins_path = dubins.shortest_path([x_n, y_n, theta_n], task_sequence_state[0][:3], uav.Rmin)
            path_route.extend(dubins_path.sample_many(uav.velocity / 10)[0])
            for p in range(len(task_sequence_state)-1):
                sp = task_sequence_state[p][:3]
                gp = task_sequence_state[p+1][:3] if task_sequence_state[p][:3] != task_sequence_state[p+1][:3] else \
                    [task_sequence_state[p+1][0], task_sequence_state[p+1][1], task_sequence_state[p+1][2] - 1e-3]
                dubins_path = dubins.shortest_path(sp, gp, uav.Rmin)
                path_route.extend(dubins_path.sample_many(uav.velocity / 10)[0][1:])
        return path_route, task_sequence_state

    while True:
        # communication layer
        #       broadcast
        if not ga2control_queue.empty() and not failure:
            #       ga thread to task execution thread
            fitness, best_solution = ga2control_queue.get()
            broadcast_confirm = True
        if broadcast_confirm and int(time.time() * 10 % 10) == broadcast_timestamp:
            current_best_packet = pack_broadcast_packet(fitness, best_solution)
            packets.append(current_best_packet)
            for q in broadcast_list:
                u2u_communication[q].put(current_best_packet)
            terminated_tasks, new_targets = [], []
            receive_confirm = True
            broadcast_confirm = False
            print(f'broadcast: {uav.id}')
        #       task execution thread to ga thread
        if receive_confirm and int(time.time() * 10 % 10) == 0:
            while not u2u_communication[uav.id - 1].empty():
                packets.append(u2u_communication[uav.id - 1].get(timeout=1e-5))
            packets.sort()
            # choose the solution which has the highest fitness value. (when it comes to the same fitness value,
            #                                                            choose the solution of the smaller uav id)
            to_ga_message = repack_packets2ga_thread(packets)
            control2ga_queue.put(to_ga_message)
            for unknown_target in to_ga_message[9]:
                print(f'loop {unknown_target}')
                if unknown_target not in targets_sites:
                    targets_sites.append(unknown_target)
            if not task_confirm and not to_ga_message[8]:
                path, target = generate_path(sorted(packets, key=lambda f: f[4], reverse=True)[0][7])
                desire_point_index = 0  # clear path index
            packets.clear()
            receive_confirm = False
            print(f'receive: {uav.id}')

        # control layer
        if path and time.time()-previous_time >= .1 and not failure:
            # back to the base or not
            if math.hypot(uav.depot[0]-x_n, uav.depot[1]-y_n) <= waypoint_radius and target[:-1] == []:
                print(f'UAV_{uav.id} mission complete')
                break
            # identify the task is completed or not
            if math.hypot(target[0][0] - x_n, target[0][1] - y_n) <= waypoint_radius:
                print(uav.id, target[0][3:], time.time())
                del target[0]
                task_confirm = False
            # fix the solution and path or not
            if math.hypot(target[0][0] - x_n, target[0][1] - y_n) <= 2*uav.Rmin and not terminated_tasks:
                task_confirm = True
                if target[0][3:]:
                    terminated_tasks.append(target[0][3:])
            # path following algorithm
            future_point = np.array(
                [x_n + uav.velocity*cos(theta_n)*recede_horizon, y_n + uav.velocity*sin(theta_n)*recede_horizon])
            world_record = 1e10
            desire_point = 0
            start = desire_point_index
            for i in range(start, start + path_window):
                try:
                    a = np.array([path[i][0], path[i][1]])
                    b = np.array([path[i + 1][0], path[i + 1][1]])
                except IndexError:
                    a = np.array([path[-2][0], path[-2][1]])
                    b = np.array([path[-1][0], path[-1][1]])
                    i = -2
                va = future_point - a
                vb = b - a
                projection = np.dot(va, vb) / np.dot(vb, vb) * vb
                normal_point = a + projection
                # check the normal in line or not
                if max(path[i][0], path[i + 1][0]) > normal_point[0]\
                        > min(path[i][0], path[i + 1][0]) and max(path[i][1], path[i + 1][1]) >\
                        normal_point[1] > min(path[i][1], path[i + 1][1]):
                    normal_point = normal_point[:]
                else:
                    normal_point = b
                # update distance
                d = np.linalg.norm(va - (normal_point - a))
                if d < world_record:
                    world_record = d
                    desire_point = normal_point + 0 * vb
                    desire_point_index = i
            # control to seek a target
            angle_between_two_points = angle_between((x_n, y_n), desire_point)
            dt = time.time() - previous_time if not previous_time == 0 else 0
            previous_time = time.time()
            x_n, y_n, theta_n = step(uav, x_n, y_n, theta_n, u, dt)
            relative_angle = angle_between_two_points - theta_n
            error_of_heading = relative_angle if abs(relative_angle) <= np.pi else \
                (-relative_angle/abs(relative_angle))*(relative_angle + 2*np.pi)
            if error_of_heading < 0:
                u = -1
            elif error_of_heading > 0:
                u = 1
            else:
                u = 0

            # actual_x.append(x_n)
            # actual_y.append(y_n)
            t += dt
            # list_for_u.append(u)
            # list_for_t.append(t)
            # plt.clf()
            # plt.plot(actual_x, actual_y, 'r-')
            # plt.plot([p[0] for p in path], [p[1] for p in path], 'k')
            # plt.pause(1e-10)
            if plot_index % 3 == 0:
                u2g.put([uav.id, x_n, y_n])
            plot_index += 1

            # dynamic setting
            if time.time() - start_time >= 60 and uav.id == 1:  # agent lost
                print(f'UAV {uav.id} failure')
                u2g.put([223, x_n, y_n])
                failure = True
            if time.time() - start_time >= 30 and uav.id == 2 and not new_t:
                new_targets.append([1800, 3600])
                u2g.put([222, 1800, 3600])
                new_t = True


if __name__ == '__main__':
    # targets = [[500, 3700], [2300, 2500], [2000, 3900]]
    targets = [[3100, 2600], [500, 2400], [1800, 2100]]
    # uavs = [[1, 2, 3], [1, 2, 3], [70, 80, 90], [200, 250, 300],
    #         [[1000, 300, -np.pi], [1500, 700, np.pi / 2], [3000, 0, np.pi / 3]],
    #         [[0, 0, -np.pi / 2], [0, 0, -np.pi / 2], [0, 0, -np.pi / 2]],
    #         [], [], [], []]
    uavs = [[1, 2, 3], [1, 2, 3], [70, 80, 90], [200, 250, 300],
            [[700, 1200, -np.pi], [1500, 700, np.pi / 2], [3600, 1000, np.pi / 3]],
            [[2500, 4500, np.pi / 2] for _ in range(3)],
            [], [], [], []]
    uav_num = len(uavs[0])
    broadcast = [mp.Queue() for _ in range(uav_num)]
    control2ga_thread = [mp.Queue() for _ in range(uav_num)]
    ga2control_thread = [mp.Queue() for _ in range(uav_num)]
    GCS = mp.Queue()
    # build uav
    UAVs = [UAV(uavs[0][n], uavs[1][n], uavs[2][n], uavs[3][n], uavs[4][n], uavs[5][n])for n in range(uav_num)]
    # GA
    GA_threads = [mp.Process(target=GA_thread, args=(targets, ga2control_thread[n], control2ga_thread[n]))
                  for n in range(uav_num)]
    # control and communication
    TaskSequenceExecution_threads = [mp.Process(target=TaskExecution_thread,
                                                args=(UAVs[n], targets, broadcast, ga2control_thread[n],
                                                      control2ga_thread[n], GCS))
                                     for n in range(uav_num)]
    for a in range(uav_num):
        GA_threads[a].start()
        TaskSequenceExecution_threads[a].start()
    for h in range(uav_num):
        # plt.plot(uavs[4][h][0], uavs[4][h][1], 'r^', markerfacecolor='none', markersize=8)
        plt.text(uavs[4][h][0] - 100, uavs[4][h][1] - 200, f'UAV {uavs[0][h]}', fontsize='8')
        plt.axis("equal")
    plt.plot([x[0] for x in uavs[4]], [x[1] for x in uavs[4]], 'k^', label='UAV start point',
             markerfacecolor='none', markersize=8)
    plt.plot([b[0] for b in targets], [b[1] for b in targets], 'ms', label='Target position',
             markerfacecolor='none', markersize=6)
    plt.plot(uavs[5][0][0], uavs[5][0][1], 'r*', markerfacecolor='none', markersize=10, label='Airport')
    color_style = ['tab:blue', 'tab:green', 'tab:orange', '#DC143C', '#808080', '#030764', '#C875C4', '#008080',
                   '#DAA520', '#580F41', '#7BC8F6', '#06C2AC']
    while True:
        try:
            surveillance = GCS.get(timeout=1e-5)
            # print(surveillance)
            if surveillance[0] == 222:
                plt.plot(surveillance[1], surveillance[2], 'bs', label='Target position',
                         markerfacecolor='none', markersize=6)
            elif surveillance[0] == 223:
                plt.plot(surveillance[1], surveillance[2], 'kx', label='failure position', markersize=6)
            else:
                plt.plot(surveillance[1], surveillance[2], 'o', color=color_style[surveillance[0] - 1], markersize=1)
                plt.pause(1e-10)
        except queue.Empty:
            pass
