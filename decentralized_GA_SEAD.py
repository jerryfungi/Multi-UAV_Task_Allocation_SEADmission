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


class DynamicSEADMissionSimulator(object):
    def __init__(self, targets_sites, uav_id, uav_type, cruise_speed, turning_radii, initial_states, base_locations):
        self.targets_sites = targets_sites
        self.uavs = [uav_id, uav_type, cruise_speed, turning_radii, initial_states, base_locations, [], [], [], []]

    def task_allocation_process(self, ga2control_queue, control2ga_queue, output_interval=0.5):
        population, update = None, True
        mission = GA_SEAD(self.targets_sites, 100)
        uavs = control2ga_queue.get()
        while True:
            solution, fitness_value, population = mission.run_GA_time_period_version(output_interval, uavs, population,
                                                                                     update, distributed=True)
            ga2control_queue.put([fitness_value, solution])  # put the best solution to task execution thread
            if not control2ga_queue.empty():
                uavs = control2ga_queue.get()  # get other uav information from task execution thread
                update = True
                if uavs == [44]:
                    break
            else:
                update = False

    def main_process(self, uav, u2u_communication, ga2control_queue, control2ga_queue, u2g, unknown_targets=None, uav_failure=None):
        targets_sites = self.targets_sites[:]
        # build uav
        x_n = uav.x0
        y_n = uav.y0
        e_previous, v, headingRate = None, 0, 0
        theta_n = uav.theta0
        pos = 0
        # communication setting
        broadcast_list = [i for i in range(len(u2u_communication)) if not i + 1 == uav.id]
        receive_confirm = False
        interval = 2
        terminated_tasks, new_targets = [], []
        packets, path, target = [], [], None
        fitness, best_solution = 0, []
        # path following setting
        recede_horizon = 1
        path_window = 50
        desire_point_index = 0
        proj_value = 10
        waypoint_radius = 80
        previous_time, previous_time_ = 0, 0
        task_confirm = False
        update = True
        into = False
        plot_index = 0
        AT, NT = [], []
        back_to_base, return_pub = False, False
        failure = False

        def pack_broadcast_packet(fit, chromosome, position):
            return [uav.id, uav.type, uav.velocity, uav.Rmin, position, uav.depot, fit, chromosome,
                    terminated_tasks, new_targets, int(task_confirm)], position

        def repack_packets2ga_thread(msg):
            repack_packets = [[] for _ in range(10)]
            fixx = [fix[-1] for fix in msg]
            task_accomplished, new_target = [], []
            for uav_msg in msg:
                for i in range(8):
                    repack_packets[i].append(uav_msg[i])
                for i in range(2):
                    if uav_msg[8 + i]:
                        repack_packets[8 + i].extend(uav_msg[8 + i])
                task_accomplished.extend(uav_msg[8])
                new_target.extend(uav_msg[9])
            return repack_packets, fixx, task_accomplished, new_target

        def generate_path(chromosome, position, path, targ, index):
            if chromosome and not back_to_base:
                path_route, task_sequence_state = [], []
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
                dubins_path = dubins.shortest_path(position, task_sequence_state[0][:3], uav.Rmin)
                path_route.extend(dubins_path.sample_many(uav.velocity / 10)[0])
                for p in range(len(task_sequence_state) - 1):
                    sp = task_sequence_state[p][:3]
                    gp = task_sequence_state[p + 1][:3] if task_sequence_state[p][:3] != task_sequence_state[p + 1][
                                                                                         :3] else \
                        [task_sequence_state[p + 1][0], task_sequence_state[p + 1][1],
                         task_sequence_state[p + 1][2] - 1e-3]
                    dubins_path = dubins.shortest_path(sp, gp, uav.Rmin)
                    path_route.extend(dubins_path.sample_many(uav.velocity / 10)[0][1:])
                return path_route, task_sequence_state, 0
            elif back_to_base:
                return path, targ, index
            else:
                return [], [], 0

        start_time = time.time()
        while True:
            # communication layer
            #       broadcast
            while not ga2control_queue.empty():
                #       ga thread to task execution thread
                fitness, best_solution = ga2control_queue.get()

            if int(time.time() * 10) % int(interval * 10) == 0 and time.time() - previous_time_ >= interval / 1.01:
                previous_time_ = time.time()
                current_best_packet, pos = pack_broadcast_packet(fitness, best_solution, [x_n, y_n, theta_n])

                packets.append(current_best_packet)
                for q in broadcast_list:
                    u2u_communication[q].put(current_best_packet)
                terminated_tasks, new_targets = [], []
                receive_confirm = True

            #       task execution thread to ga thread
            if int(time.time() * 10 % 10) == 6 and receive_confirm:
                while not u2u_communication[uav.id - 1].empty():
                    packets.append(u2u_communication[uav.id - 1].get(timeout=1e-5))
                # choose the solution which has the highest fitness value. (when it comes to the same fitness value,
                #                                                            choose the solution of the smaller uav id)
                to_ga_message, fix_target, at, nt = repack_packets2ga_thread(packets)
                AT.extend(at)
                NT.extend(nt)
                to_ga_message[8], to_ga_message[9] = AT, NT

                for target_found in to_ga_message[9]:
                    if target_found not in targets_sites:
                        targets_sites.append(target_found)
                if sum(fix_target) == 0:
                    control2ga_queue.put(to_ga_message)
                    AT, NT = [], []
                    for task in terminated_tasks:
                        if task in to_ga_message[8]:
                            terminated_tasks.pop(terminated_tasks.index(task))
                    for task in new_targets:
                        if task in to_ga_message[9]:
                            new_targets.pop(new_targets.index(task))
                    if update:
                        path, target, desire_point_index = generate_path(
                            sorted(packets, key=lambda f: f[6], reverse=True)[0][7], pos, path, target, desire_point_index)
                    update = True
                else:
                    update = False

                packets.clear()
                receive_confirm = False

            # control layer
            if path and time.time() - previous_time >= .1:
                # back to the base or not
                if math.hypot(uav.depot[0] - x_n, uav.depot[1] - y_n) <= waypoint_radius and back_to_base and not return_pub:
                    print(f'UAV {uav.id} => mission complete {np.round(time.time() - start_time, 3)} sec')
                    return_pub = True
                    control2ga_queue.put([44])
                # identify the target is reached or not
                if math.hypot(target[0][0] - x_n, target[0][1] - y_n) <= waypoint_radius and not into:
                    into = True
                # 做完任務之條件
                if math.hypot(target[0][0] - x_n, target[0][1] - y_n) >= waypoint_radius and into:
                    if target[0][3:]:
                        terminated_tasks.append(target[0][3:])
                        print(uav.id, target[0][3:], time.time())
                        del target[0]
                        task_confirm = False
                        into = False
                # fix the target and path or not
                if math.hypot(target[0][0] - x_n, target[0][1] - y_n) <= 2 * uav.Rmin:
                    if target[0][3:]:
                        task_confirm = True
                    else:
                        back_to_base = True
                if (back_to_base and v <= 0.01) and not failure:
                    u2g.put([44, uav.id])
                    break

                # path following algorithm
                future_point = np.array(
                    [x_n + uav.velocity * cos(theta_n) * recede_horizon,
                     y_n + uav.velocity * sin(theta_n) * recede_horizon])
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
                    if max(path[i][0], path[i + 1][0]) > normal_point[0] \
                            > min(path[i][0], path[i + 1][0]) and max(path[i][1], path[i + 1][1]) > \
                            normal_point[1] > min(path[i][1], path[i + 1][1]):
                        normal_point = normal_point[:]
                    else:
                        normal_point = b
                    # update distance
                    d = np.linalg.norm(va - (normal_point - a))
                    if d < world_record:
                        world_record = d
                        desire_point = normal_point
                        proj_value = np.dot(va, desire_point - np.array([x_n, y_n]))
                        desire_point_index = i
                # control to seek a target
                angle_between_two_points = angle_between((x_n, y_n), desire_point)
                relative_angle = angle_between_two_points - theta_n
                error_of_heading = relative_angle if abs(relative_angle) <= np.pi else \
                    (-relative_angle / abs(relative_angle)) * (relative_angle + 2 * np.pi)

                difference = error_of_heading - e_previous if e_previous else 0
                u = 3 * error_of_heading + 10 * difference  # yaw_rate_command (PD control)
                if u > v / uav.Rmin:
                    u = v / uav.Rmin
                elif u < -v / uav.Rmin:
                    u = -v / uav.Rmin
                e_previous = error_of_heading

                if proj_value <= 0 and back_to_base:
                    v_command, u = 0, 0
                else:
                    v_command = uav.velocity

                dt = time.time() - previous_time if not previous_time == 0 else .1
                previous_time = time.time()
                x_n, y_n, theta_n, v, headingRate = step_pid(v, headingRate, x_n, y_n, theta_n, u, v_command, dt)

                if plot_index % 3 == 0:
                    u2g.put([0, uav.id, x_n, y_n, time.time()])
                plot_index += 1

                # dynamic environments
                if uav_failure:
                    if time.time() - start_time >= uav_failure:  # agent lost
                        print(f'UAV {uav.id} failure --- {time.time() - start_time}')
                        u2g.put([223, uav.id, x_n, y_n])
                        u2g.put([44, uav.id])
                        break
                if unknown_targets:
                    for tt in unknown_targets:
                        if np.linalg.norm(np.array(tt) - np.array([x_n, y_n])) <= 400 \
                                and tt not in targets_sites and uav.type != 3:
                            new_targets.append(tt)
                            targets_sites.append(tt)
                            u2g.put([222, uav.id, tt[0], tt[1]])

    def start_simulation(self, realtime_plot=False, unknown_targets=None, uav_failure=None):
        uav_num = len(self.uavs[0])
        u2u_nodes = [mp.Queue() for _ in range(uav_num)]
        main2task_allocation = [mp.Queue() for _ in range(uav_num)]
        task_allocation2main = [mp.Queue() for _ in range(uav_num)]
        GCS = mp.Queue()
        uav_failure = [None for _ in range(uav_num)] if not uav_failure else uav_failure
        ' Build UAVs '
        UAVs = [UAV(self.uavs[0][n], self.uavs[1][n], self.uavs[2][n], self.uavs[3][n], self.uavs[4][n], self.uavs[5][n]) for n in range(uav_num)]
        ' Task Allocation using a genetic algorithm '
        task_allocation_proccess = [mp.Process(target=self.task_allocation_process, args=(task_allocation2main[n], main2task_allocation[n])) for n in range(uav_num)]
        ' Control and Communication '
        main_process = [mp.Process(target=self.main_process, args=(UAVs[n], u2u_nodes, task_allocation2main[n], main2task_allocation[n], GCS, unknown_targets, uav_failure[n])) for n in range(uav_num)]
        ' Start the programs of each UAV '
        for a in range(uav_num):
            main_process[a].start()
            task_allocation_proccess[a].start()
        start_time = time.time()
        print("Mission start !!")

        position = [[[UAVs[i].x0, UAVs[i].y0]] for _ in range(uav_num)]
        distance = [0 for _ in range(uav_num)]
        alive, completed = [1 for _ in range(uav_num)], [0 for _ in range(uav_num)]
        found_target, new_target_index, failures = [], [], [0 for _ in range(uav_num)]
        color_style = ['tab:blue', 'tab:green', 'tab:orange', '#DC143C', '#808080', '#030764', '#C875C4', '#008080',
                       '#DAA520', '#580F41', '#7BC8F6', '#06C2AC']  # 12 UAvs

        ' Receive data from all agents '
        while alive != completed:
            surveillance = GCS.get()
            if surveillance[0] == 222:
                if realtime_plot:
                    plt.plot(surveillance[2], surveillance[3], 'bs', label='Target position', markerfacecolor='none', markersize=6)
                found_target.append([surveillance[2], surveillance[3]])
                new_target_index.append(len(position[0]))
                print(
                    f"target {[surveillance[2], surveillance[3]]} found by UAV{surveillance[1]}: {np.round(time.time() - start_time, 3)} sec")
            elif surveillance[0] == 223:
                if realtime_plot:
                    plt.plot(surveillance[2], surveillance[3], 'kx', markersize=6)
                failures[self.uavs[0].index(surveillance[1])] = len(position[0])
            elif surveillance[0] == 44:
                alive[self.uavs[0].index(surveillance[1])] = 0
            else:
                index = self.uavs[0].index(surveillance[1])
                if realtime_plot:
                    plt.plot(surveillance[2], surveillance[3], 'o', color=color_style[index], markersize=1)
                    plt.pause(1e-10)
                position[index].append([surveillance[2], surveillance[3], surveillance[1]])
        mission_time = time.time() - start_time

        font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 8}
        font0 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
        font1 = {'family': 'Times New Roman', 'weight': 'normal', 'color': 'm', 'size': 8}
        font2 = {'family': 'Times New Roman', 'weight': 'normal', 'color': 'r', 'size': 8}
        font3 = {'family': 'Times New Roman', 'weight': 'normal', 'color': 'b', 'size': 8}
        font4 = {'family': 'Times New Roman', 'weight': 'normal', 'color': '#DC143C', 'size': 8}

        ' Final trajectories plot '
        fig, ax = plt.subplots()
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        for i in range(len(position)):
            plt.plot([p[0] for p in position[i]], [p[1] for p in position[i]], '-', linewidth=1, color=color_style[i],
                     label=f'UAV {self.uavs[0][i]}')
            # plt.text(position[i][0][0] - 100, position[i][0][1] - 200, f'UAV {uavs[0][i]}', font)
        for h in range(uav_num):
            # plt.plot(uavs[4][h][0], uavs[4][h][1], 'r^', markerfacecolor='none', markersize=8)
            plt.text(self.uavs[4][h][0] - 100, self.uavs[4][h][1] - 200, f'UAV {self.uavs[0][h]}', fontsize='8')
            plt.plot(self.uavs[5][h][0], self.uavs[5][h][1], 'r*', markerfacecolor='none', markersize=10)
            if failures[h]:
                plt.plot(position[h][-1][0], position[h][-1][1], 'x', markersize=6, color=color_style[h])
                plt.text(position[h][-1][0] + 100, position[h][-1][1] + 100, 'Dead',
                         {'family': 'Times New Roman', 'weight': 'normal', 'color': color_style[h], 'size': 8})
            plt.axis("equal")
        plt.plot([x[0] for x in self.uavs[4]], [x[1] for x in self.uavs[4]], 'k^',
                 markerfacecolor='none', markersize=8)
        plt.plot([b[0] for b in targets_sites], [b[1] for b in targets_sites], 'ms', label='Target position',
                 markerfacecolor='none', markersize=6)
        for m, n in enumerate(new_target_index):
            plt.plot(found_target[m][0], found_target[m][1], 'bs', label='Target position', markerfacecolor='none', markersize=6)
            plt.text(found_target[m][0] + 100, found_target[m][1] + 100, f'Target {len(targets_sites) + 1 + m}', font3)
        for t in targets_sites:
            plt.text(t[0] + 100, t[1] + 100, f'Target {targets_sites.index(t) + 1}', font1)
        plt.text(UAVs[0].depot[0] - 100, UAVs[0].depot[1] - 200, 'Base', font2)
        plt.plot(self.uavs[5][0][0], self.uavs[5][0][1], 'r*', markerfacecolor='none', markersize=10, label='Base')

        plt.legend(loc='upper right', prop=font)
        plt.xlabel('East, m', font0)
        plt.ylabel('North, m', font0)
        plt.grid()
        plt.show()

        'Time flow plot '
        fig, ax = plt.subplots()
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        part = int(len(max(position, key=len)) / 6)
        time_ = [position[position.index(max(position, key=len))][part*t-1][2]-start_time for t in range(1, 6)]
        time_.append(position[position.index(max(position, key=len))][-1][2]-start_time)
        fig, ax = plt.subplots(3, 2)
        q = 1
        for j in range(3):
            for k in range(2):
                labels = ax[j][k].get_xticklabels() + ax[j][k].get_yticklabels()
                [label.set_fontname('Times New Roman') for label in labels]
                for i in range(len(position)):
                    ax[j][k].plot([p[0] for p in position[i][:part*q]], [p[1] for p in position[i][:part*q]], '--', linewidth=1, color=color_style[i])
                    # plt.plot(uavs[4][h][0], uavs[4][h][1], 'r^', markerfacecolor='none', markersize=8)
                    ax[j][k].text(position[i][0][0], position[i][0][1], f'UAV {self.uavs[0][i]}', fontsize='8')
                    try:
                        ax[j][k].plot(position[i][part*q-1][0], position[i][part*q-1][1], 'o', markerfacecolor='none', markersize=5, color=color_style[i])
                    except:
                        pass
                    ax[j][k].plot(position[i][0][0], position[i][0][1], 'k^',
                             markerfacecolor='none', markersize=8)
                for m, n in enumerate(new_target_index):
                    if part * q >= n:
                        ax[j][k].plot(found_target[m][0], found_target[m][1], 'bs', label='Target position',
                                      markerfacecolor='none', markersize=6)
                        ax[j][k].text(found_target[m][0] + 100, found_target[m][1] + 100, f'Target {len(targets_sites) + 1 + m}', font3)
                try:
                    for m in range(len(failures)):
                        if failures[m]:
                            if part * q >= failures[m]:
                                ax[j][k].plot(position[m][-1][0], position[m][-1][1], 'x', markersize=6, color=color_style[3])
                                ax[j][k].text(position[m][-1][0] + 100, position[m][-1][1] + 100, 'Dead',
                                              {'family': 'Times New Roman', 'weight': 'normal', 'color': color_style[h], 'size': 8})
                except:
                    pass
                ax[j][k].plot([b[0] for b in targets_sites], [b[1] for b in targets_sites], 'ms', label='Target position',
                         markerfacecolor='none', markersize=6)
                for t in targets_sites:
                    ax[j][k].text(t[0] + 100, t[1] + 100, f'Target {targets_sites.index(t) + 1}', font1)
                ax[j][k].text(UAVs[0].depot[0] + 200, UAVs[0].depot[1] + 100, 'Base', font2)
                ax[j][k].plot(self.uavs[5][0][0] + 100, self.uavs[5][0][1] + 200, 'r*', markerfacecolor='none', markersize=10, label='Base')
                ax[j][k].set_xlabel('East, m', font0)
                ax[j][k].set_ylabel('North, m', font0)
                ax[j][k].set_title(f't = {round(time_[q-1], 3)} s', font0)
                ax[j][k].axis("equal")
                q += 1
        fig.tight_layout()
        plt.subplots_adjust(wspace=.3, hspace=.3)
        plt.show()

        ' Numerical results '
        print(" <<<<<<<<<< <<<<<<<<<<< Numerical results >>>>>>>>>>> >>>>>>>>>> ")
        u = 0
        for uav in position:
            for p in range(1, len(uav) - 1):
                distance[u] += np.linalg.norm(np.subtract(uav[p][:2], uav[p - 1][:2]))
            u += 1
        objectives = mission_time + np.sum(distance) / np.sum(self.uavs[2])
        print(f"mission time: {mission_time} \n"
              f"total distance: {np.sum(distance)} \n"
              f"objectives: {objectives}"
              f"distance: {distance}")


if __name__ == '__main__':
    scenario_1 = {"targets_sites": [[3100, 2600], [500, 2400], [1800, 1600], [1800, 3600]],
                  "uav_id": [1, 2, 3],
                  "uav_type": [1, 2, 3],
                  "cruise_speed": [70, 80, 90],
                  "turning_radii": [200, 250, 300],
                  "initial_states": [[1000, 300, -np.pi], [1500, 700, np.pi / 2], [3000, 0, np.pi / 3]],
                  "base_locations": [[2500, 4500, np.pi / 2] for _ in range(3)],
                  "unknown_targets": None,
                  "uav_failure": None}

    scenario_2 = {"targets_sites": [[3850, 1650], [3900, 4700], [2000, 2050], [4800, 3600], [2800, 3900], [150, 3600]],
                  "uav_id": [1, 2, 3, 4, 5, 6],
                  "uav_type": [1, 2, 3, 2, 1, 2],
                  "cruise_speed": [70, 80, 90, 60, 100, 80],
                  "turning_radii": [200, 250, 300, 180, 300, 260],
                  "initial_states": [[500, 300, -60*np.pi/180], [1500, 700, 90*np.pi/180], [200, 1100, 135*np.pi/180]],
                  "base_locations": [[0, 5000, 140*np.pi/180] for _ in range(6)],
                  "unknown_targets": None,
                  "uav_failure": None}

    scenario_3 = {"targets_sites": [[3100, 2600], [500, 2400], [1800, 1600], [1800, 3600]],
                  "uav_id": [1, 2, 3],
                  "uav_type": [1, 2, 3],
                  "cruise_speed": [70, 80, 90],
                  "turning_radii": [200, 250, 300],
                  "initial_states": [[1000, 300, -np.pi], [1500, 700, np.pi / 2], [3000, 0, np.pi / 3]],
                  "base_locations": [[0, 0, -np.pi / 2] for _ in range(3)],
                  "unknown_targets": [[2500, 2850]],
                  "uav_failure": None}

    scenario_4 = {"targets_sites": [[3850, 1650], [3900, 4700], [2000, 2050], [4800, 3600], [2800, 3900], [150, 3600]],
                  "uav_id": [1, 2, 3, 4, 5, 6],
                  "uav_type": [1, 2, 3, 2, 1, 2],
                  "cruise_speed": [70, 80, 90, 60, 100, 80],
                  "turning_radii": [200, 250, 300, 180, 300, 260],
                  "initial_states": [[500, 300, -60*np.pi/180], [1500, 700, 90*np.pi/180], [200, 1100, 135*np.pi/180]],
                  "base_locations": [[0, 5000, 140*np.pi/180] for _ in range(6)],
                  "unknown_targets": [[3500, 3000], [1400, 4600]],
                  "uav_failure": [None, None, None, 70, None, None]}

    targets_sites = [[3100, 2600], [500, 2400]]
    uav_id = [1, 2, 3]
    uav_type = [1, 2, 3]
    cruise_speed = [70, 80, 90]
    turning_radii = [200, 250, 300]
    initial_states = [[700, 1200, -np.pi], [1500, 700, np.pi / 2], [3600, 1000, np.pi / 3]]
    base_locations = [[2500, 4500, np.pi / 2] for _ in range(3)]
    dynamic_SEAD_mission = DynamicSEADMissionSimulator(targets_sites, uav_id, uav_type, cruise_speed, turning_radii, initial_states, base_locations)
    dynamic_SEAD_mission.start_simulation(realtime_plot=False, unknown_targets=[[600, 2850]], uav_failure=[False, False, 55])

