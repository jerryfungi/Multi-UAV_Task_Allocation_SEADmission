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
            'optimization operation'
            solution, fitness_value, population = mission.run_GA_time_period_version(output_interval, uavs, population,
                                                                                     update, distributed=True)
            'transmit the best solution to the main process'
            ga2control_queue.put([fitness_value, solution])
            if not control2ga_queue.empty():
                'get the information of other UAVs from the main process'
                uavs = control2ga_queue.get()
                update = True
                'disable the task allocation process'
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

        # path following setting
        recede_horizon = 1
        path_window = 50
        desire_point_index = 0
        proj_value = 10
        waypoint_radius = 80
        previous_time, previous_broadcast_time, previous_u2g_time = 0, 0, 0

        terminated_tasks, new_targets = [], []
        packets, path, target = [], [], None
        fitness, best_solution = 0, []
        task_confirm = False
        update = True
        into = False
        plot_index = 0
        AT, NT = [], []
        back_to_base, return_pub = False, False
        failure = False
        task_type = ["classification task", "attack task", "verification task"]

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
            ' <<<<<<<<<<<<<<< Communication Phase >>>>>>>>>>>>>>> '
            'receive solution from task allocation process'
            while not ga2control_queue.empty():
                fitness, best_solution = ga2control_queue.get()

            'broadcast the information'
            if int(time.time() * 10) % int(interval * 10) == 0 and time.time() - previous_broadcast_time >= interval / 1.01:
                previous_broadcast_time = time.time()
                current_best_packet, pos = pack_broadcast_packet(fitness, best_solution, [x_n, y_n, theta_n])
                packets.append(current_best_packet)
                for q in broadcast_list:
                    u2u_communication[q].put(current_best_packet)
                terminated_tasks, new_targets = [], []
                receive_confirm = True

            'receive the information from the task allocation process'
            if int(time.time() * 10 % 10) == 3 and receive_confirm:
                while not u2u_communication[uav.id - 1].empty():
                    packets.append(u2u_communication[uav.id - 1].get(timeout=1e-5))
                to_ga_message, fix_target, at, nt = repack_packets2ga_thread(packets)
                AT.extend(at)
                NT.extend(nt)
                to_ga_message[8], to_ga_message[9] = AT, NT
                'check the unknown targets'
                for target_found in to_ga_message[9]:
                    if target_found not in targets_sites:
                        targets_sites.append(target_found)
                'task-locking mechanism'
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
                        '''
                        choose the solution with the highest fitness value (when it comes to the same fitness value, 
                                                                        choose the solution with the smallest UAV ID)
                        '''
                        path, target, desire_point_index = \
                            generate_path(sorted(packets, key=lambda f: f[6], reverse=True)[0][7],
                                          pos, path, target, desire_point_index)
                    update = True
                else:
                    update = False
                packets.clear()
                receive_confirm = False

            ' <<<<<<<<<<<<<<< Control Phase >>>>>>>>>>>>>>> '
            if path and time.time() - previous_time >= .1:
                'back to the base or not'
                if math.hypot(uav.depot[0] - x_n, uav.depot[1] - y_n) <= waypoint_radius and back_to_base and not return_pub:
                    print(f'UAV {uav.id} => mission completed: {np.round(time.time() - start_time, 3)} sec')
                    return_pub = True
                    control2ga_queue.put([44])

                'identify the target is reached or not'
                if math.hypot(target[0][0] - x_n, target[0][1] - y_n) <= waypoint_radius and not into:
                    into = True
                'the condition of the task completed'
                if math.hypot(target[0][0] - x_n, target[0][1] - y_n) >= waypoint_radius and into:
                    if target[0][3:]:
                        terminated_tasks.append(target[0][3:])
                        print(f"UAV {uav.id} => Target {target[0][3]} {task_type[target[0][4] - 1]} finished: {np.round(time.time() - start_time, 3)} sec")
                        del target[0]
                        task_confirm = False
                        into = False

                'activate the task-locking mechanism or not'
                if math.hypot(target[0][0] - x_n, target[0][1] - y_n) <= 2 * uav.Rmin:
                    if target[0][3:]:
                        task_confirm = True
                    else:
                        back_to_base = True

                'return successfully and disarm the UAV'
                if (back_to_base and v <= 0.01) and not failure:
                    u2g.put([44, uav.id])
                    break

                'path-following algorithm'
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
                    'check the normal on line or not'
                    if max(path[i][0], path[i + 1][0]) > normal_point[0] \
                            > min(path[i][0], path[i + 1][0]) and max(path[i][1], path[i + 1][1]) > \
                            normal_point[1] > min(path[i][1], path[i + 1][1]):
                        normal_point = normal_point[:]
                    else:
                        normal_point = b
                    d = np.linalg.norm(va - (normal_point - a))
                    if d < world_record:
                        world_record = d
                        desire_point = normal_point
                        proj_value = np.dot(va, desire_point - np.array([x_n, y_n]))
                        desire_point_index = i
                'control to seek a waypoint'
                angle_between_two_points = angle_between((x_n, y_n), desire_point)
                relative_angle = angle_between_two_points - theta_n
                error_of_heading = relative_angle if abs(relative_angle) <= np.pi else \
                    (-relative_angle / abs(relative_angle)) * (relative_angle + 2 * np.pi)
                difference = error_of_heading - e_previous if e_previous else 0
                'control input for steering (PD control)'
                u = 3 * error_of_heading + 10 * difference
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
                'update the UAV states using velocity and yaw rate commands'
                x_n, y_n, theta_n, v, headingRate = step_pid(v, headingRate, x_n, y_n, theta_n, u, v_command, dt)

            'send the location to the GCS'
            if time.time() - previous_u2g_time >= 0.5:
                u2g.put([0, uav.id, x_n, y_n, time.time(), theta_n])
                previous_u2g_time = time.time()

            'dynamic environments'
            if uav_failure:
                'UAV failure'
                if time.time() - start_time >= uav_failure:
                    print(f'UAV {uav.id} failure --- {np.round(time.time() - start_time, 3)}')
                    u2g.put([223, uav.id, x_n, y_n])
                    u2g.put([44, uav.id])
                    break
            if unknown_targets:
                'unknown targets'
                for tt in unknown_targets:
                    if np.linalg.norm(np.array(tt) - np.array([x_n, y_n])) <= 400 \
                            and tt not in targets_sites and uav.type != 3:
                        new_targets.append(tt)
                        targets_sites.append(tt)
                        u2g.put([222, uav.id, tt[0], tt[1]])

    def start_simulation(self, realtime_plot=False, unknown_targets=None, uav_failure=None):
        uav_num = len(self.uavs[0])
        target_num = len(targets_sites)
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

        position = [[[UAVs[_].x0, UAVs[_].y0, 0]] for _ in range(uav_num)]
        x, y, yaw = [[UAVs[_].x0] for _ in range(uav_num)], [[UAVs[_].y0] for _ in range(uav_num)], \
                      [UAVs[_].theta0 for _ in range(uav_num)]
        distance = [0 for _ in range(uav_num)]
        state, completed = [1 for _ in range(uav_num)], [0 for _ in range(uav_num)]
        found_target, new_target_index, failures = [], [], [0 for _ in range(uav_num)]
        color_style = ['tab:blue', 'tab:green', 'tab:orange', '#DC143C', '#808080', '#030764', '#C875C4', '#008080',
                       '#DAA520', '#580F41', '#7BC8F6', '#06C2AC']  # 12 UAVs
        font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 8}
        font0 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
        font1 = {'family': 'Times New Roman', 'weight': 'normal', 'color': 'm', 'size': 8}
        font2 = {'family': 'Times New Roman', 'weight': 'normal', 'color': 'r', 'size': 8}
        font3 = {'family': 'Times New Roman', 'weight': 'normal', 'color': 'b', 'size': 8}

        origin_plot_list = [[p[0] for p in self.uavs[4]], [p[1] for p in self.uavs[4]]]
        targets_plot_list = [[p[0] for p in targets_sites], [p[1] for p in targets_sites]]
        base_plot_list = [[p[0] for p in self.uavs[5]], [p[1] for p in self.uavs[5]]]
        failure_uav_list = []

        theta = np.arange(0, 2 * np.pi, 0.1)
        fuselage = np.array([150 * np.cos(theta), 40 * np.sin(theta)])
        wing = 40 * np.array([[-0.5, -0.5, 0.5, 0.5, -0.5], [-6, 6, 6, -6, -6]])
        rotation_translation = lambda angle, pos, bias_x, bias_y: np.array(
            [np.add(pos[0] * np.cos(angle) + pos[1] * np.sin(angle), bias_x),
             np.add(-pos[0] * np.sin(angle) + pos[1] * np.cos(angle), bias_y)])
        mngr = plt.get_current_fig_manager()
        mngr.window.geometry("+1000+300")

        ' Start the programs of each UAV '
        for a in range(uav_num):
            main_process[a].start()
            task_allocation_proccess[a].start()
        start_time = time.time()
        print("Mission start !!")
        yaw_list = [[UAVs[_].theta0] for _ in range(uav_num)]

        ' Receive data from all agents '
        while state != completed:
            surveillance = GCS.get()
            if surveillance[0] == 222:  # Unknown target found
                found_target.append([surveillance[2], surveillance[3]])
                new_target_index.append(len(position[position.index(max(position, key=len))]))
                print(
                    f"target {[surveillance[2], surveillance[3]]} found by UAV{surveillance[1]}: {np.round(time.time() - start_time, 3)} sec")
            elif surveillance[0] == 223:  # UAV failure
                failures[self.uavs[0].index(surveillance[1])] = len(position[position.index(max(position, key=len))])
                failure_uav_list.append([surveillance[2], surveillance[3]])
            elif surveillance[0] == 44:  # UAV shut down
                state[self.uavs[0].index(surveillance[1])] = 0
            else:
                index = self.uavs[0].index(surveillance[1])
                position[index].append([surveillance[2], surveillance[3], surveillance[4]])
                x[index].append(surveillance[2])
                y[index].append(surveillance[3])
                yaw[index] = surveillance[5]
                yaw_list[index].append(surveillance[5])
                if realtime_plot:
                    plt.cla()
                    plt.axis("equal")

                    plt.plot(origin_plot_list[0], origin_plot_list[1], 'k^',
                             markerfacecolor='none', markersize=8)
                    plt.plot(targets_plot_list[0], targets_plot_list[1], 'ms',
                             markerfacecolor='none', markersize=6)
                    plt.plot(base_plot_list[0], base_plot_list[1], 'r*',
                             markerfacecolor='none', markersize=10)
                    plt.title(f"t = {np.round(time.time() - start_time, 3)}", font0)
                    for u in range(uav_num):
                        plt.plot(x[u], y[u], '--', linewidth=1, color=color_style[u])
                        plt.text(x[u][-1] + 100, y[u][-1] - 120, f'UAV {self.uavs[0][u]}', fontsize='8')
                        # Draw UAVs
                        if state[u]:
                            yaw_angle = -yaw[u]
                            uav_fuselage = rotation_translation(yaw_angle, fuselage, x[u][-1], y[u][-1])
                            uav_wing = rotation_translation(yaw_angle, wing, x[u][-1], y[u][-1])
                            plt.plot(uav_fuselage[0], uav_fuselage[1], 'k-', linewidth=1)
                            plt.fill_between(uav_wing[0], uav_wing[1], facecolor="black")
                    for t in range(target_num):
                        plt.text(targets_sites[t][0] + 100, targets_sites[t][1] + 100, f'Target {t + 1}', font1)
                    i = 1
                    for t in found_target:
                        plt.plot(t[0], t[1], 'bs', markerfacecolor='none', markersize=6)
                        plt.text(t[0] + 100, t[1] + 100, f'Target {target_num + i}', font3)
                        i += 1
                    for fail in failure_uav_list:
                        plt.plot(fail[0], fail[1], 'kx', markersize=12)
                    plt.pause(1e-5)
        mission_time = time.time() - start_time

        print("mission complete")
        input("....")
        if not realtime_plot:
            position_total = max(position, key=len)
            for tt in range(len(position_total)):
                plt.cla()
                plt.axis("equal")
                plt.plot(origin_plot_list[0], origin_plot_list[1], 'k^',
                         markerfacecolor='none', markersize=8)
                plt.plot(targets_plot_list[0], targets_plot_list[1], 'ms',
                         markerfacecolor='none', markersize=6)
                plt.plot(base_plot_list[0], base_plot_list[1], 'r*',
                         markerfacecolor='none', markersize=10)
                plt.title(f"t = {np.round(position_total[tt][2] - start_time, 3)}", font0)
                for u in range(uav_num):
                    if tt < len(x[u]):
                        plt.plot(x[u][:tt], y[u][:tt], '--', linewidth=1, color=color_style[u])
                        plt.text(x[u][tt] + 100, y[u][tt] - 120, f'UAV {self.uavs[0][u]}', fontsize='8')
                        yaw_angle = -yaw_list[u][tt]
                        uav_fuselage = rotation_translation(yaw_angle, fuselage, x[u][tt], y[u][tt])
                        uav_wing = rotation_translation(yaw_angle, wing, x[u][tt], y[u][tt])
                        plt.plot(uav_fuselage[0], uav_fuselage[1], 'k-', linewidth=1)
                        plt.fill_between(uav_wing[0], uav_wing[1], facecolor="black")
                    else:
                        plt.plot(x[u], y[u], '--', linewidth=1, color=color_style[u])
                        plt.text(x[u][-1] + 100, y[u][-1] - 120, f'UAV {self.uavs[0][u]}', fontsize='8')
                for t in range(target_num):
                    plt.text(targets_sites[t][0] + 100, targets_sites[t][1] + 100, f'Target {t + 1}', font1)
                for m, n in enumerate(new_target_index):
                    if tt >= n:
                        plt.plot(found_target[m][0], found_target[m][1], 'bs', label='Target position',
                                      markerfacecolor='none', markersize=6)
                        plt.text(found_target[m][0] + 100, found_target[m][1] + 100,
                                      f'Target {len(targets_sites) + 1 + m}', font3)
                try:
                    for m in range(len(failures)):
                        if failures[m]:
                            if tt >= failures[m]:
                                plt.plot(position[m][-1][0], position[m][-1][1], 'x', markersize=12, color="black")
                except:
                    pass
                plt.pause(1e-3)
        input("----")
        ' Final trajectories plot '
        fig, ax = plt.subplots()
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        for i in range(len(position)):
            plt.plot([p[0] for p in position[i]], [p[1] for p in position[i]], '-', linewidth=1, color=color_style[i],
                     label=f'UAV {self.uavs[0][i]}')
        for h in range(uav_num):
            plt.text(self.uavs[4][h][0] - 100, self.uavs[4][h][1] - 200, f'UAV {self.uavs[0][h]}', fontsize='8')
            plt.plot(self.uavs[5][h][0], self.uavs[5][h][1], 'r*', markerfacecolor='none', markersize=10)
            if failures[h]:
                plt.plot(position[h][-1][0], position[h][-1][1], 'x', markersize=6, color="black")
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
        plt.title('Trajectories', font0)
        plt.xlabel('East, m', font0)
        plt.ylabel('North, m', font0)
        plt.grid()
        plt.show()

        'Time flow plot '
        fig, ax = plt.subplots(3, 2)
        part = int(len(max(position, key=len)) / 6)
        time_ = [np.round(position[position.index(max(position, key=len))][part*t-1][2]-start_time, 3) for t in range(1, 6)]
        time_.append(np.round(position[position.index(max(position, key=len))][-1][2]-start_time, 3))
        q = 1
        for j in range(3):
            for k in range(2):
                for i in range(len(position)):
                    ax[j][k].plot([p[0] for p in position[i][:part*q]], [p[1] for p in position[i][:part*q]], '--', linewidth=1, color=color_style[i])
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
                                ax[j][k].plot(position[m][-1][0], position[m][-1][1], 'x', markersize=6, color="black")
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
                ax[j][k].set_title(f't = {time_[q - 1]}s', font0)
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
    targets_sites = [[3100, 2600], [500, 2400]]
    uav_id = [1, 2, 3]
    uav_type = [1, 2, 3]
    cruise_speed = [70, 80, 90]
    turning_radii = [200, 250, 300]
    initial_states = [[700, 1200, -np.pi], [1500, 700, np.pi / 2], [3600, 1000, np.pi / 3]]
    base_locations = [[2500, 4500, np.pi / 2] for _ in range(3)]   # same base
    dynamic_SEAD_mission = DynamicSEADMissionSimulator(targets_sites, uav_id, uav_type, cruise_speed, turning_radii, initial_states, base_locations)
    dynamic_SEAD_mission.start_simulation(realtime_plot=False, unknown_targets=[[2800, 3050]], uav_failure=[False, False, 55])
