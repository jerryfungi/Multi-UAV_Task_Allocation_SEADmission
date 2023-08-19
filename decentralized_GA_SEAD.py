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
    time_interval = 0.5
    ga_population, update = None, True
    sead_mission = GA_SEAD(targets_sites, 100)
    uavs = control2ga_queue.get()
    while True:
        solution, fitness_value, ga_population = sead_mission.run_GA_time_period_version(time_interval, uavs,
                                                                                         ga_population, update)
        ga2control_queue.put([fitness_value, solution])  # put the best solution to task execution thread
        if not control2ga_queue.empty():
            uavs = control2ga_queue.get()  # get other uav information from task execution thread
            update = True
            if uavs == [44]:
                break
        else:
            update = False



def TaskExecution_thread(uav, targets_sites, u2u_communication, ga2control_queue, control2ga_queue, u2g):
    # build uav
    u = 0
    t = 0
    x_n = uav.x0
    y_n = uav.y0
    e_previous, v, headingRate = None, 0, 0
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
    interval = 2
    terminated_tasks, new_targets = [], []
    packets, path, target = [], [], None
    fitness, best_solution = 0, []

    # path following setting
    recede_horizon = 1
    path_window = 50
    desire_point_index = 0
    waypoint_radius = 5
    previous_time, previous_time_ = 0, 0
    waypoint_confirm = False
    task_confirm = False
    update = True
    into = False
    plot_index = 0
    AT, NT = [], []
    back_to_base = False
    failure = False
    # unknown_target = [[2500, 2850]]
    # unknown_target = [[3500, 3000], [1400, 4600]]
    unknown_target = [[-115, 30]]

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
                if uav_msg[8+i]:
                    repack_packets[8+i].extend(uav_msg[8+i])
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
            for p in range(len(task_sequence_state)-1):
                sp = task_sequence_state[p][:3]
                gp = task_sequence_state[p+1][:3] if task_sequence_state[p][:3] != task_sequence_state[p+1][:3] else \
                    [task_sequence_state[p+1][0], task_sequence_state[p+1][1], task_sequence_state[p+1][2] - 1e-3]
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

        if int(time.time() * 10) % int(interval * 10) == 0 and time.time() - previous_time_ >= interval/1.01:
            previous_time_ = time.time()
            current_best_packet, pos = pack_broadcast_packet(fitness, best_solution, [x_n, y_n, theta_n])

            packets.append(current_best_packet)
            for q in broadcast_list:
                u2u_communication[q].put(current_best_packet)
            terminated_tasks, new_targets = [], []
            receive_confirm = True
            # print(f'broadcast: {uav.id}')

        #       task execution thread to ga thread
        if int(time.time() * 10 % 10) == 6 and receive_confirm:
            while not u2u_communication[uav.id - 1].empty():
                packets.append(u2u_communication[uav.id - 1].get(timeout=1e-5))
            # packets = consider_communication_range(packets, pos)
            # packets.sort()
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
                # for task in terminated_tasks:
                #     if task in to_ga_message[8]:
                #         terminated_tasks.pop(terminated_tasks.index(task))
                # for task in new_targets:
                #     if task in to_ga_message[9]:
                #         new_targets.pop(new_targets.index(task))
                if update:
                    path, target, desire_point_index = generate_path(sorted(packets, key=lambda f: f[6], reverse=True)[0][7], pos, path, target, desire_point_index)
                    # desire_point_index = 0  # clear path index
                    # print('update')
                update = True
            else:
                update = False
            # print(to_ga_message[7])

            # if not task_confirm:
            #     path, target = generate_path(sorted(packets, key=lambda f: f[6], reverse=True)[0][7], pos)
            #     desire_point_index = 0  # clear path index

            packets.clear()
            receive_confirm = False
            # print(f'receive: {uav.id}')

        # control layer
        if path and time.time()-previous_time >= .1:
            # back to the base or not
            if math.hypot(uav.depot[0]-x_n, uav.depot[1]-y_n) <= waypoint_radius and target[:-1] == []:
                # print(f'UAV_{uav.id} mission complete ({time.time()})')
                u2g.put([44, uav.id])
                control2ga_queue.put([44])
                # break
            # identify the target is reached or not
            if math.hypot(target[0][0] - x_n, target[0][1] - y_n) <= waypoint_radius and not into:
                # print(uav.id, target[0][3:], time.time())
                into = True
            # 做完任務之條件
            if math.hypot(target[0][0] - x_n, target[0][1] - y_n) >= waypoint_radius and into:
                if target[0][3:]:
                    terminated_tasks.append(target[0][3:])
                    print(uav.id, target[0][3:], time.time())
                    del target[0]
                    # terminated_tasks.clear()
                    task_confirm = False
                    into = False
            # fix the target and path or not
            if math.hypot(target[0][0] - x_n, target[0][1] - y_n) <= 2*uav.Rmin:
                if target[0][3:]:
                    task_confirm = True
                else:
                    back_to_base = True
                # if target[0][3:] not in terminated_tasks and target[0][3:]:
                #     terminated_tasks.append(target[0][3:])
            if (back_to_base and v <= 0.01) and not failure:
                break
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
                    desire_point = normal_point
                    proj_value = np.dot(va, desire_point - np.array([x_n, y_n]))
                    desire_point_index = i
            # control to seek a target
            angle_between_two_points = angle_between((x_n, y_n), desire_point)
            relative_angle = angle_between_two_points - theta_n
            error_of_heading = relative_angle if abs(relative_angle) <= np.pi else \
                (-relative_angle/abs(relative_angle))*(relative_angle + 2*np.pi)

            difference = error_of_heading - e_previous if e_previous else 0
            u = 3 * error_of_heading + 10 * difference  # yaw_rate_command
            if u > v / uav.Rmin:
                u = v / uav.Rmin
            elif u < -v / uav.Rmin:
                u = -v / uav.Rmin
            e_previous = error_of_heading
            # v_command = uav.velocity if math.hypot(path[-1][0] - x_n, path[-1][1] - y_n) >= waypoint_radius else 0
            if proj_value <= 0 and back_to_base:
                v_command, u = 0, 0
            else:
                v_command = uav.velocity

            dt = time.time() - previous_time if not previous_time == 0 else .1
            previous_time = time.time()
            x_n, y_n, theta_n, v, headingRate = step_pid(v, headingRate, x_n, y_n, theta_n, u, v_command, dt)

            t += dt
            if plot_index % 3 == 0:
                u2g.put([uav.id, x_n, y_n, time.time()])
            plot_index += 1

            # dynamic environments
            # if time.time() - start_time >= 70 and uav.id == 4:  # agent lost
            #     print(f'UAV {uav.id} failure --- {time.time() - start_time}')
            #     # u2g.put([223, x_n, y_n])
            #     u2g.put([44, uav.id])
            #     break
            if time.time() - start_time >= 55 and uav.id == 3:  # agent lost
                print(f'UAV {uav.id} failure --- {time.time() - start_time}')
                # u2g.put([223, x_n, y_n])
                u2g.put([44, uav.id])
                break
            for tt in unknown_target:
                if np.linalg.norm(np.array(tt) - np.array([x_n, y_n])) <= 50 \
                        and tt not in targets_sites and uav.type != 3:
                    new_targets.append(tt)
                    targets_sites.append(tt)
                    u2g.put([222, tt[0], tt[1], uav.id])


if __name__ == '__main__':
    # targets = [[500, 3700], [2300, 2500], [2000, 3900]]
    # targets = [[3100, 2600], [500, 2400], [1800, 2100]]
    # targets = [[3100, 2600], [500, 2400], [1800, 1600], [1800, 3600]]  # t4
    # targets = [[3100, 2600]]
    # targets = [[3100, 2200], [500, 3700], [2300, 2500], [2000, 3900]]
    # targets = [[3850, 1650], [3900, 4700], [2000, 2050], [4800, 3600], [2800, 3900], [150, 3600]]
    # uavs = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 2, 1, 2], [70, 80, 90, 60, 100, 80], [200, 250, 300, 180, 300, 260],
    #         [[500, 300, -60*np.pi/180], [1500, 700, 90*np.pi/180], [200, 1100, 135*np.pi/180],
    #          [3500, 120, 20*np.pi/180], [5000, 1000, 135*np.pi/180], [4740, 2500, 115*np.pi/180]],
    #         [[0, 5000, 140*np.pi/180] for _ in range(6)],
    #         [], [], [], []]
    # uavs = [[1, 2, 3], [1, 2, 3], [70, 80, 90], [200, 250, 300],
    #         [[1000, 300, -np.pi], [1500, 700, np.pi / 2], [3000, 0, np.pi / 3]],
    #         [[0, 0, -np.pi / 2], [0, 0, -np.pi / 2], [0, 0, -np.pi / 2]],
    #         [], [], [], []]
    # targets = [[4500, 2000], [500, 2400], [1800, 1600], [3800, 3600]]
    # uavs = [[1, 2, 3], [2, 2, 2], [70, 80, 90], [200, 250, 300],
    #         [[1000, 300, 135*np.pi / 180], [100, 100, 45*np.pi / 180], [2700, 500, np.pi / 3]],
    #         [[300, 4700, 75*np.pi / 180], [300, 4700, 75*np.pi / 180], [5000, 4300, 0]],
    #         [], [], [], []]  # comm limit
    # uavs = [[1, 2, 3], [1, 2, 3], [70, 80, 90], [200, 250, 300],
    #         [[700, 1200, -np.pi], [1500, 700, np.pi / 2], [3600, 1000, np.pi / 3]],
    #         [[2500, 4500, np.pi / 2] for _ in range(3)],
    #         [], [], [], []]

    # real flights experiments
    targets = [[-80, 100], [-160, 70], [-110, -20]]
    uavs = [[1, 2, 3], [2, 3, 1], [3, 4, 5], [9, 12, 15],
            [[-25, 40, 200*np.pi/180], [-30, 20, 200*np.pi/180], [-20, 0, 200*np.pi/180]],
            [[-25, 40, 20*np.pi/180], [-30, 20, 20*np.pi/180], [-20, 0, 20*np.pi/180]],
            [], [], [], []]

    # targets = [[-200, 10], [-180, 150]]
    # uavs = [[1, 2, 3], [2, 1, 3], [18, 15, 13], [70, 60, 50],
    #         [[-25, 40, 200 * np.pi / 180], [-30, 20, 200 * np.pi / 180], [-20, 0, 200 * np.pi / 180]],
    #         [[-25, 40, 20 * np.pi / 180], [-30, 20, 20 * np.pi / 180], [-20, 0, 20 * np.pi / 180]],
    #         [], [], [], []]

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

    position = [[[UAVs[i].x0, UAVs[i].y0]] for i in range(uav_num)]
    distance = [0 for i in range(uav_num)]
    alive = [1 for i in range(uav_num)]
    fit_table = [[] for i in range(uav_num)]
    found_target, new_target_index = [], []

    for a in range(uav_num):
        GA_threads[a].start()
        TaskSequenceExecution_threads[a].start()
    start_time = time.time()
    print(f"start time: {start_time}")
    color_style = ['tab:blue', 'tab:green', 'tab:orange', '#DC143C', '#808080', '#030764', '#C875C4', '#008080',
                   '#DAA520', '#580F41', '#7BC8F6', '#06C2AC']
    while alive != [0 for _ in range(uav_num)]:
        surveillance = GCS.get()
        # print(surveillance)
        if surveillance[0] == 222:
            # plt.plot(surveillance[1], surveillance[2], 'bs', label='Target position',
            #          markerfacecolor='none', markersize=6)
            found_target.append([surveillance[1], surveillance[2]])
            new_target_index.append(len(position[0]))
            print(f"target {[surveillance[1], surveillance[2]]} found by UAV{surveillance[3]}: {time.time() - start_time} sec")
        elif surveillance[0] == 223:
            plt.plot(surveillance[1], surveillance[2], 'kx', label='failure position', markersize=6)
        elif surveillance[0] == 44:
            if surveillance[1] == 4:
                failure = len(position[0])
            alive[surveillance[1] - 1] = 0
        else:
            # plt.plot(surveillance[1], surveillance[2], 'o', color=color_style[surveillance[0] - 1], markersize=1)
            # plt.pause(1e-10)
            position[surveillance[0] - 1].append([surveillance[1], surveillance[2], surveillance[3]])
    mission_time = time.time() - start_time
    # print(position)

    fig, ax = plt.subplots()
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 8}
    font0 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'color': 'm', 'size': 8}
    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'color': 'r', 'size': 8}
    font3 = {'family': 'Times New Roman', 'weight': 'normal', 'color': 'b', 'size': 8}
    font4 = {'family': 'Times New Roman', 'weight': 'normal', 'color': '#DC143C', 'size': 8}


    # plt.legend(loc='upper right', prop=font)
    plt.xlabel('East, m', font0)
    plt.ylabel('North, m', font0)
    # plt.grid()

    # part = int(len(max(position, key=len)) / 6)
    # time_ = [position[position.index(max(position, key=len))][part*t-1][2]-start_time for t in range(1, 6)]
    # time_.append(position[position.index(max(position, key=len))][-1][2]-start_time)
    # fig, ax = plt.subplots(3, 2)
    # q = 1
    # for j in range(3):
    #     for k in range(2):
    #         labels = ax[j][k].get_xticklabels() + ax[j][k].get_yticklabels()
    #         [label.set_fontname('Times New Roman') for label in labels]
    #         for i in range(len(position)):
    #             ax[j][k].plot([p[0] for p in position[i][:part*q]], [p[1] for p in position[i][:part*q]], '--', linewidth=1, color=color_style[i])
    #             # plt.plot(uavs[4][h][0], uavs[4][h][1], 'r^', markerfacecolor='none', markersize=8)
    #             ax[j][k].text(position[i][0][0], position[i][0][1], f'UAV {uavs[0][i]}', fontsize='8')
    #             try:
    #                 ax[j][k].plot(position[i][part*q-1][0], position[i][part*q-1][1], 'o', markerfacecolor='none', markersize=5, color=color_style[i])
    #             except:
    #                 pass
    #             ax[j][k].plot(position[i][0][0], position[i][0][1], 'k^',
    #                      markerfacecolor='none', markersize=8)
    #         for m, n in enumerate(new_target_index):
    #             if part * q >= n:
    #                 ax[j][k].plot(found_target[m][0], found_target[m][1], 'bs', label='Target position',
    #                               markerfacecolor='none', markersize=6)
    #                 ax[j][k].text(found_target[m][0] + 100, found_target[m][1] + 100, f'Target {len(targets) + 1 + m}', font3)
    #         try:
    #             if part * q >= failure:
    #                 ax[j][k].plot(position[3][-1][0], position[3][-1][1], 'x', markersize=6, color=color_style[3])
    #                 ax[j][k].text(position[3][-1][0] + 100, position[3][-1][1] + 100, 'Dead', font4)
    #         except:
    #             pass
    #         ax[j][k].plot([b[0] for b in targets], [b[1] for b in targets], 'ms', label='Target position',
    #                  markerfacecolor='none', markersize=6)
    #         for t in targets:
    #             ax[j][k].text(t[0] + 100, t[1] + 100, f'Target {targets.index(t) + 1}', font1)
    #         ax[j][k].text(UAVs[0].depot[0] + 200, UAVs[0].depot[1] + 100, 'Base', font2)
    #         ax[j][k].plot(uavs[5][0][0] + 100, uavs[5][0][1] + 200, 'r*', markerfacecolor='none', markersize=10, label='Base')
    #         ax[j][k].set_xlabel('East, m', font0)
    #         ax[j][k].set_ylabel('North, m', font0)
    #         ax[j][k].set_title(f't = {round(time_[q-1], 3)} s', font0)
    #         ax[j][k].axis("equal")
    #         q += 1

    for i in range(len(position)):
        plt.plot([p[0] for p in position[i]], [p[1] for p in position[i]], '-', linewidth=1, color=color_style[i],
                 label=f'UAV {uavs[0][i]}')
        # plt.text(position[i][0][0] - 100, position[i][0][1] - 200, f'UAV {uavs[0][i]}', font)
    for h in range(uav_num):
        # plt.plot(uavs[4][h][0], uavs[4][h][1], 'r^', markerfacecolor='none', markersize=8)
        plt.text(uavs[4][h][0] - 100, uavs[4][h][1] - 200, f'UAV {uavs[0][h]}', fontsize='8')
        plt.axis("equal")
    plt.plot([x[0] for x in uavs[4]], [x[1] for x in uavs[4]], 'k^',
             markerfacecolor='none', markersize=8)
    plt.plot([b[0] for b in targets], [b[1] for b in targets], 'ms', label='Target position',
             markerfacecolor='none', markersize=6)
    for t in targets:
        plt.text(t[0] + 100, t[1] + 100, f'Target {targets.index(t) + 1}', font1)
    plt.text(UAVs[0].depot[0] - 100, UAVs[0].depot[1] - 200, 'Base', font2)
    plt.plot(uavs[5][0][0], uavs[5][0][1], 'r*', markerfacecolor='none', markersize=10, label='Base')
    plt.legend(loc='upper right', prop=font)

    plt.plot([-115], [30], 'bs', markerfacecolor='none')

    u = 0
    for uav in position:
        for p in range(1, len(uav)-1):
            distance[u] += np.linalg.norm(np.subtract(uav[p][:2], uav[p-1][:2]))
        u += 1
    objectives = mission_time + np.sum(distance) / np.sum(uavs[2])
    print(f"mission time: {mission_time} \n"
          f"total distance: {np.sum(distance)} \n"
          f"objectives: {objectives}"
          f"distance: {distance}")

    # fig.tight_layout()
    # plt.subplots_adjust(wspace=.3, hspace=.3)
    plt.show()
    # plt.savefig('C:/Users/user/Desktop/Monte_Carlo/Normal_T4U3_1.png')

    # import pymysql
    #
    # connect_db = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='ncku5895',
    #                              charset='utf8', db='Dynamic_TA_data')
    # with connect_db.cursor() as cursor:
    #     sql = f"""
    #     INSERT INTO DDGA_data (`mission time`, `total distance`, `objectives`) VALUES
    #     ({mission_time}, {np.sum(distance)}, {objectives})
    #     """
    #
    #     # 執行 SQL 指令
    #     cursor.execute(sql)
    #
    #     # 提交至 SQL
    #     connect_db.commit()
    #
    # # 關閉 SQL 連線
    # connect_db.close()
