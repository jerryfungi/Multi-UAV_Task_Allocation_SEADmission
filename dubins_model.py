import math
import time

import numpy as np
from matplotlib import pyplot as plt
from math import sin, cos
from math import sqrt, pi
from numpy import arctan2
import dubins
from GA_SEAD_process import *
import scipy.linalg as la


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
        # position for calculate ga
        self.x_k = self.x0
        self.y_k = self.y0
        self.theta_k = self.theta0

        self.v = 0

class Car(object):

    '''
    Dubin's car
    '''

    def __init__(self, velocity, Rmin, initial_pos):
        self.velocity = velocity
        self.Rmin = Rmin
        self.omega_max = self.velocity / self.Rmin
        self.x0 = initial_pos[0]
        self.y0 = initial_pos[1]


def dubins_update(car, v, x, y, theta, us, utheta, dt):

    '''
    Returns a new state (xn, yn, thetan),
    given an initial state (x, y, theta) and control phi.
    Numerical integration is done at a time step of dt [sec].
    '''
    if utheta > 1:
        utheta = 1
    elif utheta < 1:
        utheta = -1


    # state rate
    dx     = v * cos(theta)
    dy     = v * sin(theta)
    dtheta = v / car.Rmin * utheta
    v      = v + us * dt


    # new state (forward Euler integration)
    xn     = x     + dt*dx
    yn     = y     + dt*dy
    thetan = theta + dt*dtheta

    if thetan < -np.pi:
        thetan += 2*np.pi
    elif thetan > np.pi:
        thetan -= 2*np.pi
    # thetan = (thetan - np.sign(thetan)*2*np.pi) % (2*np.pi)

    return xn, yn, thetan, v


def step(car, x, y, theta, u, dt):
    # state rate
    dx = car.velocity * cos(theta)
    dy = car.velocity * sin(theta)
    dtheta = car.omega_max * u

    # new state (forward Euler integration)
    xn = x + dt * dx
    yn = y + dt * dy
    thetan = theta + dt * dtheta

    if thetan < -np.pi:
        thetan += 2 * np.pi
    elif thetan > np.pi:
        thetan -= 2 * np.pi

    return xn, yn, thetan


def step_pid(v, headingRate, x, y, theta, desire_yaw_rate, desire_v, dt):

    # state rate
    dx = v * cos(theta)
    dy = v * sin(theta)

    # system controller (acceleration)
    u = 5 * (desire_yaw_rate - headingRate)
    us = 2 * (desire_v - v)

    # new state (forward Euler integration)
    xn = x + dt * dx
    yn = y + dt * dy
    thetan = theta + dt * headingRate
    vn = v + dt * us
    headingRaten = headingRate + dt * u

    if thetan < -np.pi:
        thetan += 2 * np.pi
    elif thetan > np.pi:
        thetan -= 2 * np.pi

    return xn, yn, thetan, vn, headingRaten




def angle_between(p1, p2):
    return arctan2((p2[1]-p1[1]), (p2[0]-p1[0]))


def distance_between_points(p1, p2):
    return sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))


def bang_bang_control(car, path, waypoint_radius):
    theta = 0
    u = 0
    t = 0
    xn = car.x0
    yn = car.y0
    list_for_u = [0]
    list_for_t = [0]
    # rad error constant
    c = 0.1

    actual_x = [car.x0]
    actual_y = [car.y0]

    for x in range(len(path)):
        # waypoint_radius = 5 if x < len(path)-1 else 0.5
        print('1')
        waypoint_radius = 25
        while distance_between_points([xn, yn], path[x]) > waypoint_radius:
            angle_between_two_points = angle_between((xn, yn), path[x])

            xn, yn, thetan = step(car, xn, yn, theta, u, 0.1)
            relative_angle = angle_between_two_points - thetan
            error_of_heading = relative_angle if abs(relative_angle) <= 2*pi - abs(relative_angle) \
                else -(relative_angle/abs(relative_angle))*(2*pi - abs(relative_angle))
            if c >= error_of_heading >= - c:
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
            t += 0.1
            list_for_u.append(u)
            list_for_t.append(t)

    controls_list, time_list = list_for_u, list_for_t
    return controls_list, time_list, actual_x, actual_y


def trajectory_tracking(car, path):
    theta = 0
    u = 0
    t = 0
    xn = car.x0
    yn = car.y0
    list_for_u = [0]
    list_for_t = [0]
    # rad error constant
    c = 0.1
    x, y = [p[0] for p in path], [p[1] for p in path]

    actual_x = [car.x0]
    actual_y = [car.y0]
    i = 5
    previous_time, path_time = time.time(), time.time()
    start = time.time()

    while math.hypot(path[-1][0]-xn, path[-1][1]-yn) >= 0.3 or i != len(path)-1:
        angle_between_two_points = angle_between((xn, yn), path[i])
        dt = time.time() - previous_time
        previous_time = time.time()
        if time.time()-path_time >= 0.18 and i != len(path)-1:
            i += 1
            path_time = time.time()
        if i == len(path)-1:
            print(time.time()-start)
        xn, yn, thetan = step(car, xn, yn, theta, u, dt)
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
        plt.clf()
        plt.plot(x, y, 'k-', markersize=1)
        plt.plot(car.x0, car.y0, 'ko')
        plt.plot(actual_x, actual_y, 'r--', markersize=1)
        plt.plot(path[i][0], path[i][1], 'o')
        plt.pause(1e-10)

    controls_list, time_list = list_for_u, list_for_t
    return controls_list, time_list, actual_x, actual_y


def path_following(car, path):
    theta = 45*np.pi/180
    u = 0
    t = 0
    xn = -5
    yn = car.y0
    list_for_u = [0]
    list_for_t = [0]
    # rad error constant
    c = 0.001
    recede_horizon = 1
    path_window = 50
    next_s = 0
    x, y = [p[0] for p in path], [p[1] for p in path]

    actual_x = [car.x0]
    actual_y = [car.y0]
    previous_time, path_time = 0, time.time()

    while math.hypot(path[-1][0]-xn, path[-1][1]-yn) >= 0.7:
        if time.time() - previous_time >= 0:
            future_point = np.array([xn+car.velocity*cos(theta)*recede_horizon, yn+car.velocity*sin(theta)*recede_horizon])
            world_record = 1e10
            desire_point = 0
            start = next_s
            for i in range(start, start+path_window):
                try:
                    a = np.array([path[i][0], path[i][1]])
                    b = np.array([path[i+1][0], path[i+1][1]])
                except IndexError:
                    a = np.array([path[-2][0], path[-2][1]])
                    b = np.array([path[-1][0], path[-1][1]])
                    i = -2
                va = future_point - a
                vb = b - a
                projection = np.dot(va, vb)/np.dot(vb, vb)*vb
                normal_point = a + projection
                # check the normal in line or not
                if max(path[i][0], path[i + 1][0]) > normal_point[0] > min(path[i][0], path[i + 1][0]) and \
                        max(path[i][1], path[i + 1][1]) > normal_point[1] > min(path[i][1], path[i + 1][1]):
                    normal_point = normal_point[:]
                else:
                    normal_point = b
                # update distance
                d = np.linalg.norm(va - (normal_point - a))
                if d < world_record:
                    world_record = d
                    desire_point = normal_point + 0*vb
                    next_s = i
            angle_between_two_points = angle_between((xn, yn), desire_point)
            dt = 0 if previous_time == 0 else time.time() - previous_time
            # dt = time.time() - previous_time
            # print(f'dt = {dt}')
            # dt = 0.01
            previous_time = time.time()
            xn, yn, thetan = step(car, xn, yn, theta, u, dt)
            relative_angle = angle_between_two_points - thetan
            error_of_heading = relative_angle if abs(relative_angle) <= np.pi \
                else (-relative_angle/abs(relative_angle))*(relative_angle + 2*np.pi)
            if error_of_heading < 0:
                u = -1
                theta = thetan
            elif error_of_heading > 0:
                u = 1
                theta = thetan
            else:
                u = 0
                theta = thetan

            actual_x.append(xn)
            actual_y.append(yn)
            t += dt
            list_for_u.append(u)
            list_for_t.append(t)
            plt.clf()
            plt.plot(x, y, 'k-', markersize=1)
            plt.plot(car.x0, car.y0, 'ko')
            plt.plot(actual_x, actual_y, 'r--', markersize=1)
            plt.plot(desire_point[0], desire_point[1], 'o')
            plt.plot(future_point[0], future_point[1], '*')
            # plt.plot([path[point][0], path[point+1][0]], [path[point][1], path[point+1][1]], 'r-')
            plt.plot([xn, future_point[0]], [yn, future_point[1]], 'g-')
            # plt.show()
            plt.pause(0.0001)

        controls_list, time_list = list_for_u, list_for_t
    return controls_list, time_list, actual_x, actual_y


def path_following_LQR(car, path):
    lqr_Q = np.eye(5)
    lqr_Q[0, 0] = 10
    lqr_R = np.eye(2)

    theta = car.theta0
    v = 0
    us, utheta = 0, 0
    pe, pth_e = 0, 0
    t = 0
    xn = 2
    yn = 2
    list_for_u = [0]
    list_for_t = [0]
    # rad error constant
    recede_horizon = 1
    path_window = 50
    next_s = 0
    x, y = [p[0] for p in path], [p[1] for p in path]

    actual_x = [4]
    actual_y = [2]
    previous_time, path_time = 0, time.time()

    while math.hypot(path[-1][0]-xn, path[-1][1]-yn) >= 1:
        if time.time() - previous_time >= .1:
            future_point = np.array([xn+car.velocity*cos(theta)*recede_horizon, yn+car.velocity*sin(theta)*recede_horizon])
            world_record = 1e5
            desire_point = 0
            start = next_s
            for i in range(start, start+path_window):
                try:
                    a = np.array([path[i][0], path[i][1]])
                    b = np.array([path[i+1][0], path[i+1][1]])
                except IndexError:
                    a = np.array([path[-2][0], path[-2][1]])
                    b = np.array([path[-1][0], path[-1][1]])
                    i = -2
                va = future_point - a
                vb = b - a
                projection = np.dot(va, vb)/np.dot(vb, vb)*vb
                normal_point = a + projection
                # check the normal in line or not
                if max(path[i][0], path[i + 1][0]) > normal_point[0] > min(path[i][0], path[i + 1][0]) and \
                        max(path[i][1], path[i + 1][1]) > normal_point[1] > min(path[i][1], path[i + 1][1]):
                    normal_point = normal_point[:]
                else:
                    normal_point = b
                # update distance
                d = np.linalg.norm(va - (normal_point - a))
                if d < world_record:
                    world_record = d
                    desire_point = normal_point + 0*vb
                    next_s = i
            angle_between_two_points = angle_between((xn, yn), desire_point)
            dt = .1 if previous_time == 0 else time.time() - previous_time
            previous_time = time.time()
            xn, yn, thetan, v = dubins_update(car, v, xn, yn, theta, us, utheta, dt)
            # print(car.v)
            relative_angle = angle_between_two_points - thetan
            error_of_heading = relative_angle if abs(relative_angle) <= np.pi \
                else (-relative_angle/abs(relative_angle))*(relative_angle + 2*np.pi)
            tv = car.velocity if math.hypot(path[-1][0]-xn, path[-1][1]-yn) >= 5 else 0.05
            utheta, us = lqr_speed_steering_control(car.Rmin, v, tv, world_record, -error_of_heading, pe, pth_e, lqr_Q, lqr_R, 0.1)
            print(utheta)
            pe, pth_e = world_record, error_of_heading
            theta = thetan

            actual_x.append(xn)
            actual_y.append(yn)
            t += dt
            list_for_u.append(utheta)
            list_for_t.append(t)
            plt.clf()
            plt.plot(x, y, 'k-', markersize=1)
            plt.plot(car.x0, car.y0, 'ko')
            plt.plot(actual_x, actual_y, 'ro', markersize=4)
            plt.plot(desire_point[0], desire_point[1], 'o')
            plt.plot(future_point[0], future_point[1], '*')
            # plt.plot([path[point][0], path[point+1][0]], [path[point][1], path[point+1][1]], 'r-')
            plt.plot([xn, future_point[0]], [yn, future_point[1]], 'g-')
            plt.title(f"speed {round(v, 3)} m/s")
            # plt.show()
            plt.pause(0.0001)

        controls_list, time_list = list_for_u, list_for_t
    return controls_list, time_list, actual_x, actual_y


def solve_dare(A, B, Q, R):
    """
    solve a discrete time_Algebraic Riccati equation (DARE)
    """
    x = Q
    x_next = Q
    max_iter = 150
    eps = 0.01

    for i in range(max_iter):
        x_next = A.T @ x @ A - A.T @ x @ B @ \
                 la.inv(R + B.T @ x @ B) @ B.T @ x @ A + Q
        if (abs(x_next - x)).max() < eps:
            break
        x = x_next

    return x_next


def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    # first, try to solve the ricatti equation
    X = solve_dare(A, B, Q, R)

    # compute the LQR gain
    K = la.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    eig_result = la.eig(A - B @ K)

    return K, X, eig_result[0]


def lqr_speed_steering_control(Rmin, v, tv, e, th_e, pe, pth_e, Q, R, dt):
    if pe is None:
        pe = e
        pth_e = th_e

    # A = [1.0, dt, 0.0, 0.0, 0.0
    #      0.0, 0.0, v, 0.0, 0.0]
    #      0.0, 0.0, 1.0, dt, 0.0]
    #      0.0, 0.0, 0.0, 0.0, 0.0]
    #      0.0, 0.0, 0.0, 0.0, 1.0]
    A = np.zeros((5, 5))
    A[0, 0] = 1.0
    A[0, 1] = dt
    A[1, 2] = v
    A[2, 2] = 1.0
    A[2, 3] = dt
    A[4, 4] = 1.0

    # B = [0.0, 0.0
    #     0.0, 0.0
    #     0.0, 0.0
    #     v/R, 0.0
    #     0.0, dt]
    B = np.zeros((5, 2))
    B[3, 0] = v / Rmin
    B[4, 1] = dt

    K, _, _ = dlqr(A, B, Q, R)

    # state vector
    # x = [e, dot_e, th_e, dot_th_e, delta_v]
    # e: lateral distance to the path
    # dot_e: derivative of e
    # th_e: angle difference to the path
    # dot_th_e: derivative of th_e
    # delta_v: difference between current speed and target speed
    x = np.zeros((5, 1))
    x[0, 0] = e
    x[1, 0] = (e - pe) / dt
    x[2, 0] = th_e
    x[3, 0] = (th_e - pth_e) / dt
    x[4, 0] = v - tv

    # input vector
    # u = [delta, accel]
    # delta: steering angle
    # accel: acceleration
    ustar = -K @ x

    # calc steering input
    delta = ustar[0, 0]

    # calc accel input
    accel = ustar[1, 0]

    return delta, accel


def path_following_velocity(car, path):
    theta = car.theta0
    us, utheta = 0, 0
    t = 0
    xn = car.x0
    yn = car.y0
    list_for_u = [0]
    list_for_t = [0]
    # rad error constant
    recede_horizon = 1
    path_window = 50
    next_s = 0
    x, y = [p[0] for p in path], [p[1] for p in path]

    actual_x = [car.x0]
    actual_y = [car.y0]
    previous_time, path_time = 0, time.time()

    while math.hypot(path[-1][0]-xn, path[-1][1]-yn) >= 1:
        if time.time() - previous_time >= .1:
            future_point = np.array([xn+car.velocity*cos(theta)*recede_horizon, yn+car.velocity*sin(theta)*recede_horizon])
            world_record = 1e10
            desire_point = 0
            start = next_s
            for i in range(start, start+path_window):
                try:
                    a = np.array([path[i][0], path[i][1]])
                    b = np.array([path[i+1][0], path[i+1][1]])
                except IndexError:
                    a = np.array([path[-2][0], path[-2][1]])
                    b = np.array([path[-1][0], path[-1][1]])
                    i = -2
                va = future_point - a
                vb = b - a
                projection = np.dot(va, vb)/np.dot(vb, vb)*vb
                normal_point = a + projection
                # check the normal in line or not
                if max(path[i][0], path[i + 1][0]) > normal_point[0] > min(path[i][0], path[i + 1][0]) and \
                        max(path[i][1], path[i + 1][1]) > normal_point[1] > min(path[i][1], path[i + 1][1]):
                    normal_point = normal_point[:]
                else:
                    normal_point = b
                # update distance
                d = np.linalg.norm(va - (normal_point - a))
                if d < world_record:
                    world_record = d
                    desire_point = normal_point + 0*vb
                    next_s = i
            dt = .1 if previous_time == 0 else time.time() - previous_time
            previous_time = time.time()
            command_v = desire_point - np.array([xn, yn]) + future_point - np.array([xn, yn])
            next_point = (np.array([xn, yn]) + command_v*dt)
            xn = next_point[0]
            yn = next_point[1]
            theta = np.arctan2(command_v[1], command_v[0])
            if theta < -np.pi:
                theta += 2 * np.pi
            elif theta > np.pi:
                theta -= 2 * np.pi

            actual_x.append(xn)
            actual_y.append(yn)
            t += dt
            list_for_t.append(t)
            plt.clf()
            plt.plot(x, y, 'k-', markersize=1)
            plt.plot(car.x0, car.y0, 'ko')
            plt.plot(actual_x, actual_y, 'ro', markersize=4)
            plt.plot(desire_point[0], desire_point[1], 'o')
            plt.plot(future_point[0], future_point[1], '*')
            # plt.plot([path[point][0], path[point+1][0]], [path[point][1], path[point+1][1]], 'r-')
            plt.plot([xn, future_point[0]], [yn, future_point[1]], 'g-')
            plt.pause(0.0001)

        controls_list, time_list = list_for_u, list_for_t
    return controls_list, time_list, actual_x, actual_y


def path_following_pid(car, path, kp, kd):
    theta = car.theta0
    u, us = 0, 0
    t = 0
    v, headingRate = 0, 0
    xn = car.x0
    yn = car.y0
    list_for_u = [0]
    list_for_t = [0]

    recede_horizon = 1
    path_window = 30
    next_s = 0
    x, y = [p[0] for p in path], [p[1] for p in path]
    e_previous, sum_error = 0, 0

    actual_x = [xn]
    actual_y = [yn]
    previous_time, path_time = 0, time.time()
    kd_ = 0
    end = np.array([path[-1][0], path[-1][1]])
    finish = False
    check = False

    # while math.hypot(path[-1][0]-xn, path[-1][1]-yn) >= 0.7:
    while True:
        # path = \
        # dubins.shortest_path([xn, yn, theta], car.depot, car.Rmin).sample_many(
        #     uav.velocity / 10)[0]
        if np.linalg.norm(end-np.array([xn, yn])) <= 1:
            finish = True
        # if time.time() - previous_time >= .1:
        if True:
            future_point = np.array([xn+car.velocity*cos(theta)*recede_horizon, yn+car.velocity*sin(theta)*recede_horizon])
            world_record = 1e10
            desire_point = 0
            start = next_s
            for i in range(start, start+path_window):
                try:
                    a = np.array([path[i][0], path[i][1]])
                    b = np.array([path[i+1][0], path[i+1][1]])
                except IndexError:
                    a = np.array([path[-2][0], path[-2][1]])
                    b = np.array([path[-1][0], path[-1][1]])
                    i = -2
                va = future_point - a
                vb = b - a
                projection = np.dot(va, vb)/np.dot(vb, vb)*vb
                normal_point = a + projection
                # check the normal in line or not
                if not max(path[i][0], path[i + 1][0]) > normal_point[0] > min(path[i][0], path[i + 1][0]) or \
                        not max(path[i][1], path[i + 1][1]) > normal_point[1] > min(path[i][1], path[i + 1][1]):
                    normal_point = b
                # update distance
                d = np.linalg.norm(va - (normal_point - a))
                if d < world_record:
                    world_record = d
                    desire_point = normal_point
                    if future_point[0] >= desire_point[0]:
                        d_optimal = d
                    else:
                        d_optimal = -d
                    proj_value = np.dot(vb, desire_point - np.array([xn, yn]))
                    next_s = i
            # print(np.linalg.norm(desire_point - np.array([xn, yn])))
            angle_between_two_points = angle_between((xn, yn), desire_point)
            # dt = 0 if previous_time == 0 else time.time() - previous_time
            # dt = time.time() - previous_time
            # print(f'dt = {dt}')
            dt = 0.1 if check else 0
            previous_time = time.time()

            relative_angle = angle_between_two_points - theta
            error_of_heading = relative_angle if abs(relative_angle) <= np.pi \
                else (-relative_angle / abs(relative_angle)) * (relative_angle + 2 * np.pi)
            error = error_of_heading + 0 * d_optimal
            sum_error += error
            u = kp * error + kd * (error_of_heading - e_previous) + 0 * sum_error # yaw_rate_command
            if u > car.velocity / car.Rmin:
                u = car.velocity / car.Rmin
            elif u < -car.velocity / car.Rmin:
                u = -car.velocity / car.Rmin
            e_previous = error
            # v_command = car.velocity if math.hypot(path[-1][0] - xn, path[-1][1] - yn) >= 5 else 0.1*math.hypot(path[-1][0] - xn, path[-1][1] - yn)
            # v_command = car.velocity if math.hypot(path[-1][0] - xn, path[-1][1] - yn) >= 5 else 0

            # v_command = 20 * proj_value
            # if v_command > car.velocity:
            #     v_command = car.velocity
            # elif v_command < 0:
            #     v_command = 0
            #     u = 0
            if proj_value < 0 and finish:
                v_command, u = 0, 0
            elif np.linalg.norm(end-np.array([xn, yn])) <= 5:
                v_command = 0.8*np.linalg.norm(end-np.array([xn, yn]))
            else:
                v_command = car.velocity
            # print(v_command)

            xn, yn, theta, v, headingRate = step_pid(v, headingRate, xn, yn, theta, u, v_command, dt)

            actual_x.append(xn)
            actual_y.append(yn)
            t += dt
            # list_for_u.append(u)
            # list_for_t.append(t)
            plt.clf()
            plt.plot([p[0] for p in path], [p[1] for p in path], 'k-', markersize=1, linewidth=1)
            # plt.plot(x, y, 'k-', markersize=1)
            plt.plot(car.x0, car.y0, 'ko')
            plt.plot(actual_x, actual_y, 'r--', markersize=1, linewidth=2)
            plt.plot(desire_point[0], desire_point[1], 'o')
            plt.plot(future_point[0], future_point[1], '*')
            plt.title(f"speed: {round(v, 3)} m/s, Control input: {round(headingRate/(car.velocity / car.Rmin), 3)}")
            # plt.plot([path[point][0], path[point+1][0]], [path[point][1], path[point+1][1]], 'r-')
            plt.plot([xn, future_point[0]], [yn, future_point[1]], 'g-')
            # plt.show()
            plt.axis("equal")
            if check:
                plt.pause(1e-100)
            else:
                plt.pause(3)
                check = True

        controls_list, time_list = list_for_u, list_for_t
    return controls_list, time_list, actual_x, actual_y


if __name__ == "__main__":
    # uav = Car(velocity=70, Rmin=200, initial_pos=[0, 0])
    uav = UAV(1, 1, 3, 10, [-6.878, 15.867, -200 * pi / 180], [0, 0, -np.pi / 2])
    # point = [[-6.878, 15.867, -200*pi/180], [-52.277, -9.575, 135*pi/180], [-77.314, 41.584, 25*pi/180], [-29.44, 37.207, 270*pi/180]]
    point = [[-6.878, 15.867, -200 * pi / 180], [-42.277, -9.575, 135 * pi / 180]]
    # point = [[-6.878, 15.867, -200 * pi / 180], [-52.277, -9.575, 135 * pi / 180], [-52.277, -9.575, 155 * pi / 180], [-52.277, -9.575, 115 * pi / 180], [-77.314, 41.584, 25 * pi / 180]]
    path = dubins.shortest_path(point[0], point[1], uav.Rmin).sample_many(1)[0]
    print(path)
    plt.plot([x[0] for x in path], [x[1] for x in path], 'o')
    plt.show()
    # path = [[0, 0], [200, 200]]
    # # _, _, x, y = bang_bang_control(uav, path, 50)
    # # plt.plot(x, y)
    # # plt.show()
    # path = dubins.shortest_path([1000, 300, -np.pi], [0, 0, -np.pi / 2], uav.Rmin).sample_many(uav.velocity/10)[0]
    # # p = dubins.path([0,0,0], [0,8,-np.pi], uav.Rmin, 1).sample_many(uav.velocity/10)[0]
    # # plt.plot([x[0] for x in p], [x[1] for x in p], 'k-')
    # # path, _ = dubins_path.sample_many(uav.velocity/10)
    # # print(path)
    # # plt.plot([x[0] for x in path], [x[1] for x in path])
    # # # plt.show()
    # uav01 = UAV(1, 1, 70, 200, [1000, 300, -np.pi], [0, 0, -np.pi / 2])
    # # # # route = np.array([[30, 5], [62, 36], [50, 65], [50, 65], [20, 50], [-10, 45], [0, 0]])
    # # # # route = np.array([[x[i], y[i]] for i in range(len(x))])
    # # # # route *= 2
    # # # # s = time.time()
    uav01 = UAV(1, 1, 3, 10, [-6.878, 15.867, -200*pi/180], [0, 0, -np.pi / 2])
    # point = [[-6.878, 15.867, -200*pi/180], [-52.277, -9.575, 135*pi/180], [-77.314, 41.584, 25*pi/180], [-29.44, 37.207, 270*pi/180]]
    point = [[-6.878, 15.867, -200 * pi / 180], [-42.277, -9.575, 135 * pi / 180], [-47.314, 21.584, 25 * pi / 180]]
    # point = [[-6.878, 15.867, -200 * pi / 180], [-52.277, -9.575, 135 * pi / 180], [-52.277, -9.575, 155 * pi / 180], [-52.277, -9.575, 115 * pi / 180], [-77.314, 41.584, 25 * pi / 180]]
    path = [point[0]]
    for i in range(len(point)-1):
        path.extend(dubins.shortest_path(point[i], point[i+1], uav01.Rmin).sample_many(uav01.velocity/10)[0][1:])
    dubins_path = dubins.shortest_path([-6.878, 15.867, -200*pi/180], [15,65,-np.pi/4], uav01.Rmin)
    print(path)
    # # dubins_path2 = dubins.shortest_path([30,32,np.pi], [30,32,np.pi-1e-2], uav01.Rmin)
    # # dubins_path3 = dubins.shortest_path([30,32,np.pi], [60,60,2*np.pi], uav01.Rmin)
    # print(dubins_path.path_length())
    # path, _ = dubins_path.sample_many(uav01.velocity/10)

    # path2, _ = dubins_path2.sample_many(uav01.velocity/5)
    # path3, _ = dubins_path3.sample_many(uav01.velocity/5)
    # path = path + path2[1:] + path3[1:]
    # print((dubins_path.path_length()+dubins_path2.path_length())/uav01.velocity)
    # print(len(path))
    xy_path = [a[0:-1] for a in path]
    path_l = len(path)
    # for i in range(1, path_l):
    #     plt.clf()
    #     dubins_path = dubins.shortest_path(path[i], [15, 65, -np.pi/4], uav01.Rmin)
    #     dpath, _ = dubins_path.sample_many(uav01.velocity / 10)
    #     plt.plot([x[0] for x in path], [x[1] for x in path])
    #     if math.hypot(path[i][0]-15, path[i][1]-65) <= 2*uav01.Rmin:
    #         plt.plot([x[0] for x in path[i:]], [x[1] for x in path[i:]], 'b')
    #         # print('lock', i)
    #     else:
    #         plt.plot([x[0] for x in dpath], [x[1] for x in dpath], 'b')
    #     plt.pause(0.001)
        # print(dubins_path.path_length(), [path[i], [15, 65, np.pi]], math.hypot(path[i][0]-15, path[i][1]-65))
    # controls_u, times_t, actual_x, actual_y = path_following_LQR(uav01, xy_path)

    controls_u, times_t, actual_x, actual_y = path_following_pid(uav01, path, 5, 20)
    # controls_u, times_t, actual_x, actual_y = path_following(uav01, xy_path)
    # # controls_u, times_t, actual_x, actual_y = solution(uav01, route, 5)
    # # # print((time.time()-s)*100)
    # # print(controls_u)
    # # print(actual_x)
    #
    # # plt.subplot(1, 2, 1)
    # # plt.plot(actual_x, actual_y, '-')
    # # plt.plot(route[:, 0], route[:, 1], 'bo')
    # # plt.plot([0, 20*2, 30*2], [0, 20*2, 45*2], 'o')
    # # plt.plot(uav01.x0, uav01.y0, 'ro')
    # # plt.title(f'spend time: {round(times_t[-1], 5)} sec')
    # # plt.subplot(1, 2, 2)
    # # plt.plot(times_t, controls_u)
    # # plt.show()

    # targets_sites = [[0, 400], [200, 800]]
    # terminal = [[-500, 1200, np.pi / 2]]
    # uavs = [[3], [2], [25], [75],
    #             [[-500, 0, np.pi / 2]], terminal,
    #             [], [], [], []]
    # ga = GA_SEAD(targets_sites)
    # solution, fitness_value, ga_population = ga.run_GA_time_period_version(3, uavs)
    # uav_num = len(terminal)
    # UAVs = [UAV(uavs[0][n], uavs[1][n], uavs[2][n], uavs[3][n], uavs[4][n], terminal[n]) for n in range(uav_num)]
    # uav01 = UAVs[0]
    #
    # def generate_path(chromosome, uav):
    #     path_route, task_sequence_state, task_check_list = [], [], []
    #     if chromosome:
    #         for a in range(len(chromosome[0])):
    #             if chromosome[3][a] == uav.id:
    #                 assign_target = chromosome[1][a]
    #                 assign_heading = chromosome[4][a] * 10
    #                 task_sequence_state.append([targets_sites[assign_target - 1][0],
    #                                             targets_sites[assign_target - 1][1], assign_heading,
    #                                             assign_target, chromosome[2][a]])
    #         if task_sequence_state:
    #             for state in task_sequence_state:
    #                 state[2] *= np.pi / 180
    #             dubins_path = dubins.shortest_path([uav.x_k, uav.y_k, uav.theta_k], task_sequence_state[0][:3], uav.Rmin)
    #             path_route.extend(dubins_path.sample_many(.1)[0])
    #             for a in range(len(task_sequence_state) - 1):
    #                 sp = task_sequence_state[a][:3]
    #                 gp = task_sequence_state[a + 1][:3] if task_sequence_state[a][:3] != task_sequence_state[a + 1][:3] else \
    #                     [task_sequence_state[a + 1][0], task_sequence_state[a + 1][1], task_sequence_state[a + 1][2] - 1e-3]
    #                 dubins_path = dubins.shortest_path(sp, gp, uav.Rmin)
    #                 path_route.extend(dubins_path.sample_many(.1)[0][1:])
    #             dubins_path = dubins.shortest_path(task_sequence_state[-1][:3], uav.depot, uav.Rmin)
    #             path_route.extend(dubins_path.sample_many(.1)[0][1:])
    #         else:
    #             dubins_path = dubins.shortest_path([uav.x_k, uav.y_k, uav.theta_k], uav.depot,
    #                                                uav.Rmin)
    #             path_route.extend(dubins_path.sample_many(.1)[0])
    #     return path_route, task_sequence_state
    # a, b = generate_path(solution, uav01)
    # controls_u, times_t, actual_x, actual_y = path_following(uav01, a)
    # plt.subplot(1, 2, 1)
    # plt.plot([p[0] for p in xy_path], [p[1] for p in xy_path], 'k-', markersize=1)
    # plt.plot(uav01.x0, uav01.y0, 'ko')
    # plt.plot(actual_x, actual_y, 'r--', markersize=1)
    # plt.title(f'spend time: {round(times_t[-1], 5)} sec')
    # plt.subplot(1, 2, 2)
    # plt.plot(times_t, controls_u)
    # plt.show()

    # tu = Car(velocity=70, Rmin=200, initial_pos=[0, 0])
    # dubins_path1 = dubins.shortest_path([100,0,np.pi/2], [-1370,4800,20*np.pi/180], tu.Rmin)
    # dubins_path = dubins.shortest_path([-1370,4800,20*np.pi/180], [100,0,np.pi/2], tu.Rmin)
    # print(dubins_path.path_length()/tu.velocity, dubins_path1.path_length()/tu.velocity)
    # a, b = dubins_path1.sample_many(.1)[0], dubins_path.sample_many(.1)[0]
    # plt.plot([s[0] for s in a], [s[1] for s in a])
    # plt.plot([s[0] for s in b], [s[1] for s in b])
    # plt.show()
    # dubins_path2 = dubins.shortest_path([0,0,np.pi/2], [-1370,4800,190*np.pi/180], tu.Rmin)
    # dubins_path3 = dubins.shortest_path([-1370,4800,190*np.pi/180], [-3700,2520,230*np.pi/180], tu.Rmin)
    # dubins_path4 = dubins.shortest_path([-3700,2520,230*np.pi/180], [-3700,2520,20*np.pi/180], tu.Rmin)
    # print((dubins_path2.path_length()+dubins_path3.path_length()+dubins_path4.path_length())/tu.velocity)
    # tu = Car(velocity=80, Rmin=250, initial_pos=[0, 0])
    # dubins_path2 = dubins.shortest_path([50,0,np.pi/2], [3900,2400,50*np.pi/180], tu.Rmin)
    # dubins_path3 = dubins.shortest_path([3900,2400,50*np.pi/180], [3900,2400,100*np.pi/180], tu.Rmin)
    # dubins_path4 = dubins.shortest_path([3900,2400,100*np.pi/180], [3900,2400,140*np.pi/180], tu.Rmin)
    # dubins_path5 = dubins.shortest_path([3900,2400,140*np.pi/180], [-1370,4800,170*np.pi/180], tu.Rmin)
    # print((dubins_path2.path_length()+dubins_path3.path_length()+dubins_path4.path_length()+dubins_path5.path_length())/tu.velocity)
