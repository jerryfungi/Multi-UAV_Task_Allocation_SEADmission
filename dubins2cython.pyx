import numpy as np
import time
# from libc.math cimport sin
from math import sin, pi


def ortho(vect2d):
    return np.array((-vect2d[1], vect2d[0]))


def dist(pt_a, pt_b):
    return ((pt_a[0] - pt_b[0]) ** 2 + (pt_a[1] - pt_b[1]) ** 2) ** .5


def all_options(start, end, radius):
    center_0_left = find_center(start, 'L', radius)
    center_0_right = find_center(start, 'R', radius)
    center_2_left = find_center(end, 'L', radius)
    center_2_right = find_center(end, 'R', radius)
    options = [lsl(start, end, center_0_left, center_2_left, radius),
               rsr(start, end, center_0_right, center_2_right, radius),
               rsl(start, end, center_0_right, center_2_left, radius),
               lsr(start, end, center_0_left, center_2_right, radius),
               rlr(start, end, center_0_right, center_2_right, radius),
               lrl(start, end, center_0_left, center_2_left, radius)]
    return options


def find_center(point, side, radius):
    assert side in 'LR'
    angle = point[2] + (np.pi / 2 if side == 'L' else -np.pi / 2)
    return np.array((point[0] + np.cos(angle) * radius,
                     point[1] + np.sin(angle) * radius))


def lsl(start, end, center_0, center_2, radius):
    straight_dist = dist(center_0, center_2)
    alpha = np.arctan2((center_2 - center_0)[1], (center_2 - center_0)[0])
    beta_2 = (end[2] - alpha) % (2 * np.pi)
    beta_0 = (alpha - start[2]) % (2 * np.pi)
    total_len = radius * (beta_2 + beta_0) + straight_dist
    return (total_len, (beta_0, beta_2, straight_dist), True)


def rsr(start, end, center_0, center_2):
    straight_dist = dist(center_0, center_2)
    alpha = np.arctan2((center_2 - center_0)[1], (center_2 - center_0)[0])
    beta_2 = (-end[2] + alpha) % (2 * np.pi)
    beta_0 = (-alpha + start[2]) % (2 * np.pi)
    total_len = radius * (beta_2 + beta_0) + straight_dist
    return (total_len, (-beta_0, -beta_2, straight_dist), True)


def rsl(start, end, center_0, center_2):
    median_point = (center_2 - center_0) / 2
    psia = np.arctan2(median_point[1], median_point[0])
    half_intercenter = np.linalg.norm(median_point)
    if half_intercenter < radius:
        return (float('inf'), (0, 0, 0), True)
    alpha = np.arccos(radius / half_intercenter)
    beta_0 = -(psia + alpha - start[2] - np.pi / 2) % (2 * np.pi)
    beta_2 = (np.pi + end[2] - np.pi / 2 - alpha - psia) % (2 * np.pi)
    straight_dist = 2 * (half_intercenter ** 2 - radius ** 2) ** .5
    total_len = radius * (beta_2 + beta_0) + straight_dist
    return (total_len, (-beta_0, beta_2, straight_dist), True)


def lsr(start, end, center_0, center_2):
    median_point = (center_2 - center_0) / 2
    psia = np.arctan2(median_point[1], median_point[0])
    half_intercenter = np.linalg.norm(median_point)
    if half_intercenter < radius:
        return (float('inf'), (0, 0, 0), True)
    alpha = np.arccos(radius / half_intercenter)
    beta_0 = (psia - alpha - start[2] + np.pi / 2) % (2 * np.pi)
    beta_2 = (.5 * np.pi - end[2] - alpha + psia) % (2 * np.pi)
    straight_dist = 2 * (half_intercenter ** 2 - radius ** 2) ** .5
    total_len = radius * (beta_2 + beta_0) + straight_dist
    return (total_len, (beta_0, -beta_2, straight_dist), True)


def lrl(start, end, center_0, center_2):
    dist_intercenter = dist(center_0, center_2)
    intercenter = (center_2 - center_0) / 2
    psia = np.arctan2(intercenter[1], intercenter[0])
    if 2 * radius < dist_intercenter > 4 * radius:
        return (float('inf'), (0, 0, 0), False)
    gamma = 2 * np.arcsin(dist_intercenter / (4 * radius))
    beta_0 = (psia - start[2] + np.pi / 2 + (np.pi - gamma) / 2) % (2 * np.pi)
    beta_1 = (-psia + np.pi / 2 + end[2] + (np.pi - gamma) / 2) % (2 * np.pi)
    total_len = (2 * np.pi - gamma + abs(beta_0) + abs(beta_1)) * radius
    return (total_len,
            (beta_0, beta_1, 2 * np.pi - gamma),
            False)


def rlr(start, end, center_0, center_2):
    dist_intercenter = dist(center_0, center_2)
    intercenter = (center_2 - center_0) / 2
    psia = np.arctan2(intercenter[1], intercenter[0])
    if 2 * radius < dist_intercenter > 4 * radius:
        return (float('inf'), (0, 0, 0), False)
    gamma = 2 * np.arcsin(dist_intercenter / (4 * radius))
    beta_0 = -((-psia + (start[2] + np.pi / 2) + (np.pi - gamma) / 2) % (2 * np.pi))
    beta_1 = -((psia + np.pi / 2 - end[2] + (np.pi - gamma) / 2) % (2 * np.pi))
    total_len = (2 * np.pi - gamma + abs(beta_0) + abs(beta_1)) * radius
    return (total_len,
            (beta_0, beta_1, 2 * np.pi - gamma),
            False)

def dubins_path(start, end):
    options = all_options(start, end)
    dubins_length, dubins_path, straight = min(options, key=lambda x: x[0])[:]
    return generate_points(start, end, dubins_path, straight), dubins_length


def dubins_length(start, end):
    options = all_options(start, end)
    dubins_length, dubins_path, straight = min(options, key=lambda x: x[0])[:]
    return dubins_length


def generate_points(start, end, dubins_path, straight):
    if straight:
        return generate_points_straight(start, end, dubins_path)
    return generate_points_curve(start, end, dubins_path)


def generate_points_straight(start, end, path):
    total = radius * (abs(path[1]) + abs(path[0])) + path[2]  # Path length
    center_0 = find_center(start, 'L' if path[0] > 0 else 'R')
    center_2 = find_center(end, 'L' if path[1] > 0 else 'R')

    # We first need to find the points where the straight segment starts
    if abs(path[0]) > 0:
        angle = start[2] + (abs(path[0]) - np.pi / 2) * np.sign(path[0])
        ini = center_0 + radius * np.array([np.cos(angle), np.sin(angle)])
    else:
        ini = np.array(start[:2])
    # We then identify its end
    if abs(path[1]) > 0:
        angle = end[2] + (-abs(path[1]) - np.pi / 2) * np.sign(path[1])
        fin = center_2 + radius * np.array([np.cos(angle), np.sin(angle)])
    else:
        fin = np.array(end[:2])
    dist_straight = dist(ini, fin)

    # We can now generate all the points with the desired precision
    points = []
    for x in np.arange(0, total, point_separation):
        if x < abs(path[0]) * radius:  # First turn
            points.append(circle_arc(start, path[0], center_0, x))
        elif x > total - abs(path[1]) * radius:  # Last turn
            points.append(circle_arc(end, path[1], center_2, x - total))
        else:  # Straight segment
            coeff = (x - abs(path[0]) * radius) / dist_straight
            points.append(coeff * fin + (1 - coeff) * ini)
    points.append(end[:2])
    return np.array(points)


def generate_points_curve(start, end, path):
    total = radius * (abs(path[1]) + abs(path[0]) + abs(path[2]))
    center_0 = find_center(start, 'L' if path[0] > 0 else 'R')
    center_2 = find_center(end, 'L' if path[1] > 0 else 'R')
    intercenter = dist(center_0, center_2)
    center_1 = (center_0 + center_2) / 2 + \
               np.sign(path[0]) * ortho((center_2 - center_0) / intercenter) \
               * (4 * radius ** 2 - (intercenter / 2) ** 2) ** .5
    psi_0 = np.arctan2((center_1 - center_0)[1],
                       (center_1 - center_0)[0]) - np.pi

    points = []
    for x in np.arange(0, total, point_separation):
        if x < abs(path[0]) * radius:  # First turn
            points.append(circle_arc(start, path[0], center_0, x))
        elif x > total - abs(path[1]) * radius:  # Last turn
            points.append(circle_arc(end, path[1], center_2, x - total))
        else:  # Middle Turn
            angle = psi_0 - np.sign(path[0]) * (x / radius - abs(path[0]))
            vect = np.array([np.cos(angle), np.sin(angle)])
            points.append(center_1 + radius * vect)
    points.append(end[:2])
    return np.array(points)


def circle_arc(reference, beta, center, x):
    angle = reference[2] + ((x / radius) - np.pi / 2) * np.sign(beta)
    vect = np.array([np.cos(angle), np.sin(angle)])
    return center + radius * vect
