import time

import numpy as np
import matplotlib.pyplot as plt


def twopify(alpha):
    return alpha - np.pi * 2 * np.floor(alpha / (np.pi * 2))


def pify(alpha):
    v = np.fmod(alpha, 2*np.pi)
    if v < - np.pi:
        v += 2 * np.pi
    else:
        v -= 2 * np.pi
    return v


class DubinsPath(object):
    def __init__(self, t=0, p=1e10, q=0, type=None):
        self.t = t
        self.p = p
        self.q = q
        self.length_ = [t, p, q]
        self.type = type

        self.controls = None

    def length(self):
        return self.t + self.p + self.q


class DubinsControl(object):
    def __init__(self):
        self.delta_s = 0.0
        self.kappa = 0.0


class Dubins(object):
    def __init__(self):
        self.constant = {
            "dubins_zero": -1e-9,
            "dubins_eps": 1e-6
        }

    def dubinsLSL(self, d, alpha, beta):
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        tmp = 2 + d * d - 2 * (ca * cb + sa * sb - d * (sa - sb))
        if (tmp >= self.constant["dubins_zero"]):
            theta = np.arctan2(cb - ca, d + sa - sb)
            t = twopify(-alpha + theta)
            p = np.sqrt(np.amax([tmp, 0]))
            q = twopify(beta - theta)

            return DubinsPath(t, p, q, ["L", "S", "L"])

        else:
            return None

    def dubinsLRL(self, d, alpha, beta):
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        tmp = .125 * (6. - d * d + 2. * (ca * cb + sa * sb - d * (sa - sb)))
        if (np.abs(tmp) < 1.):
            p = 2 * np.pi - np.arccos(tmp)
            theta = np.arctan2(-ca + cb, d + sa - sb)
            t = twopify(-alpha + theta + .5 * p)
            q = twopify(beta - alpha - t + p)

            return DubinsPath(t, p, q, ["L", "R", "L"])

        else:
            return None

    def dubinsLSR(self, d, alpha, beta):
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        tmp = -2 + d * d + 2 * (ca * cb + sa * sb + d * (sa + sb))
        if (tmp >= self.constant["dubins_zero"]):
            theta = np.arctan2(-cb - ca, d + sa + sb)
            p = np.sqrt(np.amax([tmp, 0]))
            t = twopify(-alpha + theta - np.arctan2(-2,p))

            q = twopify(-beta + theta - np.arctan2(-2,p))

            return DubinsPath(t, p, q, ["L", "S", "R"])

        else:
            return None

    def dubinsRSR(self, d, alpha, beta):
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        tmp = +2 + d * d - 2 * (ca * cb + sa * sb - d * (sb - sa))
        if (tmp >= self.constant["dubins_zero"]):
            theta = np.arctan2(-cb + ca, d - sa + sb)
            p = np.sqrt(np.amax([tmp, 0]))
            t = twopify(alpha - theta)

            q = twopify(-beta + theta)

            return DubinsPath(t, p, q, ["R", "S", "R"])

        else:
            return None

    def dubinsRSL(self, d, alpha, beta):
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        tmp = -2 + d * d + 2 * (ca * cb + sa * sb - d * (sb + sa))
        if (tmp >= self.constant["dubins_zero"]):
            theta = np.arctan2(cb + ca, d - sa - sb)
            p = np.sqrt(np.amax([tmp, 0]))
            t = twopify(alpha - theta + np.arctan2(2,p))
            q = twopify(+beta - theta + np.arctan2(2,p))

            return DubinsPath(t, p, q, ["R", "S", "L"])

        else:
            return None

    def dubinsRLR(self, d, alpha, beta):
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        tmp = .125 * (6. - d * d + 2. * (ca * cb + sa * sb + d * (sa - sb)))
        if (np.abs(tmp) < 1.):
            p = 2*np.pi - np.arccos(tmp)
            theta = np.arctan2(ca - cb, d - sa + sb)
            t = twopify(alpha - theta + .5 * p)
            q = twopify(-beta + alpha - t + p)

            return DubinsPath(t, p, q, ["R", "L", "R"])

        else:
            return None


    def get_best_dubins_path(self, d, alpha, beta):
        dubins_functions = [
            self.dubinsLSL, self.dubinsLRL, self.dubinsRLR, self.dubinsLSR, self.dubinsRSR, self.dubinsRSL
        ]

        min_length = 1e10
        path = None
        for dubins_function in dubins_functions:
            tmp_path = dubins_function(d, alpha, beta)
            if tmp_path is not None:
                if (tmp_path.length() < min_length):
                    min_length = tmp_path.length()
                    path = tmp_path
        return path

    def plan(self, state1, state2, kappa):
        dx = state2[0] - state1[0]
        dy = state2[1] - state1[1]
        th = np.arctan2(dy, dx)

        d = np.hypot(dx, dy) * kappa
        alpha = twopify(state1[2] - th)
        beta = twopify(state2[2] - th)

        dubins_path = self.get_best_dubins_path(d, alpha, beta)
        controls = self.dubins_path_to_controls(dubins_path, kappa)
        cartesian_path = self.controls_to_cartesian_path(controls, state1)

        return cartesian_path, controls, dubins_path

    def path_length(self, state1, state2, kappa):
        dx = state2[0] - state1[0]
        dy = state2[1] - state1[1]
        th = np.arctan2(dy, dx)

        d = np.hypot(dx, dy) * kappa
        alpha = twopify(state1[2] - th)
        beta = twopify(state2[2] - th)
        return self.get_best_dubins_path(d, alpha, beta).length()

    def dubins_path_to_controls(self, dubins_path, kappa):
        controls = []
        kappa_inv = 1.0/kappa

        if dubins_path is not None:
            for i in range(3):
                control = DubinsControl()
                type = dubins_path.type[i]
                length = dubins_path.length_[i]
                delta_s = kappa_inv * length

                control.delta_s = delta_s
                if (type == "L"):
                    control.kappa = kappa

                if (type == "S"):
                    control.kappa = 0

                if (type == "R"):
                    control.kappa = -kappa

                controls.append(control)
            return controls

        else:
            return None

    def controls_to_cartesian_path(self, controls, state1, discretization=0.1):
        if controls is None:
            return None

        x, y, yaw = state1
        xs, ys, yaws = [], [], []

        for control in controls:
            delta_s = control.delta_s
            abs_delta_s = np.abs(delta_s)
            kappa = control.kappa

            s_seg = 0
            integration_step = 0.0
            for j in range(int(np.ceil(abs_delta_s / discretization))):
                s_seg += discretization
                if (s_seg > abs_delta_s):
                    integration_step = discretization - (s_seg - abs_delta_s)
                    s_seg = abs_delta_s
                else:
                    integration_step = discretization

                if np.abs(kappa) > 0.0001:
                    x += 1/kappa * (-np.sin(yaw) + np.sin(yaw + integration_step * kappa))
                    y += 1/kappa * (np.cos(yaw) - np.cos(yaw + integration_step * kappa))
                    yaw = pify(yaw + integration_step * kappa)
                else:
                    x += integration_step * np.cos(yaw)
                    y += integration_step * np.sin(yaw)

                xs.append(x)
                ys.append(y)
                yaws.append(yaw)

        return xs, ys, yaws


if __name__ == '__main__':


    kappa_ = 1./10
    dubins = Dubins()

    # dubins.dubins(d=10.0, alpha=0.1, beta=0.1)
    sx, sy = np.random.uniform(-5, 0, size=2)

    stheta = np.random.uniform(0, np.pi * 2)

    start_state = [sx, sy, stheta]
    print(start_state)
    start_state = [0,0,0]

    gx, gy = np.random.uniform(5, 15, size=2)
    gtheta = np.random.uniform(0, np.pi * 2)

    goal_state = [gx, gy, gtheta]
    goal_state = [10, 10, 2]

    print(goal_state)
    t = time.time()
    cartesian_path, controls, dubins_path = dubins.plan(start_state, goal_state, kappa_)
    path_x, path_y, path_yaw = cartesian_path
    plt.plot(sx, sy, 'ro')
    plt.title("path length: %.2f, type: %s, %s, %s" % (dubins_path.length(),
                                                       dubins_path.type[0],
                                                       dubins_path.type[1],
                                                       dubins_path.type[2]))
    l = 2.0
    plt.plot([sx, sx + l * np.cos(stheta)], [sy, sy + l * np.sin(stheta)], 'r-')
    plt.plot(gx, gy, 'rx')
    plt.plot([gx, gx + l * np.cos(gtheta)], [gy, gy + l * np.sin(gtheta)], 'r-')
    plt.plot(path_x, path_y, 'b-')
    plt.axis("equal")
    plt.show()