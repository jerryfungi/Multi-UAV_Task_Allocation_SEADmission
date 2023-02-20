import numpy as np


def cs(start_point, start_heading, target_point, r):
    # Relaxed Dubins circle-straight --> start(0,0,0)
    cp = np.sqrt((target_point[0]-start_point[0])**2 + (target_point[1]-start_point[1]-r)**2)
    tc = r
    tp = np.sqrt(cp**2+tc**2)

    theta = np.arctan(tp/tc)
    gamma = np.arctan((r-target_point[1]+start_point[1])/(target_point[0]-start_point[0]))
    arc_path_theta = gamma - theta
    s = r * arc_path_theta
    distance = s + tp
    relative_heading = arc_path_theta if target_point[1]-start_point[1]>0 else -arc_path_theta
    return distance, start_heading+relative_heading

def cc(start_point, start_heading, target_point, r):
    # Relaxed Dubins circle-circle --> start(0,0,0)

