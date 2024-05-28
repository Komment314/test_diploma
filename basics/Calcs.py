import numpy as np
import numpy.linalg as la

UNIT_BALLS_VOLUMES = {1: 2,
                      2: np.pi,
                      3: (4 / 3) * np.pi,
                      4: (1 / 2) * np.pi  ** 2,
                      5: (8 / 15) * np.pi ** 2,
                      6: (1 / 6) * np.pi ** 3,
                      7: (16 / 105) * np.pi ** 3,
                      8: (1 / 24) * np.pi ** 4,
                      9: (32 / 945) * np.pi ** 4,
                      10: (1 / 120) * np.pi ** 5,
                      }

def rrt_star_informed_calculate_radius(n_dims, x_free, card_v, cut_radius):
    gamma_rrt = ((2 * (1 + 1 / n_dims)) ** (1 / n_dims)) * (x_free / UNIT_BALLS_VOLUMES[n_dims]) ** (1 / n_dims)
    radius = min(gamma_rrt * (np.log(card_v)/card_v) ** (1 / n_dims), cut_radius)
    return radius + 0.001

def make_goals_probs(goal_dists):
    distribution = 1 / goal_dists ** 1
    probs = distribution / distribution.sum()
    return probs

def get_default_path(nodes_list, goal_index):
    path = []
    last_node = nodes_list[goal_index]
    parent = last_node.parent
    while parent is not None:
        last_node_coords = last_node.get_joints()
        path.append(last_node_coords)
        last_node = parent
        parent = last_node.parent
    last_node_coords = last_node.get_joints()
    path.append(last_node_coords)
    path.reverse()
    return np.array(path)

def upsize_path(path, delta):
    '''path.shape = (n_dots, n_dims)'''
    n_dots, n_dims = path.shape
    path_new = path.reshape((n_dots, n_dims))
    diffs = np.diff(path_new, axis=0)

    result = []
    costs = la.norm(diffs, axis=1)
    for i in range(n_dots - 1):
        cost = costs[i]
        if cost > delta:
            add_count = int(cost // delta)
            to_add = diffs[i] / add_count
            for j in range(0, add_count + 1):
                result.append(path_new[i] + to_add * j)
        else:
            result.append(path_new[i])
    result.append(path_new[-1])

    result = np.array(result)
    return result