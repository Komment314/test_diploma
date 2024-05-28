import time
import numpy as np


from Node import Node
from Partition import PartitionCells
from Calcs import rrt_star_informed_calculate_radius, UNIT_BALLS_VOLUMES
from Optimizer import PathOptimizer

def create_opt_informed_rrt_star(env, start, goals, delta, cut_radius, obs_collection, max_iters=1000, goal_bias_prob=0.1):
    begin_time = time.time()

    added_nodes = [Node(0, start)]
    iterations = 0
    num_nodes = 1

    iters_without_opt = 0
    real_c_best = np.inf
    c_best_for_sampling = np.inf
    best_opt_path = None
    used_radius = rrt_star_informed_calculate_radius(len(env.j_ranges), np.prod(env.j_ranges), 2, cut_radius)

    near_partition = PartitionCells(env.n_dims, used_radius)
    near_partition.put(0, added_nodes[0].get_joints())

    optimizer = PathOptimizer(env, delta * 2, obs_collection)

    goal = None
    goal_bias_probs = np.random.random(max_iters)
    goal_found = False
    goal_index = None

    for iterations in range(1, max_iters + 1):
        is_goal = False
        iters_without_opt += 1
        # Goal biasing
        # If goal is found then no need to goal bias
        if goal_found == False:
            n_goals = len(goals)
            if goal_bias_probs[iterations - 1] < goal_bias_prob:
                new_coords = goals[np.random.randint(n_goals)]
                is_goal = True
            else:
                new_coords = env.sample_random_config()
        else:
            # print(f'direct_sampling: \nstart = {start}, \ngoal = {goal}, \ndist = {env.calculate_distance(start, goal)}, \nc_best = {c_best_for_sampling}')
            new_coords = env.sample_from_start_to_goal_ellipse(start, goal, c_best_for_sampling)
            
        # Nearest node in tree
        nearest_node = added_nodes[0]
        nearest_node_dist = env.calculate_distance(nearest_node.get_joints(), new_coords)

        neighbors_indeces = near_partition.get_neighbors_id(new_coords)
        
        if len(neighbors_indeces) == 0:
            for selected_node in added_nodes:
                selected_node_dist = env.calculate_distance(selected_node.get_joints(), new_coords)
                if selected_node_dist < nearest_node_dist:
                    nearest_node = selected_node
                    nearest_node_dist = selected_node_dist
        else:
            for neighbor_index in neighbors_indeces:
                selected_node = added_nodes[neighbor_index]
                selected_node_dist = env.calculate_distance(selected_node.get_joints(), new_coords)
                if selected_node_dist < nearest_node_dist:
                    nearest_node = selected_node
                    nearest_node_dist = selected_node_dist

        # Steering
        nearest_node_coords = nearest_node.get_joints()
        diff = new_coords - nearest_node_coords
        norm = env.calculate_distance(new_coords, nearest_node_coords)
        if norm > cut_radius:
            is_goal = False
            new_coords = nearest_node_coords + diff / norm * cut_radius
            new_coords = new_coords

        # Checking connection between coords
        connection, last_coords_in_path = env.config_connectivity(nearest_node.get_joints(), new_coords, delta)

        # If coords are reachable then add new node
        if connection == True:
            # See below
            # Finding indexes of neighbor nodes of {new_coords} in tree
            new_coords_neighbors_indexes = near_partition.get_neighbors_id(new_coords)

            newest_coords_neighbors_indeces = []
            for neighbor_index in new_coords_neighbors_indexes:
                    selected_node = added_nodes[neighbor_index]
                    if env.calculate_distance(selected_node.get_joints(), new_coords) < used_radius:
                        newest_coords_neighbors_indeces.append(neighbor_index)

            # First RRT* change
            # If node in {new_coords} neighbors has cost less than {nearest_node} then change {nearest_node} to this node and repeat
            if len(newest_coords_neighbors_indeces) != 0:
                for neighbor_index in newest_coords_neighbors_indeces:
                    new_coords_neighbor = added_nodes[neighbor_index]
                    if new_coords_neighbor.get_cost() < nearest_node.get_cost() and env.config_connectivity(new_coords_neighbor.get_joints(), new_coords, delta)[0] == True:
                        nearest_node = new_coords_neighbor
            new_node = Node(num_nodes, last_coords_in_path, parent=nearest_node, \
                                             cost=env.calculate_distance(new_coords, nearest_node.get_joints()))
            added_nodes.append(new_node)
            near_partition.put(num_nodes, new_node.get_joints())
            num_nodes += 1
            
            if goal_found == False:
                current_radius = rrt_star_informed_calculate_radius(len(env.j_ranges), np.prod(env.j_ranges), num_nodes, cut_radius)
            else:
                real_c_best = added_nodes[goal_index].get_cost()
                ellipse_volume = UNIT_BALLS_VOLUMES[env.n_dims] * (c_best_for_sampling / 2) * ((np.sqrt(c_best_for_sampling ** 2 - c_min ** 2) / 2) ** (env.n_dims - 1))
                current_radius = rrt_star_informed_calculate_radius(len(env.j_ranges), ellipse_volume, num_nodes, cut_radius)
            if used_radius / current_radius > 1.05:
                used_radius = current_radius
                near_partition.change_radius(used_radius)
                # print(f'Iteration {iterations}, current radius = {used_radius}, real_c_best = {real_c_best}')

            # If added node is goal then success but we need to improve path so algirithm doesn't stop
            if is_goal == True:
                print('Success!')
                goal_found = True
                goal_index = len(added_nodes) - 1
                goal = last_coords_in_path
                c_min = env.calculate_distance(start, goal)
                real_c_best = added_nodes[goal_index].get_cost()
                c_best_for_sampling = real_c_best
                end_time = time.time()
                iters_without_opt = np.inf
                print(f'Goal was found in: {round(end_time - begin_time, 3)}s.')
                print(f'Nodes created: {goal_index + 1} / {iterations} iterations')
                print(f'Initial cost = {round(new_node.get_cost(), 3)}')
        
            # Second RRT* change
            # If dist({new_node}, {new_node_neighbor}) is less than {new_node_neighbor} cost then we can rebuild {new_node_neighbor} parent
            if len(newest_coords_neighbors_indeces) != 0:
                for neighbor_index in newest_coords_neighbors_indeces:
                    dist = env.calculate_distance(new_node.get_joints(), added_nodes[neighbor_index].get_joints())
                    if new_node.get_cost() + dist < added_nodes[neighbor_index].get_cost() and \
                        env.config_connectivity(added_nodes[neighbor_index].get_joints(), new_node.get_joints(), delta)[0] == True:
                        added_nodes[neighbor_index].parent = new_node
                        added_nodes[neighbor_index].cost = dist

        # Optimization
        if goal_found == True and iters_without_opt >= 200:
            result, x_1, x_1_cost = optimizer.optimize(start, goal, optimizer.get_path(added_nodes, goal_index), 1000, delta * 10, True, 20)
            if result == True: 
                if x_1_cost < c_best_for_sampling:
                    best_opt_path = x_1
                    c_best_for_sampling = min(x_1_cost, real_c_best)
            iters_without_opt = 0
    # If goal is not found in {max_iters} iterations then return Failure :(
    if goal_found == True:
        print(f'Result pure cost = {round(added_nodes[goal_index].get_cost(), 3)}')
        print(f'c_best_for_sampling = {round(c_best_for_sampling, 3)}')
        return  added_nodes, iterations, True, goal_index, best_opt_path, c_best_for_sampling
    else:
        print('Failure!')
        return added_nodes, iterations, False, goal_index, best_opt_path, c_best_for_sampling