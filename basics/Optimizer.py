import numpy as np
import numpy.linalg as la

import itertools
import time
import heapq
from copy import copy, deepcopy

from scipy.sparse import csc_matrix
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

import sklearn.cluster as skl_cluster


class CustomKeyHeap():
    def __init__(self, initial=None, key=lambda x: x):
        self.key = key
        self.index = 0
        if initial:
            self._data = [(key(item), i, item) for i, item in enumerate(initial)]
            self.index = len(self._data)
            heapq.heapify(self._data)
        else:
            self._data = []

    def push(self, item):
        heapq.heappush(self._data, (self.key(item), self.index, item))
        self.index += 1

    def pop(self):
        self.index -= 1
        return heapq.heappop(self._data)[2]
    
    def len(self):
        return self.index

def mvee(points, tol=0.01, max_steps=200):
    n_dots, n_dims = points.shape
    q = np.column_stack((points, np.ones(n_dots))).T
    err = tol + 1
    u = np.full(n_dots, 1 / n_dots)
    steps = 0
    while err > tol and steps < max_steps:
        steps += 1
        x = np.dot(np.dot(q, np.diag(u)), q.T)
        m = np.diag(np.dot(np.dot(q.T, la.pinv(x)), q))
        jdx = np.argmax(m)
        step_size = (m[jdx] - n_dims - 1) / ((n_dims + 1) * (m[jdx] - 1))
        new_u = (1-step_size) * u
        new_u[jdx] += step_size
        err = la.norm(new_u - u)
        u = new_u
    c = np.dot(u, points)
    a = la.pinv(np.dot(np.dot(points.T, np.diag(u)), points) - np.multiply.outer(c, c)) / n_dims
    return a, c

def sample_from_unit_circle(n_dims, n_dots, radius=1):
    vec = np.random.normal(0, 1, size=(n_dims, n_dots))
    vec /= la.norm(vec, axis=0)
    r = (np.random.random(n_dots) ** (1 / n_dims))
    vec *= r * radius
    return vec.T

def opt_sample_along_path(env, path, n_samples, neighborhood_radius, obs_collection):
    '''path.shape = (n_dots, n_dims)'''
    dists = []
    for dot in path:
        env.set_joints(dot)
        _, distData, _ = env.sim.checkDistance(env.robot_collection, obs_collection)
        dists.append(distData[-1])
    dists = np.array(dists)
    probs = 1 / (dists + 0.005) ** 2
    num_samples = np.floor(probs / np.sum(probs) * n_samples).astype(int)
    for i in range(len(path)):
        dot = path[i]
        test_dots = dot + sample_from_unit_circle(env.n_dims, num_samples[i], neighborhood_radius)
        for test_dot in test_dots:
            _ = env.traversable(test_dot)

def opt_check_path_not_in_ellipse(path, a, c, threshold=1):
    x = path - c
    return (np.einsum('ij,jk,ik->i', x, a, x) > threshold).all()

def opt_get_ellipse_obstacles(centers, path, init_n_class=10, min_in_cluster=10, max_clusters=100, threshold=1):
    ''' 
    - path.shape = (n_dots, n_dims)
    - min_in_cluster must be >= n_dims + 1
    '''
    max_n_clusters = len(centers) // min_in_cluster
    if max_n_clusters == 0:
        return []
    init_clusters = skl_cluster.AgglomerativeClustering(n_clusters=min(init_n_class, max_n_clusters)).fit_predict(centers)
    init_clusters = [centers[init_clusters == i] for i in range(init_n_class)]
    result_ellipses = []

    clusters_heap = CustomKeyHeap(init_clusters, key=lambda x: len(x))
    while clusters_heap.len() > 0 and len(result_ellipses) < max_clusters:
        cluster = clusters_heap.pop()
        if len(cluster) > min_in_cluster:
            a, c =  mvee(cluster)
            if opt_check_path_not_in_ellipse(path, a, c, threshold) == True:
                result_ellipses.append((a, c))
            else:
                new_clusters = skl_cluster.AgglomerativeClustering(n_clusters=2).fit_predict(cluster)
                clusters_heap.push(cluster[new_clusters == 0])
                clusters_heap.push(cluster[new_clusters == 1])
    return result_ellipses


class OptProblem():
    def __init__(self, n_dims, n_dots, start, goal, eq_matrices):
        self.n_dims = n_dims
        self.n_dots = n_dots
        self.start = start
        self.goal = goal
        self.n_variables = self.n_dims * self.n_dots
        self.eq_matrices = eq_matrices
        self.n_constr = len(eq_matrices)
    
    def f(self, x):
        coords_path = np.vstack([self.start, x.reshape((self.n_dots, self.n_dims)), self.goal])
        return (np.diff(coords_path, axis=0) ** 2).sum()
    
    def jacobian(self, x):
        coords_path = np.vstack([self.start, x.reshape((self.n_dots, self.n_dims)), self.goal])
        diffs = -np.diff(2 * np.diff(coords_path, axis=0), axis=0)
        return diffs.flatten()
    
    def cons_f(self, x):
        x_mat = x.reshape((self.n_dots, self.n_dims))
        result = [(x_mat[i, :] - c) @ a @ (x_mat[i, :] - c).T for i in range(len(x_mat)) for a, c in self.eq_matrices]
        return np.array(result)
    
    def cons_jacobian(self, x):
        result = np.zeros(((self.n_dots * self.n_constr, self.n_dots * self.n_dims)))
        for i in range(self.n_dots):
            for j in range(self.n_constr):
                index = i * self.n_constr + j
                x_cur = x[i*self.n_dims:i*self.n_dims+self.n_dims]
                a, c = self.eq_matrices[j]
                result[index, i*self.n_dims:i*self.n_dims+self.n_dims] = 2 * ((x_cur - c) @ a)
        return csc_matrix(np.array(result))

    def calculate_cost(self, result):
        coords_path = np.vstack([self.start, result.reshape((len(result) // self.n_dims, self.n_dims)), self.goal])
        return la.norm(np.diff(coords_path, axis=0), axis=1).sum()



class PathOptimizer():
    def __init__(self, env, delta, obs_collection):
        self.env = env
        self.delta = delta
        self.n_dims = self.env.n_dims
        self.obs_collection = obs_collection
    
    def calculate_cost(self, start, path, goal):
        coords_path = np.vstack([start, path.reshape((-1, self.n_dims)), goal])
        real_dist = self.env.calculate_distance(start, goal)
        return max(la.norm(np.diff(coords_path, axis=0), axis=1).sum(), real_dist)
    
    def get_path(self, nodes_list, goal_index):
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
        path = np.array(path[1:-1])
        return path.flatten()
    
    def upsize_path(self, path, delta):
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
        return result.flatten()
    
    def optimize(self, start, goal, x_0, n_samples, neighborhood_radius, upsize=False, iters=5):
        '''
        x_0.shape = (n_dots * n_dims,)
        '''
        iter_steps = []
        full_x_0 = np.vstack([start, x_0.reshape((-1, self.n_dims)), goal])
        if upsize == True:
            x_0 = self.upsize_path(full_x_0, self.delta)
        else:
            x_0 = x_0.flatten()
        path_len = len(x_0) // self.n_dims
        opt_sample_along_path(self.env, x_0.reshape((-1, self.n_dims)), n_samples, neighborhood_radius, self.obs_collection)
        total_centers = np.array(self.env.c_obs)
        ellipses = opt_get_ellipse_obstacles(total_centers, x_0.reshape((-1, self.n_dims)), threshold=1)
        needed_lengths = np.ones(path_len * len(ellipses)) * 1.05
        problem = OptProblem(self.n_dims, path_len, start, goal, ellipses)
        nonlinear_constraint = NonlinearConstraint(problem.cons_f, needed_lengths, np.inf, jac=problem.cons_jacobian)
        
        def save_step(k):
            iter_steps.append(k.reshape((-1, self.n_dims)))

        if len(ellipses) != 0:
            res = minimize(problem.f, x_0.flatten(), method='SLSQP', jac=problem.jacobian,
                        constraints=[nonlinear_constraint], 
                        options={'maxiter': iters}, callback=save_step)
        else:
            res = minimize(problem.f, x_0.flatten(), method='SLSQP', jac=problem.jacobian, 
                        options={'maxiter': iters}, callback=save_step)
        result = False
        x_1 = None
        x_1_cost = np.inf
        for i in range(len(iter_steps)):
            current_x_1 = iter_steps[-i-1]
            if self.env.path_correctness(np.flip(np.vstack([start, current_x_1, goal]), 0)) == True:
                x_1 = current_x_1
                result = True
                x_1_cost = self.calculate_cost(start, x_1, goal)
                break
        return result, x_1, x_1_cost