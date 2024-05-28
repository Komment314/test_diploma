from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import numpy as np
import numpy.linalg as la
from copy import copy

class SceneUR10():
    def __init__(self, sim, world_collection, robot_collection, obs_collection,
                 sim_base, sim_tip, sim_target, sim_ik, ik_env, ik_group, 
                 n_dims, joints, j_init, delta, j_limits=None):
        self.client = RemoteAPIClient()
        self.sim = sim
        
        self.world_collection = world_collection
        self.robot_collection = robot_collection
        self.obs_collection = obs_collection

        self.sim_base = sim_base
        self.sim_tip = sim_tip
        self.sim_target = sim_target
        self.target_angles = self.sim.getObjectOrientation(self.sim_target)

        self.sim_ik = sim_ik
        self.ik_env = ik_env
        self.ik_group = ik_group

        self.n_dims = n_dims
        self.joints = joints
        self.j_init = j_init
        self.delta = delta

        if j_limits is None:
            self.j_limits = []
            for j in self.joints:
                _, interval = self.sim.getJointInterval(j)
                self.j_limits.append([interval[0], interval[0] + interval[1]])
            self.j_limits = np.array(self.j_limits)
        else:
            self.j_limits = j_limits

        self.j_limits_low = self.j_limits[:, 0].flatten()
        self.j_ranges = np.diff(self.j_limits, axis=1).flatten()

    def get_joints(self):
        joints = [self.sim.getJointPosition(j) for j in self.joints]
        return np.array(joints)

    def set_joints(self, joint_values=None):
        if joint_values is None:
            for i in range(len(self.joints)):
                self.sim.setJointPosition(self.joints[i], self.j_init[i])
        else:
            for i in range(len(self.joints)):
                self.sim.setJointPosition(self.joints[i], joint_values[i])

    def set_target(self, coords: list):
        self.sim.setObjectPosition(self.sim_target, list(coords))

    def get_ik(self):
        # res = 1 means ik is solved, otherwise res = 2
        res = 2
        while res != 1:
            self.sample_random_config()
            self.sim.setObjectOrientation(self.sim_target, self.target_angles[:2] + [float(2 * np.random.random() * np.pi - np.pi)])
            res, *_ = self.sim_ik.handleGroup(self.ik_env, self.ik_group, {'syncWorlds': True})
            if self.traversable(self.get_joints()) == False:
                res = 2
        res_joints = self.get_joints()
        for i in range(len(res_joints)):
            if res_joints[i] < -np.pi:
                res_joints[i] += 2 * np.pi
            elif res_joints[i] > np.pi:
                res_joints[i] -= 2 * np.pi
        res_joints = np.array(res_joints)
        res_joints[-1] = 0
        self.sim.setObjectOrientation(self.sim_target, self.target_angles)
        return np.array(res_joints)
    
    def traversable(self, joint_values):
        joint_values = joint_values.tolist()
        self.set_joints(joint_values)
        col_world, _ = self.sim.checkCollision(self.robot_collection, self.world_collection)
        col_self, _ = self.sim.checkCollision(self.robot_collection, self.robot_collection)
        return (col_world == 0) and (col_self == 0)
    
    def sample_random_config(self):
        joint_values = np.multiply(np.random.random(self.n_dims), self.j_ranges) + self.j_limits_low
        while not self.traversable(joint_values):
            joint_values = np.multiply(np.random.random(self.n_dims), self.j_ranges) + self.j_limits_low
        return np.array(joint_values)

    def calculate_distance(self, config1, config2):
        return la.norm((config2 - config1))

    def config_connectivity(self, config_begin, config_end, delta=None):
        if self.traversable(config_begin) and self.traversable(config_end):
            if delta is None:
                delta = self.delta
            diff = config_end - config_begin
            norm = la.norm(diff)
            if norm <= delta:
                return True, config_end
            else:
                num_moves = int(np.ceil(norm / delta)) - 1

                current = config_begin
                delta_move = diff / num_moves
                for _ in range(num_moves):
                    new = current + delta_move
                    if self.traversable(new):
                        current = new
                    else:
                        return False, current
                return True, current
        else:
            return False, config_begin
        
    def path_correctness(self, path):
        for i in range(len(path) - 1):
            current_connectivity = self.config_connectivity(path[i], path[i + 1])[0]
            if current_connectivity == False:
                return False
        return True

    def vis_path(self, path, sleep_coef=1):
        self.set_joints(path[0])
        time.sleep(0.5)
        self.sim.startSimulation()
        self.set_joints(path[0])
        for i in range(1, len(path)):
            begin_time = time.time()
            total_time_to_sleep = self.calculate_distance(path[i], path[i-1]) * sleep_coef
            joints = path[i]
            self.set_joints(joints)
            end_time = time.time()
            time_to_sleep = total_time_to_sleep - (end_time - begin_time)
            time.sleep(max(0, time_to_sleep))
        time.sleep(1)
        self.sim.stopSimulation()
    
    def get_rotate_matrix(self, v):
        x_0 = np.array([[la.norm(v)] + [0.0] * (self.n_dims - 1)])
        m_rot = np.eye(self.n_dims)
        v = v.reshape((1, -1))
        x_1 = x_0.copy().reshape((1, -1))
        for i in range(self.n_dims-2):
            if v[0, i] != x_1[0, i]:
                current_rot = np.eye(self.n_dims)
                cos = np.clip(v[0, i] / x_1[0, i], -1, 1)
                sin = np.sqrt(1 - cos ** 2)
                current_rot[i:i+2, i:i+2] = np.array([[cos, -sin],
                                                    [sin, -cos]])
                m_rot @= current_rot
                x_1 @= current_rot

        if v[0, self.n_dims-2] != x_1[0, self.n_dims-2]:
            current_rot = np.eye(self.n_dims)
            cos = np.clip(v[0, self.n_dims-2] / x_1[0, self.n_dims-2], -1, 1)
            sin = np.sqrt(1 - cos ** 2)
            current_rot[self.n_dims-2:self.n_dims, self.n_dims-2:self.n_dims] = np.array([[cos, -sin],
                                                                                          [sin, -cos]])
            if np.isclose(x_1 @ current_rot, v).all():
                m_rot @= current_rot
            else:
                current_rot[self.n_dims-2:self.n_dims, self.n_dims-2:self.n_dims] = np.array([[cos, sin],
                                                                                              [-sin, -cos]])
                m_rot @= current_rot
        return m_rot

    def sample_from_unit_circle(self, n_dots=1, radius=1):
        vec = np.random.normal(0, 1, size=(self.n_dims, n_dots))
        vec /= la.norm(vec, axis=0)
        r = (np.random.random(n_dots) ** (1 / self.n_dims))
        vec *= r * radius
        return vec.T
    
    def sample_from_start_to_goal_ellipse(self, start, goal, c_best):
        random_unit_coords = self.sample_from_unit_circle()
        c_min = la.norm(goal - start)

        add_const = np.array([c_min / 2] + [0.0] * (self.n_dims - 1))
        mult_const = np.array([c_best / 2] + [np.sqrt(c_best ** 2 - c_min ** 2) / 2] * (self.n_dims - 1))
        rotate_matrix = self.get_rotate_matrix(goal - start)
        result = add_const + random_unit_coords * mult_const
        result = start + result @ rotate_matrix
        while not self.traversable(result.flatten()):
            result = add_const + self.sample_from_unit_circle() * mult_const
            result = start + result @ rotate_matrix
        return result.flatten()
    

class SceneUR10WithObs(SceneUR10):
    def __init__(self, sim, world_collection, robot_collection, obs_collection,
                 sim_base, sim_tip, sim_target, sim_ik, ik_env, ik_group, 
                 n_dims, joints, j_init, delta, j_limits=None):
        super().__init__(sim, world_collection, robot_collection, obs_collection,
                 sim_base, sim_tip, sim_target, sim_ik, ik_env, ik_group, 
                 n_dims, joints, j_init, delta, j_limits=None)
        self.c_obs = []
    
    def clear_obs(self):
        self.c_obs = []
    
    def set_obs(self, obs):
        self.c_obs = copy(obs)
    
    def traversable(self, joint_values):
        joint_values = joint_values.tolist()
        self.set_joints(joint_values)
        col_world, _ = self.sim.checkCollision(self.robot_collection, self.world_collection)
        col_self, _ = self.sim.checkCollision(self.robot_collection, self.robot_collection)
        result = (col_world == 0) and (col_self == 0)
        if result == False:
            self.c_obs.append(joint_values)
        return result
    