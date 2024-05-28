import numpy as np

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

def set_scene(scene_class):
    np.set_printoptions(suppress=True, precision=3)

    client = RemoteAPIClient()
    sim = client.require('sim')

    n_dims = 6
    joints = [sim.getObject("/joint{0}"),
            sim.getObject("/joint{1}"),
            sim.getObject("/joint{2}"),
            sim.getObject("/joint{3}"),
            sim.getObject("/joint{4}"),
            sim.getObject("/joint{5}")]

    j_limits = []
    for j in joints:
        _, interval = sim.getJointInterval(j)
        j_limits.append([interval[0], interval[0] + interval[1]])
    j_limits = np.array(j_limits)
    j_limits /= 1.8

    world_collection = sim.createCollection(0)
    floor = sim.getObject('/Floor')
    sim.addItemToCollection(world_collection, sim.handle_tree, floor, 0)

    rack_collection = sim.createCollection(0)
    rack = sim.getObject('/Floor/rack')
    sim.addItemToCollection(rack_collection, sim.handle_tree, rack, 0)

    robot_base = sim.getObject('/UR10/joint')
    robot_collection = sim.createCollection(0)
    sim.addItemToCollection(robot_collection, sim.handle_tree , robot_base, 0)

    for j in joints:
        sim.setJointPosition(j, 0)

    targets = [sim.getObjectPosition(sim.getObject('/Floor/rack/Target_cylinder[' + str(index) + ']')) for index in range(12)]
    sim_base = sim.getObject('/UR10')
    sim_tip = sim.getObject('/UR10/Tip_D')
    sim_target = sim.getObject('/Target_D')

    target_pos = sim.getObjectPosition(sim_target)
    target_angles = sim.getObjectOrientation(sim_target)

    sim_ik = client.require('simIK')
    ik_env = sim_ik.createEnvironment()
    ik_group = sim_ik.createGroup(ik_env)

    sim_ik.setGroupCalculation(ik_env, ik_group, sim_ik.method_pseudo_inverse, 0, 20)
    sim_ik.addElementFromScene(ik_env, ik_group, sim_base, sim_tip, sim_target, sim_ik.constraint_pose)

    delta = 2 * np.pi / 100
    j_init = [0, 0, 0, 0, 0, 0]

    scene = scene_class(sim, world_collection, robot_collection, rack_collection,
                       sim_base, sim_tip, sim_target, sim_ik, ik_env, ik_group,
                       n_dims, joints, j_init, delta, j_limits)
    return scene, targets
