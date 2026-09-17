#!/usr/bin/env python3
"""ROS node that replays an OCP solution's joint trajectory in Gazebo.

Publishes a JointTrajectory to /<robot>/joint_trajectory_replay, which the
replay plugin applies. The base stays free and is moved only by the fluid.
Shuts roslaunch down when the replay is done.

Private parameters:
  ~npz_path   solution file (required)
  ~n_repeat   number of cycles (default 1)
  ~robot      optional; must match the robot recorded in the solution

Usually started by swimming_pool.launch:
  roslaunch amph swimming_pool.launch npz_path:=/home/ws/task3_solution.npz
"""
import os
import sys

import numpy as np
import rospy
from tqdm import tqdm
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

REPO_ROOT = "/home/ws"
sys.path.insert(0, REPO_ROOT)
from stage1_gait_optimization.hydro_model import get_spec  # noqa: E402
from stage1_gait_optimization.hydro_model.robot import QuadrupedRobot  # noqa: E402
from stage1_gait_optimization.hydro_model.trajectory import load_solution  # noqa: E402

# Solution state: [pos (3), quat xyzw (4), theta; base twist (6), thetadot]

# Waypoints per interval for closed-chain robots. The plugin interpolates tree
# joints linearly, which opens the loops between waypoints (BODY2: 1.7 mm gap
# with 1 subdivision, 35 um with 16).
SUBDIV_CLOSED_CHAIN = 16


def resample(theta, thd, times, n_sub):
    """Linearly subdivide each interval of the theta trajectory ``n_sub`` ways."""
    if n_sub <= 1:
        return theta, thd, times
    fine = np.unique(np.concatenate([
        np.linspace(times[i], times[i + 1], n_sub + 1)
        for i in range(len(times) - 1)
    ]))

    def interp(A):
        return np.stack([np.interp(fine, times, row) for row in A])

    return interp(theta), interp(thd), fine


def expand_trajectory(spec, theta, thd):
    """``(joint_names, positions, velocities)`` for all Gazebo joints.

    For closed-chain robots theta is expanded to every tree joint, so the
    passive joints are prescribed too and the loops stay closed.
    """
    if spec.coordinate_map is None:
        return list(spec.actuated_joint_names), theta, thd

    robot = QuadrupedRobot(spec)
    cmap, model = robot.coord_map, robot.model

    # (name, q offset, nq, v offset) per tree joint, excluding the free-flyer.
    # Continuous joints have nq = 2 (cos, sin).
    tree = [(model.names[j], model.joints[j].idx_q - 7, model.joints[j].nq,
             model.joints[j].idx_v - 6)
            for j in range(1, model.njoints)
            if model.joints[j].idx_q >= 7]

    n = theta.shape[1]
    pos = np.zeros((len(tree), n))
    vel = np.zeros((len(tree), n))
    for k in range(n):
        q_j = cmap.expand_numeric(theta[:, k])
        v_j = cmap.v_numeric(theta[:, k], thd[:, k])
        for i, (_, iq, nq_j, iv) in enumerate(tree):
            pos[i, k] = np.arctan2(q_j[iq + 1], q_j[iq]) if nq_j == 2 else q_j[iq]
            vel[i, k] = v_j[iv]

    # Unwrap so the plugin never interpolates across the +-pi cut.
    return [name for name, _, _, _ in tree], np.unwrap(pos, axis=1), vel


def main():
    rospy.init_node('trajectory_replay')

    npz_path = rospy.get_param('~npz_path', '')
    n_repeat  = rospy.get_param('~n_repeat', 1)
    declared = rospy.get_param('~robot', '')

    if not npz_path:
        rospy.logfatal('~npz_path parameter is required')
        return

    # The robot is read from the solution file.
    sol = load_solution(npz_path)
    if declared and declared != sol['robot']:
        rospy.logfatal('%s holds a solution for %r, but ~robot says %r',
                       npz_path, sol['robot'], declared)
        return

    spec = get_spec(sol['robot'])
    X = sol['X']            # (nx, N+1)
    T_total = sol['T']
    N = sol['N']
    nq = sol['nq']

    n_theta = nq - 7
    times = np.linspace(0, T_total, N + 1)
    theta = X[7:nq, :]
    thd = X[nq + 6:nq + 6 + n_theta, :]

    n_sub = 1 if spec.coordinate_map is None else SUBDIV_CLOSED_CHAIN
    theta, thd, times = resample(theta, thd, times, n_sub)
    joint_names, positions, velocities = expand_trajectory(spec, theta, thd)
    rospy.loginfo('%s: %d actuated coordinate(s) -> %d prescribed joint(s), '
                  '%d waypoints per cycle.',
                  spec.name, n_theta, len(joint_names), len(times))

    traj_pub = rospy.Publisher(
        '/%s/joint_trajectory_replay' % spec.ros,
        JointTrajectory, queue_size=1, latch=True)

    # --- Move to the start pose ---
    init_traj = JointTrajectory()
    init_traj.header.stamp = rospy.Time.now()
    init_traj.joint_names = joint_names
    init_pt = JointTrajectoryPoint()
    init_pt.positions  = positions[:, 0].tolist()
    init_pt.velocities = velocities[:, 0].tolist()
    init_pt.time_from_start = rospy.Duration(2.0)
    init_traj.points.append(init_pt)
    traj_pub.publish(init_traj)
    rospy.loginfo('Initial joint positions sent.')

    try:
        init_tick = 0.01
        init_steps = int(1.5 / init_tick)
        with tqdm(total=init_steps, desc='Initializing', unit='step',
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            for _ in range(init_steps):
                rospy.sleep(init_tick)
                pbar.update(1)
    except rospy.exceptions.ROSInterruptException:
        return

    # --- Full trajectory, repeated n_repeat times ---
    traj = JointTrajectory()
    traj.header.stamp = rospy.Time.now()
    traj.joint_names = joint_names

    for rep in range(n_repeat):
        t_offset = rep * T_total
        for i in range(len(times)):
            # The first point of later cycles duplicates the previous last one
            if rep > 0 and i == 0:
                continue
            pt = JointTrajectoryPoint()
            pt.positions  = positions[:, i].tolist()
            pt.velocities = velocities[:, i].tolist()
            pt.time_from_start = rospy.Duration(t_offset + times[i])
            traj.points.append(pt)

    T_full = n_repeat * T_total
    traj_pub.publish(traj)
    rospy.loginfo('JointTrajectory published (%d waypoints, %.2fs, %d repeat(s)).',
                  len(traj.points), T_full, n_repeat)

    tick = 0.01
    steps = int(T_full / tick)
    try:
        with tqdm(total=steps, desc='Replaying', unit='step',
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            for _ in range(steps):
                rospy.sleep(tick)
                pbar.update(1)
    except rospy.exceptions.ROSInterruptException:
        return

    rospy.sleep(0.2)
    rospy.loginfo('Trajectory finished, shutting down.')
    os.system('pkill -SIGINT -f roslaunch')


if __name__ == '__main__':
    main()
