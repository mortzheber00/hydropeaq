#!/usr/bin/env python3
"""Prescribe an OCP solution to a robot in Gazebo.

Parameters (private):
    ~npz_path   solution written by extract_solution()
    ~n_repeat   how many times to repeat the cycle
    ~robot      registered robot name; see hydro_model/robots/.  Optional --
    the solution file records which robot it belongs to, so this only has to
    be set to assert that the two agree.

Only the joints are prescribed.  They are published as a JointTrajectory that
the model's replay plugin interpolates and applies with Joint::SetPosition.

The base is never touched: it spawns wherever the world file puts it and is
free from then on, so its trajectory is whatever the fluid coupling produces.
This script used to set it once from the solution's q0 via
/gazebo/set_model_state, but that call had in fact never taken effect -- the
vendored gazebo_ros rejects a ModelState whose scale is (0,0,0), which is the
default, and reports the failure only in the response nobody read.  Both robots
have therefore always run with a free base, and that is the intended behaviour.

For a closed-chain robot the trajectory carries the *tree* joints, not the
actuated ones -- see expand_trajectory().
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

# State layout in the OCP solution, with nq read from the file:
#   q[0:3]      = base position (x, y, z)
#   q[3:7]      = base quaternion (x, y, z, w)
#   q[7:nq]     = actuated coordinates (theta)
#   dq[0:6]     = base velocity (linear xyz + angular xyz)
#   dq[6:]      = actuated velocities (thetadot)

# Waypoints per solution interval for a closed-chain robot.  The plugin lerps
# between waypoints in joint space, but tree angles are a nonlinear function of
# theta, so interpolating the knots directly leaves the loops open in between:
# on the body2 solution the pins separate by up to 1.7 mm mid-interval, against
# links ~50 mm long.  16 subdivisions puts a waypoint every ~4 ms and brings
# that to 35 um (closure is exact at the waypoints themselves).  A serial robot
# needs none of this.
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
    """Return (joint_names, positions, velocities) for the joints Gazebo has.

    A serial robot's actuated coordinates *are* its joints, so this is a
    rename.  A closed-chain robot's URDF is a tree with more joints than
    degrees of freedom -- 24 against 8 for BODY2 -- because URDF cannot express
    the loop-closure pins.  Prescribing only the 8 actuated joints would leave
    the other 16 links to swing free and the legs would come apart, so the
    coordinate map expands theta onto the whole tree and every joint of it is
    prescribed.  The loops then close by construction.
    """
    if spec.coordinate_map is None:
        return list(spec.actuated_joint_names), theta, thd

    robot = QuadrupedRobot(spec)
    cmap, model = robot.coord_map, robot.model

    # (name, offset into the joint block, nq, offset into the velocity block)
    # per tree joint, in model order, skipping the free-flyer the coordinate
    # map's blocks exclude.  An unbounded joint stores (cos, sin) rather than
    # an angle, so its nq is 2.
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

    # atan2 cuts at +-pi and BODY2's hips run right up to it, so a joint can
    # come back on the far side between two waypoints; the plugin would then
    # lerp the long way round.  Unwrapping is a no-op unless that happens.
    return [name for name, _, _, _ in tree], np.unwrap(pos, axis=1), vel


def main():
    rospy.init_node('trajectory_replay')

    npz_path = rospy.get_param('~npz_path', '')
    n_repeat  = rospy.get_param('~n_repeat', 1)
    declared = rospy.get_param('~robot', '')

    if not npz_path:
        rospy.logfatal('~npz_path parameter is required')
        return

    # The solution names its own robot (v1 files predate the field and are read
    # as what they are), so nothing here has to assume one.
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

    # ── Send initial joint positions so legs reach start pose ─────────────
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

    # ── Build trajectory repeated n_repeat times ──────────────────────────
    traj = JointTrajectory()
    traj.header.stamp = rospy.Time.now()
    traj.joint_names = joint_names

    for rep in range(n_repeat):
        t_offset = rep * T_total
        for i in range(len(times)):
            # Skip the first point of subsequent cycles — it duplicates the last
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
