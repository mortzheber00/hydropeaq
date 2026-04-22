#!/usr/bin/env python3
import os
import numpy as np
import rospy
from tqdm import tqdm
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

JOINT_NAMES = [
    'Front_Left_Side_joint', 'Front_Left_Thigh_joint', 'Front_Left_Calf_joint',
    'Front_Right_Side_joint', 'Front_Right_Thigh_joint', 'Front_Right_Calf_joint',
    'Hind_Left_Side_joint', 'Hind_Left_Thigh_joint', 'Hind_Left_Calf_joint',
    'Hind_Right_Side_joint', 'Hind_Right_Thigh_joint', 'Hind_Right_Calf_joint',
]

# State layout in the OCP solution:
#   q[0:3]   = base position (x, y, z)
#   q[3:7]   = base quaternion (x, y, z, w)
#   q[7:19]  = joint angles (12 DOF)
#   dq[0:6]  = base velocity (linear xyz + angular xyz)
#   dq[6:18] = joint velocities


def set_model_pose(set_state_srv, pos, quat, lin_vel, ang_vel):
    state = ModelState()
    state.model_name = 'amph'
    state.reference_frame = 'world'
    state.pose.position.x = float(pos[0])
    state.pose.position.y = float(pos[1])
    state.pose.position.z = float(pos[2])
    state.pose.orientation.x = float(quat[0])
    state.pose.orientation.y = float(quat[1])
    state.pose.orientation.z = float(quat[2])
    state.pose.orientation.w = float(quat[3])
    state.twist.linear.x  = float(lin_vel[0])
    state.twist.linear.y  = float(lin_vel[1])
    state.twist.linear.z  = float(lin_vel[2])
    state.twist.angular.x = float(ang_vel[0])
    state.twist.angular.y = float(ang_vel[1])
    state.twist.angular.z = float(ang_vel[2])
    set_state_srv(state)


def main():
    rospy.init_node('trajectory_replay')

    npz_path = rospy.get_param('~npz_path', '')
    n_repeat  = rospy.get_param('~n_repeat', 1)

    if not npz_path:
        rospy.logfatal('~npz_path parameter is required')
        return

    traj_pub = rospy.Publisher(
        '/amph/joint_trajectory_replay',
        JointTrajectory, queue_size=1, latch=True)


    rospy.loginfo('Waiting for /gazebo/set_model_state service...')
    rospy.wait_for_service('/gazebo/set_model_state')
    set_state_srv = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    data = np.load(npz_path)
    X = data['X']           # (nx, N+1)
    T_total = float(data['T'])
    N = int(data['N'])

    times = np.linspace(0, T_total, N + 1)

    # ── Set initial base pose in Gazebo ───────────────────────────────────
    q0 = X[:19, 0]
    dq0 = X[19:, 0]
    set_model_pose(set_state_srv,
                   pos=q0[0:3], quat=q0[3:7],
                   lin_vel=dq0[0:3], ang_vel=dq0[3:6])
    
    rospy.loginfo('Initial base pose set.')

    # ── Send initial joint positions so legs reach start pose ─────────────
    init_traj = JointTrajectory()
    init_traj.header.stamp = rospy.Time.now()
    init_traj.joint_names = JOINT_NAMES
    init_pt = JointTrajectoryPoint()
    init_pt.positions  = X[7:19, 0].tolist()
    init_pt.velocities = X[25:37, 0].tolist()
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
    traj.joint_names = JOINT_NAMES

    for rep in range(n_repeat):
        t_offset = rep * T_total
        for i in range(N + 1):
            # Skip the first point of subsequent cycles — it duplicates the last
            if rep > 0 and i == 0:
                continue
            pt = JointTrajectoryPoint()
            pt.positions  = X[7:19, i].tolist()
            pt.velocities = X[25:37, i].tolist()   # dq[6:18]
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
