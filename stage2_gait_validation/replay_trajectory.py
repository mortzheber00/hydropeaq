import numpy as np
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose, Twist
import tf.transformations

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

    traj_pub = rospy.Publisher(
        '/amph/joint_trajectory_controller/command',
        JointTrajectory, queue_size=1, latch=True)

    rospy.loginfo('Waiting for /gazebo/set_model_state service...')
    rospy.wait_for_service('/gazebo/set_model_state')
    set_state_srv = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    data = np.load('task3_solution.npz')
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

    # ── Build and send a single JointTrajectory for all N+1 waypoints ─────
    traj = JointTrajectory()
    traj.header.stamp = rospy.Time.now()
    traj.joint_names = JOINT_NAMES

    for i in range(N + 1):
        pt = JointTrajectoryPoint()
        pt.positions  = X[7:19, i].tolist()
        pt.velocities = X[25:37, i].tolist()   # dq[6:18]
        pt.time_from_start = rospy.Duration(times[i])
        traj.points.append(pt)

    traj_pub.publish(traj)
    rospy.loginfo('JointTrajectory published (%d waypoints, %.2fs).', N + 1, T_total)

    # Keep node alive until the trajectory finishes
    rospy.sleep(T_total + 0.5)


if __name__ == '__main__':
    main()
