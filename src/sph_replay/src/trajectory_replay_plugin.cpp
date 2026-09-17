// Gazebo model plugin that replays a JointTrajectory by setting joint positions
// and velocities every physics step (published by scripts/replay_trajectory.py).
#include <algorithm>
#include <cstddef>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include <gazebo/common/common.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/gazebo.hh>

#include <ros/ros.h>
#include <trajectory_msgs/JointTrajectory.h>

namespace gazebo {

class TrajectoryReplayPlugin : public ModelPlugin {
public:
    void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override {
        model_ = model;

        if (!ros::isInitialized()) {
            ROS_FATAL("TrajectoryReplayPlugin: ROS not initialized");
            return;
        }

        // Default: /<model>/joint_trajectory_replay; override with <topic>.
        std::string topic = "/" + model->GetName() + "/joint_trajectory_replay";
        if (sdf->HasElement("topic"))
            topic = sdf->Get<std::string>("topic");

        nh_ = std::make_unique<ros::NodeHandle>();
        sub_ = nh_->subscribe(topic, 1, &TrajectoryReplayPlugin::OnTrajectory, this);

        update_conn_ = event::Events::ConnectWorldUpdateBegin(
            std::bind(&TrajectoryReplayPlugin::OnUpdate, this));

        ROS_INFO("TrajectoryReplayPlugin loaded, listening on %s", topic.c_str());
    }

private:
    void OnTrajectory(const trajectory_msgs::JointTrajectory::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        traj_ = *msg;
        start_time_ = model_->GetWorld()->SimTime();

        // Sort joints parent first: SetVelocity uses the parent's current twist.
        order_.resize(traj_.joint_names.size());
        for (size_t j = 0; j < order_.size(); ++j) order_[j] = j;
        std::vector<size_t> depth(order_.size());
        for (size_t j = 0; j < order_.size(); ++j)
            depth[j] = Depth(model_->GetJoint(traj_.joint_names[j]));
        std::stable_sort(order_.begin(), order_.end(),
                         [&](size_t a, size_t b) { return depth[a] < depth[b]; });

        ROS_INFO("TrajectoryReplayPlugin: received trajectory with %zu waypoints",
                 traj_.points.size());
    }

    // Explicit zero rates (an empty vector would leave the velocities untouched).
    static std::vector<double> Zeros(size_t n) { return std::vector<double>(n, 0.0); }

    // Number of joints between this joint's parent link and the root.
    static size_t Depth(const physics::JointPtr& joint) {
        if (!joint) return 0;
        size_t d = 0;
        physics::LinkPtr link = joint->GetParent();
        while (link && d < 64) {
            const physics::Joint_V up = link->GetParentJoints();
            if (up.empty()) break;
            link = up.front()->GetParent();
            ++d;
        }
        return d;
    }

    void OnUpdate() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (traj_.points.empty()) return;

        double t = (model_->GetWorld()->SimTime() - start_time_).Double();
        const auto& pts = traj_.points;

        // Before and after the trajectory the pose is held with zero rate.
        // Using the waypoint velocity here would let SPH see moving legs that
        // never move and push the robot away before the gait starts.
        if (t <= pts.front().time_from_start.toSec()) {
            ApplyState(pts.front().positions, Zeros(pts.front().positions.size()));
            return;
        }
        if (t >= pts.back().time_from_start.toSec()) {
            ApplyState(pts.back().positions, Zeros(pts.back().positions.size()));
            return;
        }

        // Bracketing waypoints (binary search)
        size_t lo = 0, hi = pts.size() - 1;
        while (hi - lo > 1) {
            size_t mid = (lo + hi) / 2;
            if (pts[mid].time_from_start.toSec() <= t) lo = mid;
            else hi = mid;
        }

        double t0 = pts[lo].time_from_start.toSec();
        double t1 = pts[hi].time_from_start.toSec();
        double alpha = (t - t0) / (t1 - t0);

        const size_t n = traj_.joint_names.size();
        std::vector<double> pos(n), vel(n);
        for (size_t j = 0; j < n; ++j)
            pos[j] = pts[lo].positions[j] * (1.0 - alpha) + pts[hi].positions[j] * alpha;

        // Rate = slope of the interpolated position, not the waypoint velocities,
        // so position and velocity stay consistent for SPH (they can differ a
        // lot after the subdivision in replay_trajectory.py). The rate is
        // piecewise constant as a result.
        for (size_t j = 0; j < n; ++j)
            vel[j] = (pts[hi].positions[j] - pts[lo].positions[j]) / (t1 - t0);

        ApplyState(pos, vel);
    }

    // Set joint positions, then joint velocities.
    //
    // Velocities must be set too: SetPosition leaves the link twists as ODE
    // integrated them, and SPH reads boundary velocities from the links. A
    // mismatch between commanded pose and link velocity feeds back through the
    // fluid force and causes leg jitter or divergence in water.
    void ApplyState(const std::vector<double>& pos,
                    const std::vector<double>& vel) {
        // All positions first, since SetPosition moves the whole subtree.
        for (size_t j = 0; j < traj_.joint_names.size() && j < pos.size(); ++j) {
            auto joint = model_->GetJoint(traj_.joint_names[j]);
            if (joint) joint->SetPosition(0, pos[j], true);
        }
        if (vel.empty()) return;
        for (size_t k : order_) {
            if (k >= vel.size()) continue;
            auto joint = model_->GetJoint(traj_.joint_names[k]);
            if (joint) joint->SetVelocity(0, vel[k]);
        }
    }

    physics::ModelPtr model_;
    event::ConnectionPtr update_conn_;
    std::unique_ptr<ros::NodeHandle> nh_;
    ros::Subscriber sub_;
    trajectory_msgs::JointTrajectory traj_;
    gazebo::common::Time start_time_;
    std::mutex mutex_;
    std::vector<size_t> order_;   // joint indices, parent first
};

GZ_REGISTER_MODEL_PLUGIN(TrajectoryReplayPlugin)

}  // namespace gazebo
