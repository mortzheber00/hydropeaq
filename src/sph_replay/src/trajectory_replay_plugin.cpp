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

        // Default to the model's own namespace, which is what replay_trajectory.py
        // publishes on (RobotSpec.ros is the Gazebo model name).  A <topic>
        // element overrides it.
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

        // Order the joints parent first, once per trajectory.  SetVelocity
        // derives a child link's twist from its parent's *current* one, so a
        // parent has to be set before its child or the child inherits a stale
        // reference.
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

    // An explicit zero rate, not an empty vector: empty means "leave the twist
    // alone", which is the free state this drive exists to stop.
    static std::vector<double> Zeros(size_t n) { return std::vector<double>(n, 0.0); }

    // Hops from this joint's parent link up to a link with no parent joint.
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

        // Outside the trajectory the pose is *held*, so its rate is zero.  Not
        // the waypoint's own velocity: that is the rate the joint will have when
        // the replay reaches it, and imposing it on a joint that is standing
        // still tells SPH the legs are sweeping at up to 2 rad/s while their
        // pose never changes.  The fluid takes that momentum every step from a
        // body that never moves, and the reaction throws the robot across the
        // pool before the gait has started.
        if (t <= pts.front().time_from_start.toSec()) {
            ApplyState(pts.front().positions, Zeros(pts.front().positions.size()));
            return;
        }
        if (t >= pts.back().time_from_start.toSec()) {
            ApplyState(pts.back().positions, Zeros(pts.back().positions.size()));
            return;
        }

        // Binary search for bracketing waypoints
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

        // The rate is the derivative of the lerp above, not the waypoints' own
        // velocities lerped beside it.  Those two agree only where the
        // trajectory is piecewise linear between waypoints, and this one is
        // not: replay_trajectory.py subdivides theta linearly across the OCP's
        // 62.5 ms knots, so a joint travels at the secant of that interval
        // while the published velocity is the optimizer's instantaneous one.
        // On the body2 solution the pair disagrees by 3.1 rad/s RMS and up to
        // 33 rad/s -- more than the joint's own rate on the worst hips, and a
        // paddle-tip error of 1.9 m/s against a 0.44 m/s water entry.  SPH
        // reads both off the same link and turns the difference into a force,
        // which is the loop ApplyState describes below.  The secant is
        // consistent by construction whatever the waypoints carry; the cost is
        // a rate that steps at each waypoint instead of varying across it.
        for (size_t j = 0; j < n; ++j)
            vel[j] = (pts[hi].positions[j] - pts[lo].positions[j]) / (t1 - t0);

        ApplyState(pos, vel);
    }

    // Impose one prescribed state: the angles, then the joint rates that go
    // with them.
    //
    // The rates are not cosmetic.  SetPosition moves link poses behind the
    // solver's back and Link::MoveFrame(.., preserveWorldVelocity=true) leaves
    // the twist exactly as ODE last integrated it, so without this the link
    // velocities are free state that nothing ever writes.  In air that is
    // harmless -- no force, so they stay near zero.  In the fluid it closes a
    // loop: GazeboSimulatorBase::updateBoundaryParticles reads each boundary
    // particle's velocity straight off the link (WorldLinearVel/WorldAngularVel),
    // so SPH sees a body whose position is the commanded one and whose velocity
    // is whatever the last fluid force left behind.  It computes a force from
    // that inconsistency, feeds it back through Link::AddForce, ODE integrates
    // it into a larger inconsistency, and the whole thing runs at the 1 kHz step
    // rate.  Small gain is the high-frequency leg wiggle that starts exactly
    // when the legs touch water; gain above one is the divergence.
    void ApplyState(const std::vector<double>& pos,
                    const std::vector<double>& vel) {
        // Poses first, and all of them: SetPosition moves a joint's whole
        // downstream subtree, so a velocity read before the last one has landed
        // would be taken off a link that is about to move again.
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
