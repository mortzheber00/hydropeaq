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

        std::string topic = "/amph/joint_trajectory_replay";
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
        ROS_INFO("TrajectoryReplayPlugin: received trajectory with %zu waypoints",
                 traj_.points.size());
    }

    void OnUpdate() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (traj_.points.empty()) return;

        double t = (model_->GetWorld()->SimTime() - start_time_).Double();
        const auto& pts = traj_.points;

        if (t <= pts.front().time_from_start.toSec()) {
            Apply(pts.front());
            return;
        }
        if (t >= pts.back().time_from_start.toSec()) {
            Apply(pts.back());
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

        for (size_t j = 0; j < traj_.joint_names.size(); ++j) {
            auto joint = model_->GetJoint(traj_.joint_names[j]);
            if (!joint) continue;
            double pos = pts[lo].positions[j] * (1.0 - alpha) + pts[hi].positions[j] * alpha;
            joint->SetPosition(0, pos, true);
        }
    }

    void Apply(const trajectory_msgs::JointTrajectoryPoint& pt) {
        for (size_t j = 0; j < traj_.joint_names.size(); ++j) {
            auto joint = model_->GetJoint(traj_.joint_names[j]);
            if (!joint) continue;
            joint->SetPosition(0, pt.positions[j], true);
        }
    }

    physics::ModelPtr model_;
    event::ConnectionPtr update_conn_;
    std::unique_ptr<ros::NodeHandle> nh_;
    ros::Subscriber sub_;
    trajectory_msgs::JointTrajectory traj_;
    gazebo::common::Time start_time_;
    std::mutex mutex_;
};

GZ_REGISTER_MODEL_PLUGIN(TrajectoryReplayPlugin)

}  // namespace gazebo
