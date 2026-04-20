/* Desc: Fluid World plugin
 * Author: Manos Angelidis
 * Email: angelidis@fortiss.org
 * Date: 15 Dec 2018
 */

#ifndef FLUID_SIMULATOR_HH
#define FLUID_SIMULATOR_HH

#include <vector>
#include <map>
#include <string>
#include <list>

#include "gazebo/gazebo.hh"
#include "gazebo/util/system.hh"
#include "gazebo/physics/physics.hh"
#include "gazebo/physics/PhysicsTypes.hh"
//#include "SPlisHSPlasH/TriangleMesh.h"
#include "SPlisHSPlasH/Common.h"
#include "GazeboSimulatorBase.h"
#include "GazeboRigidBody.h"
#include <memory>

enum CollisionGeometry
{
  plane,
  sphere,
  box,
  cylinder,
  mesh,
  trimesh,
  polyline,
  heightmap,
  multiray,
  ray
};

namespace gazebo
{

  /// \brief FluidSimulator class
  class GAZEBO_VISIBLE FluidSimulator : public WorldPlugin
  {
    /// \brief Constructor
  public:
    FluidSimulator();

    /// \brief Destructor
  public:
    ~FluidSimulator();

    /// \brief Initialize the fluid simulator
  public:
    void Init();

    /// \brief Run a timestep in the fluid world
  public:
    void RunStep();

    /// \brief Load plugin
  protected:
    void Load(physics::WorldPtr parent, sdf::ElementPtr sdf);
    void initBoundaryData();
    void reset();
    //void loadObj(const std::string &filename, SPH::TriangleMesh &geo, const Vector3r &scale);
    void ParseSDF();
    void RegisterMesh(physics::CollisionPtr collision, std::string extension, std::string path);

    int activeParticles;
    int maxParticles;
    int particles_number;

  private:
    std::vector<event::ConnectionPtr> connections;
    std::unique_ptr<SPH::GazeboSimulatorBase> base;
    unsigned int currentFluidModel = 0;
    transport::NodePtr node;
    physics::WorldPtr world;
    sdf::ElementPtr fluidPluginSdf;
    /// \brief Publisher for fluid object visual messages.
    transport::PublisherPtr fluidObjPub;
    transport::PublisherPtr rigidObjPub;
    unsigned int simulationSteps;
    void publishFluidParticles();
    void publishBoundaryParticles();
  };
} // namespace gazebo
#endif
