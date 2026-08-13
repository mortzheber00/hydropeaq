#include "SPlisHSPlasH/Common.h"
#include "SPlisHSPlasH/TimeManager.h"
#include "Utilities/OBJLoader.h"
#include "SPlisHSPlasH/Utilities/SurfaceSampling.h"
#include "SPlisHSPlasH/Viscosity/ViscosityBase.h"
#include <fstream>
#include "SPlisHSPlasH/Simulation.h"
#include "FluidSimulator.h"
#include "Utilities/Timing.h"
#include "Utilities/Counting.h"
#include "Utilities/FileSystem.h"
#include "GazeboSceneLoader.h"
#include <memory>

// Enable memory leak detection
#ifdef _DEBUG
#ifndef EIGEN_ALIGN
#define new DEBUG_NEW
#endif
#endif

using namespace SPH;
using namespace Eigen;
using namespace std;
using namespace Utilities;
using namespace GenParam;
using namespace gazebo;
const std::string objFilePath("/home/ws/sph_boundaries/");

FluidSimulator::FluidSimulator()
{
	REPORT_MEMORY_LEAKS;
	std::cout << "Plugin loaded" << std::endl;
}

void FluidSimulator::RunStep()
{
	simulationSteps++;
	base->timeStepNoGUI();

	// publish all the boundary particles positions
	this->publishFluidParticles();
	this->publishBoundaryParticles();
}

FluidSimulator::~FluidSimulator()
{
	this->connections.clear();
	base->cleanup();
	Utilities::Timing::printAverageTimes();
	Utilities::Timing::printTimeSums();

	Utilities::Counting::printAverageCounts();
	Utilities::Counting::printCounterSums();

	delete Simulation::getCurrent();
}

void FluidSimulator::publishBoundaryParticles()
{
	msgs::Fluid boundary_particles_msg;
	boundary_particles_msg.set_name("boundary_particles");
	for (int i = 0; i < SPH::Simulation::getCurrent()->numberOfBoundaryModels(); ++i) //scene.boundaryModels.size(); ++i)
	{
		BoundaryModel_Akinci2012 *bm = static_cast<BoundaryModel_Akinci2012 *>(SPH::Simulation::getCurrent()->getBoundaryModel(i));
		for (int j = 0; j < (int)bm->numberOfParticles(); j++)
		{
			ignition::math::Vector3d boundary_particles = ignition::math::Vector3d(
				bm->getPosition(j)[0],
				bm->getPosition(j)[1],
				bm->getPosition(j)[2]);
			gazebo::msgs::Set(boundary_particles_msg.add_position(), boundary_particles);
		}
	}
	this->rigidObjPub->Publish(boundary_particles_msg);
}

void FluidSimulator::publishFluidParticles()
{
	if (simulationSteps % 5 == 0)
	{
		msgs::Fluid fluid_positions_msg;
		fluid_positions_msg.set_name("fluid_positions");
		for (unsigned int j = 0; j < Simulation::getCurrent()->numberOfFluidModels(); j++)
		{
			FluidModel *model = Simulation::getCurrent()->getFluidModel(j);
			//std::cout << "Density " << model->getViscosityBase()->VISCOSITY_COEFFICIENT << std::endl;;
			for (unsigned int i = 0; i < model->numActiveParticles(); ++i)
			{
				gazebo::msgs::Set(fluid_positions_msg.add_position(),
								  ignition::math::Vector3d(
									  model->getPosition(i)[0],
									  model->getPosition(i)[1],
									  model->getPosition(i)[2]));
			}
			this->fluidObjPub->Publish(fluid_positions_msg);
		}
	}
}

void FluidSimulator::Init()
{
	this->node = transport::NodePtr(new transport::Node());
	this->node->Init(this->world->Name());

	this->fluidObjPub = this->node->Advertise<msgs::Fluid>("~/fluid_pos", 10);
	this->rigidObjPub = this->node->Advertise<msgs::Fluid>("~/rigids_pos", 10);

	base = std::make_unique<GazeboSimulatorBase>();
	base->init(this->fluidPluginSdf);

	this->ParseSDF();

	base->initSimulation();
	base->initBoundaryData();

	unsigned int totalFluid = 0;
	for (unsigned int j = 0; j < Simulation::getCurrent()->numberOfFluidModels(); j++)
		totalFluid += Simulation::getCurrent()->getFluidModel(j)->numActiveParticles();
	std::cout << "Fluid particles: " << totalFluid << std::endl;

	unsigned int totalBoundary = 0;
	for (int i = 0; i < Simulation::getCurrent()->numberOfBoundaryModels(); i++)
		totalBoundary += static_cast<BoundaryModel_Akinci2012*>(Simulation::getCurrent()->getBoundaryModel(i))->numberOfParticles();
	std::cout << "Boundary particles: " << totalBoundary << std::endl;

	this->publishBoundaryParticles();
}

void FluidSimulator::Load(physics::WorldPtr parent, sdf::ElementPtr sdf)
{
	this->world = parent;
	this->fluidPluginSdf = sdf;

	// beforePhysicsUpdate, not worldUpdateEnd.  World::Update runs
	// worldUpdateBegin, UpdateCollision, beforePhysicsUpdate, UpdatePhysics,
	// worldUpdateEnd, so the two slots sit on opposite sides of the integration.
	// From worldUpdateEnd the boundary is read after ODE has already advanced a
	// step, and the force computed from it lands on the step after that.  Both
	// robots prescribe every joint with Joint::SetPosition, which writes poses
	// outside the solver and leaves no constraint behind, so that step of
	// integration is exactly where a prescribed link drifts: SPH never sees the
	// commanded state, only what the fluid load did to it.  In the body2
	// initialisation pose, where the commanded rate is zero, everything read
	// back is drift, and the force computed from it is fed straight back in.
	//
	// beforePhysicsUpdate is after the prescription and before the integration,
	// so the boundary is read as commanded and the force acts on the very step
	// it was computed for.  It fires after every worldUpdateBegin handler
	// whatever order the plugins loaded in, so this does not depend on the
	// replay plugin connecting first.
	this->connections.push_back(event::Events::ConnectBeforePhysicsUpdate(
		boost::bind(&FluidSimulator::RunStep, this)));
}

void FluidSimulator::RegisterMesh(physics::CollisionPtr collision, std::string extension, std::string path)
{
	// Get collision mesh by name
	const gazebo::common::Mesh *mesh = common::MeshManager::Instance()->GetMesh(collision->GetName());

	// Export the mesh to a temp file in the selected format
	std::string objFilePath = path + collision->GetModel()->GetName() + "_" + collision->GetName() + ".obj";
	common::MeshManager::Instance()->Export(mesh, FileSystem::normalizePath(objFilePath), extension);
	base->processBoundary(collision, objFilePath);
}

void FluidSimulator::ParseSDF()
{
	// get all models from the world
	physics::Model_V models = world->Models();

	// iterate through all models
	for (physics::Model_V::iterator currentModel = models.begin(); currentModel != models.end(); ++currentModel)
	{
		// get all links from the model
		physics::Link_V model_links = currentModel->get()->GetLinks();

		std::cout << "Model: " << currentModel->get()->GetName() << std::endl;

		// iterate through all the links
		for (physics::Link_V::iterator link_it = model_links.begin(); link_it != model_links.end(); ++link_it)
		{
			// get all collisions of the link
			physics::Collision_V collisions = link_it->get()->GetCollisions();

			std::cout << "\t Link: " << link_it->get()->GetName() << std::endl;

			// iterate through all the collisions
			for (physics::Collision_V::iterator collision_it = collisions.begin(); collision_it != collisions.end(); ++collision_it)
			{
				std::cout << "\t\t Collision: " << (*collision_it)->GetName() << std::endl;

				physics::CollisionPtr coll_ptr = boost::static_pointer_cast<physics::Collision>(*collision_it);

				// check the geometry type of the given collision
				sdf::ElementPtr geometry_elem = coll_ptr->GetSDF()->GetElement("geometry");

				// get the name of the geometry
				std::string geometry_type = geometry_elem->GetFirstElement()->GetName();

				// check type of the geometry
				if (geometry_type == "box")
				{
					// Get the size of the box
					ignition::math::Vector3d size = geometry_elem->GetElement(geometry_type)->Get<ignition::math::Vector3d>("size");

					// Create box shape
					common::MeshManager::Instance()->CreateBox((*collision_it)->GetName(), ignition::math::Vector3d(size.X(), size.Y(), size.Z()),
															   ignition::math::Vector2d(1, 1));
					// Generate an obj file in the temporary directory containing the mesh of the box
					RegisterMesh(*collision_it, "obj", objFilePath);
				}

				else if (geometry_type == "cylinder")
				{
					// Cylinder dimensions
					double radius = geometry_elem->GetElement(geometry_type)->GetElement("radius")->Get<double>();
					double length = geometry_elem->GetElement(geometry_type)->GetElement("length")->Get<double>();

					// Create cylinder mesh
					common::MeshManager::Instance()->CreateCylinder((*collision_it)->GetName(), radius, length, 32, 32);

					//Generate an obj file in the temporary directory containing the mesh of the cylinder
					RegisterMesh(*collision_it, "obj", objFilePath);
				}

				else if (geometry_type == "sphere")
				{
					// Sphere radius
					double radius = geometry_elem->GetElement(geometry_type)->GetElement("radius")->Get<double>();

					// Create a sphere mesh
					common::MeshManager::Instance()->CreateSphere((*collision_it)->GetName(), radius, 32, 32);

					// Generate an obj file in the temporary directory containing the mesh of the sphere
					RegisterMesh(*collision_it, "obj", objFilePath);
				}

				else if (geometry_type == "plane")
				{
					ignition::math::Vector3d normal;
					ignition::math::Vector2d size;
					// Plane dimensions. To prevent a huge plane which causes problems when
					// sampling it, for now it is harcoded
					if ((*collision_it)->GetName() == "collision_ground_plane")
					{
						normal = ignition::math::Vector3d(0, 0, 1); // = geom_elem->GetElement(geom_type)->GetElement("normal")->Get<ignition::math::Vector3d>();
						size = ignition::math::Vector2d(2.0, 2.0);	//= geom_elem->GetElement(geom_type)->GetElement("size")->Get<ignition::math::Vector2d>();
					}
					else
					{
						normal = geometry_elem->GetElement(geometry_type)->GetElement("normal")->Get<ignition::math::Vector3d>();
						size = geometry_elem->GetElement(geometry_type)->GetElement("size")->Get<ignition::math::Vector2d>();
					}

					// Generate the plane mesh
					common::MeshManager::Instance()->CreatePlane((*collision_it)->GetName(), ignition::math::Vector3d(0.0, 0.0, 1.0), 0.0, size, ignition::math::Vector2d(4.0, 4.0), ignition::math::Vector2d());

					//Generate an obj file in the temporary directory containing the mesh of the plane
					RegisterMesh(*collision_it, "obj", objFilePath);
				}

				else if (geometry_type == "mesh")
				{
					// get the uri element value
					const std::string uri = geometry_elem->GetElement(geometry_type)->GetElement("uri")->Get<std::string>();

					// get the filepath from the uri
					const std::string filepath = common::SystemPaths::Instance()->FindFileURI(uri);
					const gazebo::common::Mesh *mesh = common::MeshManager::Instance()->GetMesh(filepath);

					std::string fullMeshPath = objFilePath + (*collision_it)->GetModel()->GetName() + "_" + (*collision_it)->GetName() + ".obj";
					// Export the mesh to a temp file in the selected format
					common::MeshManager::Instance()->Export(mesh, FileSystem::normalizePath(fullMeshPath), "obj");
					base->processBoundary(*collision_it, fullMeshPath);
				}
				else
				{
					// Error for other possible weird types
					gzerr << "Collision type [" << geometry_type << "] unimplemented\n";
				}
			}
		}
	}
}

void FluidSimulator::reset()
{ /* 
	Utilities::Timing::printAverageTimes();
	Utilities::Timing::reset();

	Utilities::Counting::printAverageCounts();
	Utilities::Counting::reset();

	Simulation::getCurrent()->reset();
	base->reset(); */
}

GZ_REGISTER_WORLD_PLUGIN(FluidSimulator)
