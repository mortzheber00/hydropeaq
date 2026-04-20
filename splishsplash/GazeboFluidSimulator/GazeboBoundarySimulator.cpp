#include "GazeboBoundarySimulator.h"
#include "GazeboSimulatorBase.h"
#include "Utilities/FileSystem.h"
#include "SPlisHSPlasH/Simulation.h"
#include "Utilities/PartioReaderWriter.h"
#include "Utilities/Logger.h"
#include "Utilities/Timing.h"
#include "SPlisHSPlasH/Utilities/SurfaceSampling.h"
#include "Utilities/OBJLoader.h"
#include "SPlisHSPlasH/TriangleMesh.h"
#include "GazeboSceneConfiguration.h"
#include "SPlisHSPlasH/TimeManager.h"

using namespace std;
using namespace SPH;
using namespace Utilities;

GazeboBoundarySimulator::GazeboBoundarySimulator(GazeboSimulatorBase *base)
{
	m_base = base;
}

GazeboBoundarySimulator::~GazeboBoundarySimulator()
{
}

void GazeboBoundarySimulator::updateBoundaryForces()
{
	Real h = TimeManager::getCurrent()->getTimeStepSize();
	Simulation *sim = Simulation::getCurrent();
	//GazeboSceneLoader::Scene &scene = getScene();
	const unsigned int nObjects = sim->numberOfBoundaryModels();
	for (unsigned int i = 0; i < nObjects; i++)
	{
		BoundaryModel *bm = sim->getBoundaryModel(i);
		GazeboRigidBody *rbo = dynamic_cast<GazeboRigidBody *>(bm->getRigidBodyObject());
		if (rbo->isDynamic())
		{
			gazebo::physics::CollisionPtr gazeboCollision = rbo->getGazeboCollision();
			Vector3r force, torque;
			// change rigid body fluid position?
			bm->getForceAndTorque(force, torque);
			const ignition::math::Vector3d gazeboForce = ignition::math::Vector3d(force[0], force[1], force[2]);
			const ignition::math::Vector3d gazeboTorque = ignition::math::Vector3d(torque[0], torque[1], torque[2]);
			gazeboCollision->GetLink()->AddForce(gazeboForce);
		
     		gazeboCollision->GetLink()->AddTorque(gazeboTorque);
			bm->clearForceAndTorque();
		}
	}
}

void GazeboBoundarySimulator::timeStep()
{
	updateBoundaryForces();

	Simulation *sim = Simulation::getCurrent();
	if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
		m_base->updateBoundaryParticles(false);
	else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Koschier2017)
		m_base->updateDMVelocity();
	else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Bender2019)
		m_base->updateVMVelocity();
}

void GazeboBoundarySimulator::loadObj(const std::string &filename, TriangleMesh &mesh, const Vector3r &scale)
{
	std::vector<OBJLoader::Vec3f> x;
	std::vector<OBJLoader::Vec3f> normals;
	std::vector<MeshFaceIndices> faces;
	OBJLoader::Vec3f s = {(float)scale[0], (float)scale[1], (float)scale[2]};
	OBJLoader::loadObj(filename, &x, &faces, &normals, nullptr, s);

	mesh.release();
	const unsigned int nPoints = (unsigned int)x.size();
	const unsigned int nFaces = (unsigned int)faces.size();
	mesh.initMesh(nPoints, nFaces);
	for (unsigned int i = 0; i < nPoints; i++)
	{
		mesh.addVertex(Vector3r(x[i][0], x[i][1], x[i][2]));
	}
	for (unsigned int i = 0; i < nFaces; i++)
	{
		// Reduce the indices by one
		int posIndices[3];
		for (int j = 0; j < 3; j++)
		{
			posIndices[j] = faces[i].posIndices[j] - 1;
		}

		mesh.addFace(&posIndices[0]);
	}

	LOG_INFO << "Number of triangles: " << nFaces;
	LOG_INFO << "Number of vertices: " << nPoints;
}

void GazeboBoundarySimulator::initBoundaryData()
{
	const std::string &sceneFile = GazeboSceneConfiguration::getCurrent()->getSceneFile();
	const Utilities::GazeboSceneLoader::Scene &scene = GazeboSceneConfiguration::getCurrent()->getScene();
	Simulation *sim = Simulation::getCurrent();
	for (unsigned int i = 0; i < scene.boundaryModels.size(); i++)
	{
		GazeboRigidBody *rigidBody = new GazeboRigidBody();
		rigidBody->setGazeboCollision(scene.boundaryModels[i]->rigidBody);
		if (scene.boundaryModels[i]->rigidBody->GetModel()->GetSDF()->HasElement("static"))
		{
			rigidBody->setDynamic(scene.boundaryModels[i]->dynamic);
			std::cout << scene.boundaryModels[i]->rigidBody->GetModel()->GetName() << "is dynamic " << scene.boundaryModels[i]->dynamic << std::endl;
		}
		TriangleMesh &geo = rigidBody->getGeometry();
		loadObj(scene.boundaryModels[i]->objFilePath, geo, scene.boundaryModels[i]->scale);

		std::vector<Vector3r> boundaryParticles;
		if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
		{
			const auto samplePoissonDisk = [&]() {
				LOG_INFO << "Poisson disk surface sampling of " << scene.boundaryModels[i]->objFilePath;
				START_TIMING("Poisson disk sampling");
				PoissonDiskSampling sampling;
				sampling.sampleMesh(geo.numVertices(), geo.getVertices().data(), geo.numFaces(), geo.getFaces().data(), scene.particleRadius, 10, 1, boundaryParticles);
				STOP_TIMING_AVG;
			};
			const auto sampleRegularTriangle = [&]() {
				LOG_INFO << "Regular triangle surface sampling of " << scene.boundaryModels[i]->objFilePath;
				START_TIMING("Regular triangle sampling");
				RegularTriangleSampling sampling;
				sampling.sampleMesh(geo.numVertices(), geo.getVertices().data(), geo.numFaces(), geo.getFaces().data(), 1.5f * scene.particleRadius, boundaryParticles);
				STOP_TIMING_AVG;
			};
			if (SurfaceSamplingMode::PoissonDisk == scene.boundaryModels[i]->samplingMode)
				samplePoissonDisk();
			else if (SurfaceSamplingMode::RegularTriangle == scene.boundaryModels[i]->samplingMode)
				sampleRegularTriangle();
			else
			{
				LOG_WARN << "Unknown surface sampling method: " << scene.boundaryModels[i]->samplingMode;
				LOG_WARN << "Falling back to:";
				sampleRegularTriangle();
			}

			// transform particles
			if (!rigidBody->isDynamic())
			{
				for (unsigned int j = 0; j < (unsigned int)boundaryParticles.size(); j++)
					boundaryParticles[j] = scene.boundaryModels[i]->rotation * boundaryParticles[j] + scene.boundaryModels[i]->translation;
			}
		}

		rigidBody->setWorldSpacePosition(scene.boundaryModels[i]->translation);
		rigidBody->setWorldSpaceRotation(scene.boundaryModels[i]->rotation);
		rigidBody->setPosition(scene.boundaryModels[i]->translation);
		rigidBody->setRotation(scene.boundaryModels[i]->rotation);

		if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
		{
			BoundaryModel_Akinci2012 *bm = new BoundaryModel_Akinci2012();
			bm->initModel(rigidBody, static_cast<unsigned int>(boundaryParticles.size()), &boundaryParticles[0]);
			sim->addBoundaryModel(bm);
		}
		else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Koschier2017)
		{
			BoundaryModel_Koschier2017 *bm = new BoundaryModel_Koschier2017();
			bm->initModel(rigidBody);
			sim->addBoundaryModel(bm);
			SPH::TriangleMesh &mesh = rigidBody->getGeometry();
			//m_base->initDensityMap(mesh.getVertices(), mesh.getFaces(), scene.boundaryModels[i], md5, false, bm);
		}
		else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Bender2019)
		{
			BoundaryModel_Bender2019 *bm = new BoundaryModel_Bender2019();
			bm->initModel(rigidBody);
			sim->addBoundaryModel(bm);
			SPH::TriangleMesh &mesh = rigidBody->getGeometry();
			//m_base->initVolumeMap(mesh.getVertices(), mesh.getFaces(), scene.boundaryModels[i], md5, false, bm);
		}
		for (unsigned int j = 0; j < geo.numVertices(); j++)
			geo.getVertices()[j] = scene.boundaryModels[i]->rotation * geo.getVertices()[j] + scene.boundaryModels[i]->translation;

		geo.updateNormals();
		geo.updateVertexNormals();
	}
	sim->performNeighborhoodSearchSort();
	if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
	{
		m_base->updateBoundaryParticles(false);
		Simulation::getCurrent()->updateBoundaryVolume();
	}

	m_base->initRbVertixPositions();
#ifdef GPU_NEIGHBORHOOD_SEARCH
	// copy the particle data to the GPU
	sim->getNeighborhoodSearch()->update_point_sets();
#endif
}