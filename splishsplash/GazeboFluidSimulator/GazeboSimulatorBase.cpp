#include "GazeboSimulatorBase.h"
#include "GazeboSceneConfiguration.h"
#include "Utilities/FileSystem.h"
#include "SPlisHSPlasH/TimeManager.h"
#include "Utilities/PartioReaderWriter.h"
#include "SPlisHSPlasH/Emitter.h"
#include "SPlisHSPlasH/EmitterSystem.h"
#include "SPlisHSPlasH/Simulation.h"
#include "SPlisHSPlasH/Vorticity/MicropolarModel_Bender2017.h"
#include "SPlisHSPlasH/Viscosity/ViscosityBase.h"
#include "NumericParameter.h"
#include "Utilities/Logger.h"
#include "Utilities/Timing.h"
#include "Utilities/Counting.h"
#include "Utilities/Version.h"
#include "Utilities/SystemInfo.h"
#include "extern/partio/src/lib/Partio.h"
#include "SPlisHSPlasH/Utilities/GaussQuadrature.h"
#include "SPlisHSPlasH/Utilities/SimpleQuadrature.h"
#include "SPlisHSPlasH/Utilities/VolumeSampling.h"
#include "Utilities/OBJLoader.h"
#include "Utilities/BinaryFileReaderWriter.h"
#include "Exporter/ExporterBase.h"
#include "Exporter/ParticleExporter_Partio.h"
#include "Exporter/ParticleExporter_VTK.h"
#include "Exporter/RigidBodyExporter_BIN.h"
#include "Exporter/RigidBodyExporter_OBJ.h"
#include "Exporter/RigidBodyExporter_VTK.h"

using namespace SPH;

INIT_LOGGING
INIT_TIMING
INIT_COUNTING

using namespace SPH;
using namespace std;
using namespace GenParam;
using namespace Utilities;

int GazeboSimulatorBase::DATA_EXPORT_FPS = -1;
int GazeboSimulatorBase::PARTICLE_EXPORT_ATTRIBUTES = -1;
int GazeboSimulatorBase::STATE_EXPORT = -1;
int GazeboSimulatorBase::STATE_EXPORT_FPS = -1;

GazeboSimulatorBase::GazeboSimulatorBase()
{
	Utilities::logger.addSink(unique_ptr<Utilities::ConsoleSink>(new Utilities::ConsoleSink(Utilities::LogLevel::INFO)));

	m_boundarySimulator = nullptr;
	m_isStaticScene = true;
	m_useParticleCaching = true;
	m_enableRigidBodyVTKExport = true;
	m_enableRigidBodyExport = false;
	m_enableStateExport = false;
	m_framesPerSecond = 25;
	m_framesPerSecondState = 1;
	m_nextFrameTime = 0.0;
	m_nextFrameTimeState = 0.0;
	m_frameCounter = 1;
	m_isFirstFrame = true;
	m_isFirstFrameVTK = true;
	m_firstState = true;
	m_particleAttributes = "velocity";
	m_timeStepCB = nullptr;
	m_simulation_steps = 0;
#ifdef DL_OUTPUT
	m_nextTiming = 1.0;
#endif
}

GazeboSimulatorBase::~GazeboSimulatorBase()
{
	Utilities::Timing::printAverageTimes();
	Utilities::Timing::printTimeSums();

	Utilities::Counting::printAverageCounts();
	Utilities::Counting::printCounterSums();

	delete m_boundarySimulator;
	cleanupExporters();
}

void GazeboSimulatorBase::initParameters()
{
	ParameterObject::initParameters();

	DATA_EXPORT_FPS = createNumericParameter("dataExportFPS", "Export FPS", &m_framesPerSecond);
	setGroup(DATA_EXPORT_FPS, "Export");
	setDescription(DATA_EXPORT_FPS, "Frame rate of partio, vtk and rigid body export.");

	STATE_EXPORT = createBoolParameter("enableStateExport", "Simulation state export", &m_enableStateExport);
	setGroup(STATE_EXPORT, "Export");
	setDescription(STATE_EXPORT, "Enable/disable export of complete simulation state.");

	STATE_EXPORT_FPS = createNumericParameter("stateExportFPS", "State export FPS", &m_framesPerSecondState);
	setGroup(STATE_EXPORT_FPS, "Export");
	setDescription(STATE_EXPORT_FPS, "Frame rate of simulation state export.");

	PARTICLE_EXPORT_ATTRIBUTES = createStringParameter("particleAttributes", "Export attributes", &m_particleAttributes);
	getParameter(PARTICLE_EXPORT_ATTRIBUTES)->setReadOnly(true);
	setGroup(PARTICLE_EXPORT_ATTRIBUTES, "Export");
	setDescription(PARTICLE_EXPORT_ATTRIBUTES, "Attributes that are exported in the partio files (except id and position).");

	for (size_t i = 0; i < m_particleExporters.size(); i++)
	{
		m_particleExporters[i].m_id = createBoolParameter(
			m_particleExporters[i].m_key, m_particleExporters[i].m_name,
			[i, this]() -> bool { return m_particleExporters[i].m_exporter->getActive(); },
			[i, this](bool active) { m_particleExporters[i].m_exporter->setActive(active); });
		setGroup(m_particleExporters[i].m_id, "Export");
		setDescription(m_particleExporters[i].m_id, m_particleExporters[i].m_description);
	}

	for (size_t i = 0; i < m_rbExporters.size(); i++)
	{
		m_rbExporters[i].m_id = createBoolParameter(
			m_rbExporters[i].m_key, m_rbExporters[i].m_name,
			[i, this]() -> bool { return m_rbExporters[i].m_exporter->getActive(); },
			[i, this](bool active) { m_rbExporters[i].m_exporter->setActive(active); });
		setGroup(m_rbExporters[i].m_id, "Export");
		setDescription(m_rbExporters[i].m_id, m_rbExporters[i].m_description);
	}
}

void GazeboSimulatorBase::processBoundary(physics::CollisionPtr collision, std::string objFilePath)
{
	this->m_sceneLoader->processBoundary(GazeboSceneConfiguration::getCurrent()->getScene(), collision, objFilePath);
}

void GazeboSimulatorBase::init(sdf::ElementPtr &worldPluginSDF)
{
	createExporters();

	initParameters();
	setUseParticleCaching(true);

	LOG_INFO << "SPlisHSPlasH version: " << SPLISHSPLASH_VERSION;
	LOG_DEBUG << "Git refspec:          " << GIT_REFSPEC;
	LOG_DEBUG << "Git SHA1:             " << GIT_SHA1;
	LOG_DEBUG << "Git status:           " << GIT_LOCAL_STATUS;
	LOG_DEBUG << "Host name:            " << SystemInfo::getHostName();

	

	//////////////////////////////////////////////////////////////////////////
	// read scene
	//////////////////////////////////////////////////////////////////////////
	m_sceneLoader = std::unique_ptr<GazeboSceneLoader>(new GazeboSceneLoader());
	const std::string &sceneFile = GazeboSceneConfiguration::getCurrent()->getSceneFile();
	GazeboSceneLoader::Scene &scene = GazeboSceneConfiguration::getCurrent()->getScene();
	m_sceneLoader->readScene(worldPluginSDF, scene);

	if (!getUseParticleCaching())
		LOG_INFO << "Boundary cache disabled.";
	m_outputPath = scene.outputPath;
	LOG_INFO << "Output directory: " << m_outputPath;
	//////////////////////////////////////////////////////////////////////////
	// init boundary simulation
	//////////////////////////////////////////////////////////////////////////
	m_isStaticScene = true;
	for (unsigned int i = 0; i < scene.boundaryModels.size(); i++)
	{
		if (scene.boundaryModels[i]->dynamic)
		{
			m_isStaticScene = false;
			break;
		}
	}

	LOG_INFO << "Initialize Gazebo boundary simulation";
	m_boundarySimulator = new GazeboBoundarySimulator(this);

	initExporters();
}

void GazeboSimulatorBase::initSimulation()
{
	const std::string &sceneFile = GazeboSceneConfiguration::getCurrent()->getSceneFile();
	const Utilities::GazeboSceneLoader::Scene &scene = GazeboSceneConfiguration::getCurrent()->getScene();
	Simulation *sim = Simulation::getCurrent();
	sim->setBoundaryHandlingMethod(BoundaryHandlingMethods::Akinci2012);
	sim->init(scene.particleRadius, false);
	sim->setSimulationMethod(4);

	buildModel();
	if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
	{
		unsigned int nBoundaryParticles = 0;
		for (unsigned int i = 0; i < sim->numberOfBoundaryModels(); i++)
			nBoundaryParticles += static_cast<BoundaryModel_Akinci2012 *>(sim->getBoundaryModel(i))->numberOfParticles();

		LOG_INFO << "Number of boundary particles: " << nBoundaryParticles;
	}
	readParameters();

	TimeStep *ts = sim->getTimeStep();
	LOG_INFO << "=== SPlisHSPlasH loaded parameters ===";
	LOG_INFO << "  particleRadius:         " << sim->getParticleRadius();
	LOG_INFO << "  timeStepSize:           " << TimeManager::getCurrent()->getTimeStepSize();
	LOG_INFO << "  simulationMethod:       " << sim->getSimulationMethod();
	LOG_INFO << "  boundaryHandlingMethod: " << static_cast<int>(sim->getBoundaryHandlingMethod());
	LOG_INFO << "  outputPath:             " << scene.outputPath;
	if (ts)
	{
		LOG_INFO << "  maxIterations:          " << ts->getValue<unsigned int>(TimeStep::MAX_ITERATIONS);
		LOG_INFO << "  maxError:               " << ts->getValue<Real>(TimeStep::MAX_ERROR);
	}
	LOG_INFO << "  fluidBlocks:            " << scene.fluidBlocks.size();
	for (unsigned int i = 0; i < scene.fluidBlocks.size(); i++)
	{
		LOG_INFO << "    [" << i << "] start: ["
				 << scene.fluidBlocks[i]->box.m_minX[0] << ", "
				 << scene.fluidBlocks[i]->box.m_minX[1] << ", "
				 << scene.fluidBlocks[i]->box.m_minX[2] << "]  end: ["
				 << scene.fluidBlocks[i]->box.m_maxX[0] << ", "
				 << scene.fluidBlocks[i]->box.m_maxX[1] << ", "
				 << scene.fluidBlocks[i]->box.m_maxX[2] << "]";
	}
	for (unsigned int i = 0; i < sim->numberOfFluidModels(); i++)
	{
		FluidModel *model = sim->getFluidModel(i);
		ViscosityBase *visc = model->getViscosityBase();
		LOG_INFO << "  fluidModel[" << i << "] id=" << model->getId()
				 << "  density0=" << model->getDensity0()
				 << "  viscosityMethod=" << model->getViscosityMethod()
				 << "  viscosity=" << (visc ? visc->getValue<Real>(ViscosityBase::VISCOSITY_COEFFICIENT) : 0.0)
				 << "  surfaceTensionMethod=" << model->getSurfaceTensionMethod()
				 << "  particles=" << model->numActiveParticles();
	}
	LOG_INFO << "  boundaryModels: " << scene.boundaryModels.size();
	for (unsigned int i = 0; i < scene.boundaryModels.size(); i++)
	{
		LOG_INFO << "    [" << i << "] dynamic=" << scene.boundaryModels[i]->dynamic
				 << "  file=" << scene.boundaryModels[i]->objFilePath;
	}
	LOG_INFO << "=== end SPlisHSPlasH parameters ===";
}

void GazeboSimulatorBase::initBoundaryData()
{
	m_boundarySimulator->initBoundaryData();
}

void GazeboSimulatorBase::cleanup()
{
	delete GazeboSceneConfiguration::getCurrent();
	delete Simulation::getCurrent();
}

void GazeboSimulatorBase::readParameters()
{
	m_sceneLoader->readParameterObject("Configuration", this);
	m_sceneLoader->readParameterObject("Configuration", Simulation::getCurrent());
	m_sceneLoader->readParameterObject("Configuration", Simulation::getCurrent()->getTimeStep());

	Simulation *sim = Simulation::getCurrent();
	const Utilities::GazeboSceneLoader::Scene &scene = GazeboSceneConfiguration::getCurrent()->getScene();
	for (unsigned int i = 0; i < sim->numberOfFluidModels(); i++)
	{
		FluidModel *model = sim->getFluidModel(i);
		const std::string &key = model->getId();
		m_sceneLoader->readParameterObject(key, model);
		m_sceneLoader->readParameterObject(key, (ParameterObject *)model->getDragBase());
		m_sceneLoader->readParameterObject(key, (ParameterObject *)model->getSurfaceTensionBase());
		m_sceneLoader->readParameterObject(key, (ParameterObject *)model->getViscosityBase());
		m_sceneLoader->readParameterObject(key, (ParameterObject *)model->getVorticityBase());
		m_sceneLoader->readParameterObject(key, (ParameterObject *)model->getElasticityBase());
	}
}

void GazeboSimulatorBase::cleanupExporters()
{
	for (size_t i = 0; i < m_particleExporters.size(); i++)
		delete m_particleExporters[i].m_exporter;
	m_particleExporters.clear();
	for (size_t i = 0; i < m_rbExporters.size(); i++)
		delete m_rbExporters[i].m_exporter;
	m_rbExporters.clear();
}

void GazeboSimulatorBase::initExporters()
{
	for (size_t i = 0; i < m_particleExporters.size(); i++)
		m_particleExporters[i].m_exporter->init(m_outputPath);
	for (size_t i = 0; i < m_rbExporters.size(); i++)
		m_rbExporters[i].m_exporter->init(m_outputPath);
	activateExporter("Partio Exporter", true);
	activateExporter("VTK Exporter", true);
	activateExporter("Rigid Body VTK Exporter", true);
}

void GazeboSimulatorBase::buildModel()
{
	const Utilities::GazeboSceneLoader::Scene &scene = GazeboSceneConfiguration::getCurrent()->getScene();
	TimeManager::getCurrent()->setTimeStepSize(scene.timeStepSize);

	initFluidData();

	//createEmitters();
	Simulation *sim = Simulation::getCurrent();

	if (sim->getTimeStep())
		sim->getTimeStep()->resize();

	if (!sim->is2DSimulation())
	{
		sim->setValue(Simulation::KERNEL_METHOD, Simulation::ENUM_KERNEL_PRECOMPUTED_CUBIC);
		sim->setValue(Simulation::GRAD_KERNEL_METHOD, Simulation::ENUM_GRADKERNEL_PRECOMPUTED_CUBIC);
	}
	else
	{
		sim->setValue(Simulation::KERNEL_METHOD, Simulation::ENUM_KERNEL_CUBIC_2D);
		sim->setValue(Simulation::GRAD_KERNEL_METHOD, Simulation::ENUM_GRADKERNEL_CUBIC_2D);
	}
}

void GazeboSimulatorBase::reset()
{
	Utilities::Timing::printAverageTimes();
	Utilities::Timing::reset();

	Utilities::Counting::printAverageCounts();
	Utilities::Counting::reset();

	Simulation::getCurrent()->reset();
#ifdef USE_DEBUG_TOOLS
	Simulation::getCurrent()->getDebugTools()->reset();
#endif

	m_boundarySimulator->reset();

	const Utilities::GazeboSceneLoader::Scene &scene = GazeboSceneConfiguration::getCurrent()->getScene();
	if (Simulation::getCurrent()->getValue<int>(Simulation::CFL_METHOD) != Simulation::ENUM_CFL_NONE)
		TimeManager::getCurrent()->setTimeStepSize(scene.timeStepSize);
	m_nextFrameTime = 0.0;
	m_nextFrameTimeState = 0.0;
	m_frameCounter = 1;
	m_isFirstFrame = true;
	m_isFirstFrameVTK = true;
#ifdef DL_OUTPUT
	m_nextTiming = 1.0;
#endif

	for (size_t i = 0; i < m_particleExporters.size(); i++)
		m_particleExporters[i].m_exporter->reset();
	for (size_t i = 0; i < m_rbExporters.size(); i++)
		m_rbExporters[i].m_exporter->reset();
}

bool GazeboSimulatorBase::timeStepNoGUI()
{
	if (m_simulation_steps == 0)
	{
		m_boundarySimulator->timeStep();
	}
	// Simulation code
	Simulation *sim = Simulation::getCurrent();

	START_TIMING("SimStep");
	Simulation::getCurrent()->getTimeStep()->step();
	STOP_TIMING_AVG;

	m_boundarySimulator->timeStep();

	step();

	INCREASE_COUNTER("Time step size", TimeManager::getCurrent()->getTimeStepSize());

#ifdef USE_DEBUG_TOOLS
	Simulation::getCurrent()->getDebugTools()->step();
#endif
	m_simulation_steps++;
	if (m_timeStepCB)
		m_timeStepCB();
	return true;
}

void GazeboSimulatorBase::loadObj(const std::string &filename, TriangleMesh &mesh, const Vector3r &scale)
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

void GazeboSimulatorBase::activateExporter(const std::string &exporterName, const bool active)
{
	for (auto i = 0; i < m_particleExporters.size(); i++)
	{
		if (exporterName == m_particleExporters[i].m_name)
		{
			m_particleExporters[i].m_exporter->setActive(active);
			return;
		}
	}
	for (auto i = 0; i < m_rbExporters.size(); i++)
	{
		if (exporterName == m_rbExporters[i].m_name)
		{
			m_rbExporters[i].m_exporter->setActive(active);
			return;
		}
	}
}

void GazeboSimulatorBase::initFluidData()
{
	LOG_INFO << "Initialize fluid particles";

	Simulation *sim = Simulation::getCurrent();
	const Utilities::GazeboSceneLoader::Scene &scene = GazeboSceneConfiguration::getCurrent()->getScene();

	//////////////////////////////////////////////////////////////////////////
	// Determine number of different fluid IDs
	//////////////////////////////////////////////////////////////////////////
	std::map<std::string, unsigned int> fluidIDs;
	unsigned int index = 0;
	for (unsigned int i = 0; i < scene.fluidBlocks.size(); i++)
	{
		if (fluidIDs.find(scene.fluidBlocks[i]->id) == fluidIDs.end())
			fluidIDs[scene.fluidBlocks[i]->id] = index++;
	}
	for (unsigned int i = 0; i < scene.fluidModels.size(); i++)
	{
		if (fluidIDs.find(scene.fluidModels[i]->id) == fluidIDs.end())
			fluidIDs[scene.fluidModels[i]->id] = index++;
	}
	for (unsigned int i = 0; i < scene.emitters.size(); i++)
	{
		if (fluidIDs.find(scene.emitters[i]->id) == fluidIDs.end())
			fluidIDs[scene.emitters[i]->id] = index++;
	}
	const unsigned int numberOfFluidModels = static_cast<unsigned int>(fluidIDs.size());

	std::vector<std::vector<Vector3r>> fluidParticles;
	std::vector<std::vector<Vector3r>> fluidVelocities;
	fluidParticles.resize(numberOfFluidModels);
	fluidVelocities.resize(numberOfFluidModels);

	createFluidBlocks(fluidIDs, fluidParticles, fluidVelocities);

	std::string base_path = scene.outputPath;

	const bool useCache = getUseParticleCaching();
	std::string scene_path = FileSystem::getFilePath(base_path);
	string cachePath = scene_path + "/Cache";

	unsigned int startIndex = 0;
	unsigned int endIndex = 0;

	for (unsigned int i = 0; i < scene.fluidModels.size(); i++)
	{
		const unsigned int fluidIndex = fluidIDs[scene.fluidModels[i]->id];

		std::string fileName;
		if (FileSystem::isRelativePath(scene.fluidModels[i]->samplesFile))
			fileName = base_path + "/" + scene.fluidModels[i]->samplesFile;
		else
			fileName = scene.fluidModels[i]->samplesFile;

		string ext = FileSystem::getFileExt(fileName);
		transform(ext.begin(), ext.end(), ext.begin(), ::toupper);
		if (ext == "OBJ")
		{
			// check if mesh file has changed
			std::string md5FileName = FileSystem::normalizePath(cachePath + "/" + FileSystem::getFileNameWithExt(fileName) + "_fluid.md5");
			bool md5 = false;
			if (useCache)
			{
				string md5Str = FileSystem::getFileMD5(fileName);
				if (FileSystem::fileExists(md5FileName))
					md5 = FileSystem::checkMD5(md5Str, md5FileName);
			}

			// Cache sampling
			std::string mesh_base_path = FileSystem::getFilePath(fileName);
			std::string mesh_file_name = FileSystem::getFileName(fileName);

			std::string mode = to_string(scene.fluidModels[i]->mode);
			const string scaleStr = real2String(scene.fluidModels[i]->scale[0]) + "_" + real2String(scene.fluidModels[i]->scale[1]) + "_" + real2String(scene.fluidModels[i]->scale[2]);
			const string resStr = to_string(scene.fluidModels[i]->resolutionSDF[0]) + "_" + to_string(scene.fluidModels[i]->resolutionSDF[1]) + "_" + to_string(scene.fluidModels[i]->resolutionSDF[2]);
			const string particleFileName = FileSystem::normalizePath(cachePath + "/" + mesh_file_name + "_fluid_" + real2String(scene.particleRadius) + "_m" + mode + "_s" + scaleStr + "_r" + resStr + ".bgeo");

			// check MD5 if cache file is available
			bool foundCacheFile = false;

			if (useCache)
				foundCacheFile = FileSystem::fileExists(particleFileName);

			if (useCache && foundCacheFile && md5)
			{
				PartioReaderWriter::readParticles(particleFileName, scene.fluidModels[i]->translation, scene.fluidModels[i]->rotation, 1.0, fluidParticles[fluidIndex], fluidVelocities[fluidIndex]);
				LOG_INFO << "Loaded cached fluid sampling: " << particleFileName;
			}

			if (!useCache || !foundCacheFile || !md5)
			{
				LOG_INFO << "Volume sampling of " << fileName;

				TriangleMesh mesh;
				loadObj(fileName, mesh, scene.fluidModels[i]->scale);

				bool invert = scene.fluidModels[i]->invert;
				int mode = scene.fluidModels[i]->mode;
				std::array<unsigned int, 3> resolutionSDF = scene.fluidModels[i]->resolutionSDF;

				LOG_INFO << "SDF resolution: " << resolutionSDF[0] << ", " << resolutionSDF[1] << ", " << resolutionSDF[2];

				const unsigned int size_before_sampling = fluidParticles[fluidIndex].size();
				START_TIMING("Volume sampling");
				Utilities::VolumeSampling::sampleMesh(mesh.numVertices(), mesh.getVertices().data(), mesh.numFaces(), mesh.getFaces().data(),
													  scene.particleRadius, nullptr, resolutionSDF, invert, mode, fluidParticles[fluidIndex]);
				STOP_TIMING_AVG;
				const unsigned int size_after_sampling = fluidParticles[fluidIndex].size();

				fluidVelocities[fluidIndex].resize(fluidParticles[fluidIndex].size(), scene.fluidModels[i]->initialVelocity);

				// Cache sampling
				if (useCache && (FileSystem::makeDir(cachePath) == 0))
				{
					LOG_INFO << "Save particle sampling: " << particleFileName;
					PartioReaderWriter::writeParticles(particleFileName, size_after_sampling - size_before_sampling, &fluidParticles[fluidIndex][size_before_sampling], &fluidVelocities[fluidIndex][size_before_sampling], 0.0);
					// PartioReaderWriter::writeParticles(particleFileName, (unsigned int)fluidParticles[fluidIndex].size(), fluidParticles[fluidIndex].data(), fluidVelocities[fluidIndex].data(), 0.0);
					FileSystem::writeMD5File(fileName, md5FileName);
				}

				// transform particles
				for (unsigned int j = size_before_sampling; j < size_after_sampling; j++)
					fluidParticles[fluidIndex][j] = scene.fluidModels[i]->rotation * fluidParticles[fluidIndex][j] + scene.fluidModels[i]->translation;
			}
		}
		else
		{
			if (!PartioReaderWriter::readParticles(fileName, scene.fluidModels[i]->translation, scene.fluidModels[i]->rotation, scene.fluidModels[i]->scale[0], fluidParticles[fluidIndex], fluidVelocities[fluidIndex]))
				LOG_ERR << "File not found: " << fileName;
		}
		Simulation::getCurrent()->setValue(Simulation::PARTICLE_RADIUS, scene.particleRadius);
	}

	unsigned int nParticles = 0;
	for (auto it = fluidIDs.begin(); it != fluidIDs.end(); it++)
	{
		const unsigned int index = it->second;

		unsigned int maxEmitterParticles = 1000;
		//m_sceneLoader->readValue(it->first, "maxEmitterParticles", maxEmitterParticles);
		sim->addFluidModel(it->first, (unsigned int)fluidParticles[index].size(), fluidParticles[index].data(), fluidVelocities[index].data(), maxEmitterParticles);
		nParticles += (unsigned int)fluidParticles[index].size();
	}

	LOG_INFO << "Number of fluid particles: " << nParticles;
}

void GazeboSimulatorBase::createFluidBlocks(std::map<std::string, unsigned int> &fluidIDs, std::vector<std::vector<Vector3r>> &fluidParticles, std::vector<std::vector<Vector3r>> &fluidVelocities)
{
	const Utilities::GazeboSceneLoader::Scene &scene = GazeboSceneConfiguration::getCurrent()->getScene();
	for (unsigned int i = 0; i < scene.fluidBlocks.size(); i++)
	{
		const unsigned int fluidIndex = fluidIDs[scene.fluidBlocks[i]->id];
		const Real diam = static_cast<Real>(2.0) * scene.particleRadius;

		Real xshift = diam;
		Real yshift = diam;
		const Real eps = static_cast<Real>(1.0e-9);
		if (scene.fluidBlocks[i]->mode == 1)
			yshift = sqrt(static_cast<Real>(3.0)) * scene.particleRadius + eps;
		else if (scene.fluidBlocks[i]->mode == 2)
		{
			xshift = sqrt(static_cast<Real>(6.0)) * diam / static_cast<Real>(3.0) + eps;
			yshift = sqrt(static_cast<Real>(3.0)) * scene.particleRadius + eps;
		}

		Vector3r diff = scene.fluidBlocks[i]->box.m_maxX - scene.fluidBlocks[i]->box.m_minX;
		if (scene.fluidBlocks[i]->mode == 1)
		{
			diff[0] -= diam;
			diff[2] -= diam;
		}
		else if (scene.fluidBlocks[i]->mode == 2)
		{
			diff[0] -= xshift;
			diff[2] -= diam;
		}

		const int stepsX = (int)round(diff[0] / xshift) - 1;
		const int stepsY = (int)round(diff[1] / yshift) - 1;
		int stepsZ = (int)round(diff[2] / diam) - 1;

		Vector3r start = scene.fluidBlocks[i]->box.m_minX + static_cast<Real>(2.0) * scene.particleRadius * Vector3r::Ones();
		fluidParticles[fluidIndex].reserve(fluidParticles[fluidIndex].size() + stepsX * stepsY * stepsZ);
		fluidVelocities[fluidIndex].resize(fluidVelocities[fluidIndex].size() + stepsX * stepsY * stepsZ, scene.fluidBlocks[i]->initialVelocity);

		if (Simulation::getCurrent()->is2DSimulation())
		{
			stepsZ = 1;
			start[2] = 0.0;
		}

		for (int j = 0; j < stepsX; j++)
		{
			for (int k = 0; k < stepsY; k++)
			{
				for (int l = 0; l < stepsZ; l++)
				{
					Vector3r currPos = Vector3r(j * xshift, k * yshift, l * diam) + start;
					if (scene.fluidBlocks[i]->mode == 1)
					{
						if (k % 2 == 0)
							currPos += Vector3r(0, 0, scene.particleRadius);
						else
							currPos += Vector3r(scene.particleRadius, 0, 0);
					}
					else if (scene.fluidBlocks[i]->mode == 2)
					{
						currPos += Vector3r(0, 0, scene.particleRadius);

						Vector3r shift_vec(0, 0, 0);
						if ((j % 2) && !Simulation::getCurrent()->is2DSimulation())
						{
							shift_vec[2] += diam / (static_cast<Real>(2.0) * (k % 2 ? -1 : 1));
						}
						if (k % 2 == 0)
						{
							shift_vec[0] += xshift / static_cast<Real>(2.0);
						}
						currPos += shift_vec;
					}
					fluidParticles[fluidIndex].push_back(currPos);
				}
			}
		}
	}
}

void GazeboSimulatorBase::particleInfo(std::vector<std::vector<unsigned int>> &particles)
{
	Simulation *sim = Simulation::getCurrent();
	const int maxWidth = 25;
	for (unsigned int i = 0; i < sim->numberOfFluidModels(); i++)
	{
		FluidModel *model = sim->getFluidModel(i);
		if (particles[i].size() > 0)
		{
			LOG_INFO << "---------------------------------------------------------------------------";
			LOG_INFO << model->getId();
			LOG_INFO << "---------------------------------------------------------------------------";
		}
		for (unsigned int j = 0; j < particles[i].size(); j++)
		{
			unsigned int index = particles[i][j];
			LOG_INFO << std::left << std::setw(maxWidth) << std::setfill(' ') << "Index:" << index;
			for (unsigned int k = 0; k < model->numberOfFields(); k++)
			{
				const FieldDescription &field = model->getField(k);
				if (field.type == Scalar)
					LOG_INFO << std::left << std::setw(maxWidth) << std::setfill(' ') << field.name + ":" << *((Real *)field.getFct(index));
				else if (field.type == UInt)
					LOG_INFO << std::left << std::setw(maxWidth) << std::setfill(' ') << field.name + ":" << *((unsigned int *)field.getFct(index));
				else if (field.type == Vector3)
				{
					Eigen::Map<Vector3r> vec((Real *)field.getFct(index));
					LOG_INFO << std::left << std::setw(maxWidth) << std::setfill(' ') << field.name + ":" << vec.transpose();
				}
				else if (field.type == Vector6)
				{
					Eigen::Map<Vector6r> vec((Real *)field.getFct(index));
					LOG_INFO << std::left << std::setw(maxWidth) << std::setfill(' ') << field.name + ":" << vec.transpose();
				}
				else if (field.type == Matrix3)
				{
					Eigen::Map<Matrix3r> mat((Real *)field.getFct(index));
					LOG_INFO << std::left << std::setw(maxWidth) << std::setfill(' ') << field.name + ":" << mat.row(0);
					LOG_INFO << std::left << std::setw(maxWidth) << std::setfill(' ') << " " << mat.row(1);
					LOG_INFO << std::left << std::setw(maxWidth) << std::setfill(' ') << " " << mat.row(2);
				}
				else if (field.type == Matrix6)
				{
					Eigen::Map<Matrix6r> mat((Real *)field.getFct(index));
					LOG_INFO << std::left << std::setw(maxWidth) << std::setfill(' ') << field.name + ":" << mat.row(0);
					for (unsigned int k = 1; k < 6; k++)
						LOG_INFO << std::left << std::setw(maxWidth) << std::setfill(' ') << " " << mat.row(k);
				}
			}
			LOG_INFO << "---------------------------------------------------------------------------\n";
		}
	}
}

void GazeboSimulatorBase::step()
{
	setInitialRboVertices(getInitialRboVertices());
	if (TimeManager::getCurrent()->getTime() >= m_nextFrameTime)
	{
		m_nextFrameTime += static_cast<Real>(1.0) / m_framesPerSecond;
		for (size_t i = 0; i < m_particleExporters.size(); i++)
		{
			m_particleExporters[i].m_exporter->step(m_frameCounter);
		}
		/* for (size_t i = 0; i < m_rbExporters.size(); i++)
			m_rbExporters[i].m_exporter->step(m_frameCounter); */

		this->writeRigidBodies(m_frameCounter);
		m_frameCounter++;
	}
	/*if (TimeManager::getCurrent()->getTime() >= m_nextFrameTimeState)
	{
		m_nextFrameTimeState += static_cast<Real>(1.0) / m_framesPerSecondState;
		if (m_enableStateExport)
			saveState();
	} */
}

void GazeboSimulatorBase::updateBoundaryParticles(const bool forceUpdate = false)
{
	Simulation *sim = Simulation::getCurrent();
	const Utilities::GazeboSceneLoader::Scene &scene = GazeboSceneConfiguration::getCurrent()->getScene();

	const unsigned int nObjects = sim->numberOfBoundaryModels();
	for (unsigned int i = 0; i < nObjects; i++)
	{
		BoundaryModel_Akinci2012 *bm = static_cast<BoundaryModel_Akinci2012 *>(sim->getBoundaryModel(i));
		GazeboRigidBody *rbo = dynamic_cast<GazeboRigidBody *>(bm->getRigidBodyObject());
		if (rbo->isDynamic() || forceUpdate)
		{
			#pragma omp parallel default(shared)
			{
				physics::CollisionPtr gazeboCollision = rbo->getGazeboCollision();
			
				// Position of rigid body
				ignition::math::Pose3d gazeboRigidBodyPose = gazeboCollision->WorldPose();
				Vector3r fluidObjectPosition = Vector3r(gazeboRigidBodyPose.Pos().X(), gazeboRigidBodyPose.Pos().Y(), gazeboRigidBodyPose.Pos().Z());
			
				// Rotation of rigid body
				ignition::math::Matrix3d rigidBodyRotation = ignition::math::Matrix3d(gazeboRigidBodyPose.Rot());
				Matrix3r fluidObjectRotation;
				fluidObjectRotation << rigidBodyRotation(0, 0), rigidBodyRotation(0, 1), rigidBodyRotation(0, 2),
					rigidBodyRotation(1, 0), rigidBodyRotation(1, 1), rigidBodyRotation(1, 2),
					rigidBodyRotation(2, 0), rigidBodyRotation(2, 1), rigidBodyRotation(2, 2);
				// Linear velocity of rigid body
				ignition::math::Vector3d linearVelocityGazebo = gazeboCollision->WorldLinearVel();
				Vector3r fluidObjectLinearVel = Vector3r(linearVelocityGazebo.X(), linearVelocityGazebo.Y(), linearVelocityGazebo.Z());

				// Angular velocity of rigid body
				ignition::math::Vector3d angularVelocityGazebo = gazeboCollision->WorldAngularVel();
				Vector3r fluidObjectAngularVel = Vector3r(angularVelocityGazebo.X(), angularVelocityGazebo.Y(), angularVelocityGazebo.Z());
				for (int j = 0; j < (int)bm->numberOfParticles(); j++)
				{
					bm->getPosition(j) = fluidObjectRotation * bm->getPosition0(j) + fluidObjectPosition;
					if (rbo->isDynamic())
						bm->getVelocity(j) = fluidObjectAngularVel.cross(bm->getPosition(j) - fluidObjectPosition) + fluidObjectLinearVel;
					else
						bm->getVelocity(j).setZero();
				}
				auto CoMPose = gazeboCollision->GetLink()->WorldInertialPose().Pos();
				rbo->setPosition(Vector3r(CoMPose.X(),CoMPose.Y(),CoMPose.Z()));
				rbo->setRotation(fluidObjectRotation);
			}
#ifdef GPU_NEIGHBORHOOD_SEARCH
			// copy the particle data to the GPU
			if (forceUpdate)
				sim->getNeighborhoodSearch()->update_point_sets();
#endif
		}
	}
}

void SPH::GazeboSimulatorBase::updateDMVelocity()
{
	Simulation *sim = Simulation::getCurrent();
	const Utilities::GazeboSceneLoader::Scene &scene = GazeboSceneConfiguration::getCurrent()->getScene();

	const unsigned int nObjects = sim->numberOfBoundaryModels();
	for (unsigned int i = 0; i < nObjects; i++)
	{
		BoundaryModel_Koschier2017 *bm = static_cast<BoundaryModel_Koschier2017 *>(sim->getBoundaryModel(i));
		RigidBodyObject *rbo = bm->getRigidBodyObject();
		if (rbo->isDynamic())
		{
			const Real maxDist = bm->getMaxDist();
			const Vector3r x(maxDist, 0.0, 0.0);
			const Vector3r vel = rbo->getAngularVelocity().cross(x) + rbo->getVelocity();
			bm->setMaxVel(vel.norm());
		}
	}
}

void SPH::GazeboSimulatorBase::updateVMVelocity()
{
	Simulation *sim = Simulation::getCurrent();
	const Utilities::GazeboSceneLoader::Scene &scene = GazeboSceneConfiguration::getCurrent()->getScene();
	const unsigned int nObjects = sim->numberOfBoundaryModels();
	for (unsigned int i = 0; i < nObjects; i++)
	{
		BoundaryModel_Bender2019 *bm = static_cast<BoundaryModel_Bender2019 *>(sim->getBoundaryModel(i));
		RigidBodyObject *rbo = bm->getRigidBodyObject();
		if (rbo->isDynamic())
		{
			const Real maxDist = bm->getMaxDist();
			const Vector3r x(maxDist, 0.0, 0.0);
			const Vector3r vel = rbo->getAngularVelocity().cross(x) + rbo->getVelocity();
			bm->setMaxVel(vel.norm());
		}
	}
}

std::string GazeboSimulatorBase::real2String(const Real r)
{
	string str = to_string(r);
	str.erase(str.find_last_not_of('0') + 1, std::string::npos);
	str.erase(str.find_last_not_of('.') + 1, std::string::npos);
	return str;
}

void SPH::GazeboSimulatorBase::saveState(const std::string &stateFile)
{
	std::string stateFilePath;
	std::string exportFileName;
	const Real time = TimeManager::getCurrent()->getTime();
	const std::string timeStr = real2String(time);
	if (stateFile == "")
	{
		stateFilePath = FileSystem::normalizePath(m_outputPath + "/state");
		exportFileName = FileSystem::normalizePath(stateFilePath + "/state_" + timeStr);
	}
	else
	{
		stateFilePath = FileSystem::getFilePath(stateFile);
		exportFileName = FileSystem::normalizePath(stateFilePath + "/" + FileSystem::getFileName(stateFile));
	}
	FileSystem::makeDirs(stateFilePath);

	const std::string &sceneFile = GazeboSceneConfiguration::getCurrent()->getSceneFile();
	string md5Str = FileSystem::getFileMD5(sceneFile);

	Simulation *sim = Simulation::getCurrent();

	// Save additional data
	BinaryFileWriter binWriter;
	binWriter.openFile(exportFileName + ".bin");
	binWriter.write(md5Str);

	binWriter.write(m_nextFrameTime);
	binWriter.write(m_nextFrameTimeState);
	binWriter.write(m_frameCounter);
	binWriter.write(m_isFirstFrame);
	binWriter.write(m_isFirstFrameVTK);

	writeParameterState(binWriter);
	TimeManager::getCurrent()->saveState(binWriter);
	Simulation::getCurrent()->saveState(binWriter);

	// fluid models
	for (unsigned int i = 0; i < sim->numberOfFluidModels(); i++)
	{
		FluidModel *model = sim->getFluidModel(i);
		std::string fileName = "particle";
		fileName = fileName + "_" + model->getId(); // +"_" + std::to_string(m_frameCounter);

		// Save particle data
		std::string expFileName = FileSystem::normalizePath(exportFileName + "_" + fileName);
		writeFluidParticlesState(expFileName + ".bgeo", model);
	}

	// boundary models
	if (m_firstState)
	{
		for (unsigned int i = 0; i < sim->numberOfBoundaryModels(); i++)
		{
			BoundaryModel *model = sim->getBoundaryModel(i);
			std::string fileName = "boundary";
			fileName = fileName + "_" + to_string(i); // +"_" + std::to_string(m_frameCounter);

			// Save particle data
			std::string expFileName = FileSystem::normalizePath(exportFileName + "_" + fileName);
			writeBoundaryState(expFileName + ".bgeo", model);
		}
		m_firstState = false;
	}

	// dynamic bodies
	for (unsigned int i = 0; i < sim->numberOfBoundaryModels(); i++)
	{
		BoundaryModel *bm = sim->getBoundaryModel(i);
		if (bm->getRigidBodyObject()->isDynamic())
		{
			binWriter.writeMatrix(bm->getRigidBodyObject()->getPosition());
			binWriter.writeMatrix(bm->getRigidBodyObject()->getRotation());
			binWriter.writeMatrix(bm->getRigidBodyObject()->getVelocity());
			binWriter.writeMatrix(bm->getRigidBodyObject()->getAngularVelocity());
		}
	}
	binWriter.closeFile();

	LOG_INFO << "Saved state: " << exportFileName + ".bin";
}

#ifdef WIN32
void SPH::GazeboSimulatorBase::loadStateDialog()
{
	const std::string stateFilePath = FileSystem::normalizePath(m_outputPath + "/state");
	const std::string stateFileName = FileSystem::fileDialog(0, stateFilePath, "*.bin");
	if (stateFileName == "")
		return;
	loadState(stateFileName);
}
#endif

void GazeboSimulatorBase::writeParameterState(BinaryFileWriter &binWriter)
{
	writeParameterObjectState(binWriter, this);
	writeParameterObjectState(binWriter, Simulation::getCurrent());
	writeParameterObjectState(binWriter, Simulation::getCurrent()->getTimeStep());
#ifdef USE_DEBUG_TOOLS
	writeParameterObjectState(binWriter, Simulation::getCurrent()->getDebugTools());
#endif

	Simulation *sim = Simulation::getCurrent();
	for (unsigned int i = 0; i < sim->numberOfFluidModels(); i++)
	{
		FluidModel *model = sim->getFluidModel(i);
		writeParameterObjectState(binWriter, model);
		writeParameterObjectState(binWriter, (ParameterObject *)model->getDragBase());
		writeParameterObjectState(binWriter, (ParameterObject *)model->getSurfaceTensionBase());
		writeParameterObjectState(binWriter, (ParameterObject *)model->getViscosityBase());
		writeParameterObjectState(binWriter, (ParameterObject *)model->getVorticityBase());
		writeParameterObjectState(binWriter, (ParameterObject *)model->getElasticityBase());
	}
}

void GazeboSimulatorBase::writeParameterObjectState(BinaryFileWriter &binWriter, GenParam::ParameterObject *paramObj)
{
	if (paramObj == nullptr)
		return;

	const unsigned int numParams = paramObj->numParameters();

	for (unsigned int i = 0; i < numParams; i++)
	{
		ParameterBase *paramBase = paramObj->getParameter(i);

		if (paramBase->getType() == RealParameterType)
			binWriter.write(static_cast<NumericParameter<Real> *>(paramBase)->getValue());
		else if (paramBase->getType() == ParameterBase::UINT32)
			binWriter.write(static_cast<NumericParameter<unsigned int> *>(paramBase)->getValue());
		else if (paramBase->getType() == ParameterBase::UINT16)
			binWriter.write(static_cast<NumericParameter<unsigned short> *>(paramBase)->getValue());
		else if (paramBase->getType() == ParameterBase::UINT8)
			binWriter.write(static_cast<NumericParameter<unsigned char> *>(paramBase)->getValue());
		else if (paramBase->getType() == ParameterBase::INT32)
			binWriter.write(static_cast<NumericParameter<int> *>(paramBase)->getValue());
		else if (paramBase->getType() == ParameterBase::INT16)
			binWriter.write(static_cast<NumericParameter<short> *>(paramBase)->getValue());
		else if (paramBase->getType() == ParameterBase::INT8)
			binWriter.write(static_cast<NumericParameter<char> *>(paramBase)->getValue());
		else if (paramBase->getType() == ParameterBase::ENUM)
			binWriter.write(static_cast<EnumParameter *>(paramBase)->getValue());
		else if (paramBase->getType() == ParameterBase::BOOL)
			binWriter.write(static_cast<BoolParameter *>(paramBase)->getValue());
		else if (paramBase->getType() == RealVectorParameterType)
		{
			VectorParameter<Real> *vec = static_cast<VectorParameter<Real> *>(paramBase);
			binWriter.writeBuffer((char *)vec->getValue(), vec->getDim() * sizeof(Real));
			;
		}
		else if (paramBase->getType() == ParameterBase::STRING)
			binWriter.write(static_cast<StringParameter *>(paramBase)->getValue());
	}
}

void GazeboSimulatorBase::readParameterState(BinaryFileReader &binReader)
{
	readParameterObjectState(binReader, this);
	readParameterObjectState(binReader, Simulation::getCurrent());
	readParameterObjectState(binReader, Simulation::getCurrent()->getTimeStep());
#ifdef USE_DEBUG_TOOLS
	readParameterObjectState(binReader, Simulation::getCurrent()->getDebugTools());
#endif

	Simulation *sim = Simulation::getCurrent();
	for (unsigned int i = 0; i < sim->numberOfFluidModels(); i++)
	{
		FluidModel *model = sim->getFluidModel(i);
		readParameterObjectState(binReader, model);
		readParameterObjectState(binReader, (ParameterObject *)model->getDragBase());
		readParameterObjectState(binReader, (ParameterObject *)model->getSurfaceTensionBase());
		readParameterObjectState(binReader, (ParameterObject *)model->getViscosityBase());
		readParameterObjectState(binReader, (ParameterObject *)model->getVorticityBase());
		readParameterObjectState(binReader, (ParameterObject *)model->getElasticityBase());

		std::string field;
		binReader.read(field);

		int type;
		binReader.read(type);
	}
}

void GazeboSimulatorBase::readParameterObjectState(BinaryFileReader &binReader, GenParam::ParameterObject *paramObj)
{
	if (paramObj == nullptr)
		return;

	const unsigned int numParams = paramObj->numParameters();

	for (unsigned int i = 0; i < numParams; i++)
	{
		ParameterBase *paramBase = paramObj->getParameter(i);

		if (paramBase->getType() == RealParameterType)
		{
			Real val;
			binReader.read(val);
			static_cast<NumericParameter<Real> *>(paramBase)->setValue(val);
		}
		else if (paramBase->getType() == ParameterBase::UINT32)
		{
			unsigned int val;
			binReader.read(val);
			static_cast<NumericParameter<unsigned int> *>(paramBase)->setValue(val);
		}
		else if (paramBase->getType() == ParameterBase::UINT16)
		{
			unsigned short val;
			binReader.read(val);
			static_cast<NumericParameter<unsigned short> *>(paramBase)->setValue(val);
		}
		else if (paramBase->getType() == ParameterBase::UINT8)
		{
			unsigned char val;
			binReader.read(val);
			static_cast<NumericParameter<unsigned char> *>(paramBase)->setValue(val);
		}
		else if (paramBase->getType() == ParameterBase::INT32)
		{
			int val;
			binReader.read(val);
			static_cast<NumericParameter<int> *>(paramBase)->setValue(val);
		}
		else if (paramBase->getType() == ParameterBase::INT16)
		{
			short val;
			binReader.read(val);
			static_cast<NumericParameter<short> *>(paramBase)->setValue(val);
		}
		else if (paramBase->getType() == ParameterBase::INT8)
		{
			char val;
			binReader.read(val);
			static_cast<NumericParameter<char> *>(paramBase)->setValue(val);
		}
		else if (paramBase->getType() == ParameterBase::ENUM)
		{
			int val;
			binReader.read(val);
			static_cast<EnumParameter *>(paramBase)->setValue(val);
		}
		else if (paramBase->getType() == ParameterBase::BOOL)
		{
			bool val;
			binReader.read(val);
			static_cast<BoolParameter *>(paramBase)->setValue(val);
		}
		else if (paramBase->getType() == RealVectorParameterType)
		{
			VectorParameter<Real> *vec = static_cast<VectorParameter<Real> *>(paramBase);
			binReader.readBuffer((char *)vec->getValue(), vec->getDim() * sizeof(Real));
			;
		}
		else if (paramBase->getType() == ParameterBase::STRING)
		{
			std::string val;
			binReader.read(val);
			static_cast<StringParameter *>(paramBase)->setValue(val);
		}
	}
}

void GazeboSimulatorBase::writeFluidParticlesState(const std::string &fileName, FluidModel *model)
{
	Partio::ParticlesDataMutable &particleData = *Partio::create();

	std::vector<std::pair<unsigned int, Partio::ParticleAttribute>> partioAttr;
	for (unsigned int j = 0; j < model->numberOfFields(); j++)
	{
		const FieldDescription &field = model->getField(j);
		if (field.storeData)
		{
			//			LOG_INFO << "Store field: " << field.name;
			if (field.type == Scalar)
			{
				partioAttr.push_back({j, particleData.addAttribute(field.name.c_str(), Partio::FLOAT, 1)});
			}
			else if (field.type == UInt)
			{
				partioAttr.push_back({j, particleData.addAttribute(field.name.c_str(), Partio::INT, 1)});
			}
			else if (field.type == Vector3)
			{
				partioAttr.push_back({j, particleData.addAttribute(field.name.c_str(), Partio::VECTOR, 3)});
			}
			else
			{
				LOG_WARN << "Only scalar and vector fields are currently supported by the partio exporter.";
			}
		}
	}

	const unsigned int numParticles = model->numActiveParticles();

	for (unsigned int i = 0; i < numParticles; i++)
	{
		Partio::ParticleIndex index = particleData.addParticle();

		for (unsigned int j = 0; j < partioAttr.size(); j++)
		{
			const unsigned int fieldIndex = partioAttr[j].first;
			const Partio::ParticleAttribute &attr = partioAttr[j].second;

			const FieldDescription &field = model->getField(fieldIndex);
			if (field.type == FieldType::Scalar)
			{
				float *val = particleData.dataWrite<float>(attr, index);
				*val = (float)*((Real *)field.getFct(i));
			}
			else if (field.type == FieldType::UInt)
			{
				int *val = particleData.dataWrite<int>(attr, index);
				*val = (int)*((unsigned int *)field.getFct(i));
			}
			else if (field.type == FieldType::Vector3)
			{
				float *val = particleData.dataWrite<float>(attr, index);
				Eigen::Map<Vector3r> vec((Real *)field.getFct(i));
				val[0] = (float)vec[0];
				val[1] = (float)vec[1];
				val[2] = (float)vec[2];
			}
		}
	}

	Partio::write(fileName.c_str(), particleData, true);
	particleData.release();
}

void GazeboSimulatorBase::readFluidParticlesState(const std::string &fileName, FluidModel *model)
{
	if (!FileSystem::fileExists(fileName))
	{
		LOG_WARN << "File " << fileName << " does not exist.";
		return;
	}

	Partio::ParticlesDataMutable *data = Partio::read(fileName.c_str());
	if (!data)
	{
		LOG_WARN << "Partio file " << fileName << " not readable.";
		return;
	}

	std::vector<std::pair<unsigned int, Partio::ParticleAttribute>> partioAttr;
	for (int i = 0; i < data->numAttributes(); i++)
	{
		Partio::ParticleAttribute attr;
		data->attributeInfo(i, attr);
		for (unsigned int j = 0; j < model->numberOfFields(); j++)
		{
			const FieldDescription &field = model->getField(j);
			if (field.name == attr.name)
			{
				//LOG_INFO << "Read field: " << field.name;
				partioAttr.push_back({j, attr});
			}
		}
	}

	for (unsigned int j = 0; j < partioAttr.size(); j++)
	{
		const unsigned int fieldIndex = partioAttr[j].first;
		const Partio::ParticleAttribute &attr = partioAttr[j].second;

		const FieldDescription &field = model->getField(fieldIndex);

		for (int i = 0; i < data->numParticles(); i++)
		{
			if (field.type == FieldType::Scalar)
			{
				const float *value = data->data<float>(attr, i);
				*((Real *)field.getFct(i)) = value[0];
			}
			else if (field.type == FieldType::UInt)
			{
				const int *value = data->data<int>(attr, i);
				*((unsigned int *)field.getFct(i)) = value[0];
			}
			else if (field.type == FieldType::Vector3)
			{
				const float *value = data->data<float>(attr, i);
				Eigen::Map<Vector3r> vec((Real *)field.getFct(i));
				vec[0] = value[0];
				vec[1] = value[1];
				vec[2] = value[2];
			}
		}
	}
	data->release();
}

void GazeboSimulatorBase::writeBoundaryState(const std::string &fileName, BoundaryModel *bm)
{
	Simulation *sim = Simulation::getCurrent();
	if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
	{
		BoundaryModel_Akinci2012 *model = static_cast<BoundaryModel_Akinci2012 *>(bm);
		Partio::ParticlesDataMutable &particleData = *Partio::create();
		const Partio::ParticleAttribute &attrX0 = particleData.addAttribute("position0", Partio::VECTOR, 3);
		const Partio::ParticleAttribute &attrX = particleData.addAttribute("position", Partio::VECTOR, 3);
		const Partio::ParticleAttribute &attrVel = particleData.addAttribute("velocity", Partio::VECTOR, 3);
		const Partio::ParticleAttribute &attrVol = particleData.addAttribute("volume", Partio::FLOAT, 1);

		const unsigned int numParticles = model->numberOfParticles();

		for (unsigned int i = 0; i < numParticles; i++)
		{
			Partio::ParticleIndex index = particleData.addParticle();

			float *val = particleData.dataWrite<float>(attrX0, index);
			const Vector3r &x0 = model->getPosition0(i);
			val[0] = (float)x0[0];
			val[1] = (float)x0[1];
			val[2] = (float)x0[2];

			val = particleData.dataWrite<float>(attrX, index);
			const Vector3r &x = model->getPosition(i);
			val[0] = (float)x[0];
			val[1] = (float)x[1];
			val[2] = (float)x[2];

			val = particleData.dataWrite<float>(attrVel, index);
			const Vector3r &v = model->getVelocity(i);
			val[0] = (float)v[0];
			val[1] = (float)v[1];
			val[2] = (float)v[2];

			val = particleData.dataWrite<float>(attrVol, index);
			val[0] = (float)model->getVolume(i);
		}

		Partio::write(fileName.c_str(), particleData, true);
		particleData.release();
	}
}

void GazeboSimulatorBase::readBoundaryState(const std::string &fileName, BoundaryModel *bm)
{
	Simulation *sim = Simulation::getCurrent();
	if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
	{
		if (!FileSystem::fileExists(fileName))
		{
			LOG_WARN << "File " << fileName << " does not exist.";
			return;
		}

		BoundaryModel_Akinci2012 *model = static_cast<BoundaryModel_Akinci2012 *>(bm);
		Partio::ParticlesDataMutable *data = Partio::read(fileName.c_str());
		if (!data)
		{
			LOG_WARN << "Partio file " << fileName << " not readable.";
			return;
		}

		unsigned int pos0Index = 0xffffffff;
		unsigned int posIndex = 0xffffffff;
		unsigned int velIndex = 0xffffffff;
		unsigned int volIndex = 0xffffffff;

		for (int i = 0; i < data->numAttributes(); i++)
		{
			Partio::ParticleAttribute attr;
			data->attributeInfo(i, attr);
			if (attr.name == "position0")
				pos0Index = i;
			else if (attr.name == "position")
				posIndex = i;
			else if (attr.name == "velocity")
				velIndex = i;
			else if (attr.name == "volume")
				volIndex = i;
		}

		if ((pos0Index == 0xffffffff) ||
			(posIndex == 0xffffffff) ||
			(velIndex == 0xffffffff) ||
			(volIndex == 0xffffffff))
		{
			LOG_WARN << "File " << fileName << " does not has the correct attributes.";
			return;
		}

		Partio::ParticleAttribute attrX0;
		Partio::ParticleAttribute attrX;
		Partio::ParticleAttribute attrVel;
		Partio::ParticleAttribute attrVol;

		data->attributeInfo(pos0Index, attrX0);
		data->attributeInfo(posIndex, attrX);
		data->attributeInfo(velIndex, attrVel);
		data->attributeInfo(volIndex, attrVol);

		model->resize(data->numParticles());
		for (int i = 0; i < data->numParticles(); i++)
		{
			const float *pos0 = data->data<float>(attrX0, i);
			model->getPosition0(i) = Vector3r(pos0[0], pos0[1], pos0[2]);

			const float *pos = data->data<float>(attrX, i);
			model->getPosition(i) = Vector3r(pos[0], pos[1], pos[2]);

			const float *vel = data->data<float>(attrVel, i);
			model->getVelocity(i) = Vector3r(vel[0], vel[1], vel[2]);

			const float *vol = data->data<float>(attrVol, i);
			model->setVolume(i, vol[0]);
		}

		data->release();

		NeighborhoodSearch *neighborhoodSearch = Simulation::getCurrent()->getNeighborhoodSearch();
		neighborhoodSearch->update_point_sets();
		neighborhoodSearch->resize_point_set(model->getPointSetIndex(), &model->getPosition(0)[0], model->numberOfParticles());
	}
}

void GazeboSimulatorBase::initRbVertixPositions()
{
	Simulation *sim = Simulation::getCurrent();
	const Utilities::GazeboSceneLoader::Scene &scene = GazeboSceneConfiguration::getCurrent()->getScene();

	// give the function: sim
	m_vertixPositionsRbo_LF.resize(sim->numberOfBoundaryModels());

	for (unsigned int i = 0; i < sim->numberOfBoundaryModels(); i++)
	{
		SPH::BoundaryModel *bm = sim->getBoundaryModel(i);

		std::vector<Vector3r> verticesInOneRbo = bm->getRigidBodyObject()->getVertices();
		int n_vertices = (int)verticesInOneRbo.size();
		m_vertixPositionsRbo_LF[i].resize(n_vertices);

		ignition::math::Matrix3d rotation = ignition::math::Matrix3d(scene.boundaryModels[i]->rigidBody->GetLink()->WorldPose().Rot());
		Matrix3r R_LF_WF;
		R_LF_WF << rotation(0, 0), rotation(1, 0), rotation(2, 0),
			rotation(0, 1), rotation(1, 1), rotation(2, 1),
			rotation(0, 2), rotation(1, 2), rotation(2, 2);

		//auto linkPos = scene.boundaryModels[i]->rigidBody->GetLink()->WorldPose().Pos();
		auto collisionPos = scene.boundaryModels[i]->rigidBody->WorldPose().Pos();
		Vector3r rboPos_WF = Vector3r(collisionPos.X(), collisionPos.Y(), collisionPos.Z());

		for (int vertex = 0; vertex < n_vertices; vertex++)
		{
			Vector3r vertPos_WF = verticesInOneRbo[vertex];
			m_vertixPositionsRbo_LF[i][vertex] = R_LF_WF * (vertPos_WF - rboPos_WF);
		}
	}
}

void GazeboSimulatorBase::writeRigidBodies(const unsigned int frame)
{
	Simulation *sim = Simulation::getCurrent();
	const Utilities::GazeboSceneLoader::Scene &scene = GazeboSceneConfiguration::getCurrent()->getScene();
	const unsigned int nBoundaryModels = sim->numberOfBoundaryModels();
	std::string exportPath = FileSystem::normalizePath(m_outputPath + "/vtk");
	FileSystem::makeDirs(exportPath);

	// check if we have a static model
	bool isStatic = true;
	for (unsigned int i = 0; i < sim->numberOfBoundaryModels(); i++)
	{
		BoundaryModel *bm = sim->getBoundaryModel(i);
		if (bm->getRigidBodyObject()->isDynamic())
		{
			isStatic = false;
			break;
		}
	}

#ifdef USE_DOUBLE
	const char *real_str = " double\n";
#else
	const char *real_str = " float\n";
#endif

	bool vtkThroughVisuals = false;

	if (m_isFirstFrame || !isStatic)
	{
		/* if(vtkThroughVisuals)
		{
			std::map<std::string, unsigned int> boundariesToModels; // <ScopedName Model, boundary counter>
			for (unsigned int i = 0; i < sim->numberOfBoundaryModels(); i++)
			{
				std::string modelName = scene.boundaryModels[i]->rigidBody->GetModel()->GetScopedName();
				boundariesToModels.insert(std::pair<std::string, unsigned int>(modelName, i));
			}

			std::map<std::string, ignition::math::Pose3d> link_M; // <ScopedName Link, link pose>
			for(auto modelIterator = boundariesToModels.begin(); modelIterator != boundariesToModels.end(); modelIterator++)
			{
				physics::Link_V linksPerModel = scene.boundaryModels[modelIterator->second]->rigidBody->GetModel()->GetLinks();

				for (physics::Link_V::iterator link_it = linksPerModel.begin(); link_it != linksPerModel.end(); ++link_it)
				{
					link_M.insert(std::pair<std::string, ignition::math::Pose3d>(link_it->get()->GetScopedName(), link_it->get()->WorldPose()));
				}
			}

			std::map<std::string, ignition::math::Pose3d> pathsToLinks; // <obj path, link pose>
			std::map<std::string, std::string>::iterator scope_it = this->visualsMap.begin();
			while(scope_it != this->visualsMap.end())
			{
				auto link_it = link_M.find(scope_it->second);

				if(link_it != link_M.end()) 
				{
					pathsToLinks.insert(std::pair<std::string, ignition::math::Pose3d>(scope_it->first, link_it->second));
				}
				else gzerr << "visual key wasnt found in visual map!";

				scope_it++;
			}
			
			int visual_counter = 0;
			for(auto path_it = this->visualScales.begin(); path_it != this->visualScales.end(); path_it++)
			{
				std::string fileName = "rb_data_";
				fileName = fileName + std::to_string(visual_counter) + "_" + std::to_string(m_frameCounter) + ".vtk";
				std::string exportFileName = FileSystem::normalizePath(exportPath + "/" + fileName);

				// Open the file
				std::ofstream outfile(exportFileName, std::ios::binary);
				if (!outfile)
				{
					LOG_WARN << "Cannot open a file to save VTK mesh.";
					return;
				}
				// Header
				outfile << "# vtk DataFile Version 4.2\n";
				outfile << "SPlisHSPlasH mesh data\n";
				outfile << "BINARY\n";
				outfile << "DATASET UNSTRUCTURED_GRID\n";


				// adjust scale
				Vector3r scale = path_it->second;
				
				// create mesh to get the vertices
				auto pathFinder = this->visualsMap.find(path_it->first);
				if(pathFinder == this->visualsMap.end()) gzerr << "mesh path wasnt found!";
				std::string fullMeshPath = pathFinder->first;
				SPH::TriangleMesh mesh;
				loadObj(fullMeshPath, mesh, scale);
				
				const std::vector<Vector3r> vertices = mesh.getVertices();
				const std::vector<unsigned int> faces = mesh.getFaces();

				const unsigned int n_vertices = (int)vertices.size();
				const unsigned int n_faces = (unsigned int)faces.size();
				const unsigned int n_triangles = (int)faces.size() / 3;

				// Generate Rigid Body .vtk files
				std::vector<Vector3r> vertices_new;
				vertices_new.resize(n_vertices);

				auto linkFinder = pathsToLinks.find(path_it->first);

				if(linkFinder == pathsToLinks.end()) gzerr << "pose not found!";
				ignition::math::Pose3d linkPose = linkFinder->second;

				ignition::math::Matrix3d rotation = ignition::math::Matrix3d(linkPose.Rot());
				Matrix3r R_WF_LF;
				R_WF_LF << rotation(0, 0), rotation(0, 1), rotation(0, 2),
						rotation(1, 0), rotation(1, 1), rotation(1, 2),
						rotation(2, 0), rotation(2, 1), rotation(2, 2);

				auto linkPos = linkPose.Pos();
				Vector3r rboPos_WF = Vector3r(linkPos.X(), linkPos.Y(), linkPos.Z());


				auto poseFinder = this->visualPoses.find(path_it->first);
				if(poseFinder == this->visualPoses.end()) gzerr << "pose wasnt found in map!";
				ignition::math::Pose3d pose_visual = poseFinder->second;

				ignition::math::Matrix3d rotation_visual = ignition::math::Matrix3d(pose_visual.Rot());
				Matrix3r R_WF_LF_vis;
				R_WF_LF_vis << rotation_visual(0, 0), rotation_visual(0, 1), rotation_visual(0, 2),
							rotation_visual(1, 0), rotation_visual(1, 1), rotation_visual(1, 2),
							rotation_visual(2, 0), rotation_visual(2, 1), rotation_visual(2, 2);

				Vector3r position_visual = Vector3r(pose_visual.Pos().X(), pose_visual.Pos().Y(), pose_visual.Pos().Z());

				std::vector<Vector3r> vertices_LF;
				vertices_LF.resize(n_vertices);
				for(int vertex = 0; vertex < n_vertices; vertex++)
				{
					vertices_LF[vertex] = position_visual + R_WF_LF_vis * vertices[vertex];
					vertices_new[vertex] = rboPos_WF + R_WF_LF *  vertices_LF[vertex];
				}

				// Vertices
				{
					std::vector<Vector3r> positions;
					positions.reserve(n_vertices);
					for (int j = 0u; j < n_vertices; j++)
					{
						Vector3r x = vertices_new[j];
						//Vector3r x = vertices[j];
						swapByteOrder(&x[0]);
						swapByteOrder(&x[1]);
						swapByteOrder(&x[2]);
						positions.emplace_back(x);
					}
					// export to vtk
					outfile << "POINTS " << n_vertices << real_str;
					outfile.write(reinterpret_cast<char*>(positions[0].data()), 3 * n_vertices * sizeof(Real));
					outfile << "\n";
				}

				// Connectivity
				{
					std::vector<int> connectivity_to_write;
					connectivity_to_write.reserve(4 * n_triangles);
					for (int tri_i = 0; tri_i < n_triangles; tri_i++)
					{
						int val = 3;
						swapByteOrder(&val);
						connectivity_to_write.push_back(val);
						val = faces[3 * tri_i + 0];
						swapByteOrder(&val);
						connectivity_to_write.push_back(val);
						val = faces[3 * tri_i + 1];
						swapByteOrder(&val);
						connectivity_to_write.push_back(val);
						val = faces[3 * tri_i + 2];
						swapByteOrder(&val);
						connectivity_to_write.push_back(val);
					}
					// export to vtk
					outfile << "CELLS " << n_triangles << " " << 4 * n_triangles << "\n";
					outfile.write(reinterpret_cast<char*>(&connectivity_to_write[0]), connectivity_to_write.size() * sizeof(int));
					outfile << "\n";
				}

				// Cell types
				{
					outfile << "CELL_TYPES " << n_triangles << "\n";
					int cell_type_swapped = 5;
					swapByteOrder(&cell_type_swapped);
					std::vector<int> cell_type_arr(n_triangles, cell_type_swapped);
					outfile.write(reinterpret_cast<char*>(&cell_type_arr[0]), cell_type_arr.size() * sizeof(int));
					outfile << "\n";
				}
				outfile.close(); 
				visual_counter++;
			}
		} */
		/* else
		{ */
		for (unsigned int i = 0; i < sim->numberOfBoundaryModels(); i++)
		{
			std::string fileName = "rb_data_";
			fileName = fileName + std::to_string(i) + "_" + std::to_string(frame) + ".vtk";
			std::string exportFileName = FileSystem::normalizePath(exportPath + "/" + fileName);

			// Open the file
			std::ofstream outfile(exportFileName, std::ios::binary);
			if (!outfile)
			{
				LOG_WARN << "Cannot open a file to save VTK mesh.";
				return;
			}

			// Header
			outfile << "# vtk DataFile Version 4.2\n";
			outfile << "SPlisHSPlasH mesh data\n";
			outfile << "BINARY\n";
			outfile << "DATASET UNSTRUCTURED_GRID\n";

			BoundaryModel *bm = sim->getBoundaryModel(i);

			const std::vector<Vector3r> &vertices = bm->getRigidBodyObject()->getVertices();
			const std::vector<unsigned int> &faces = bm->getRigidBodyObject()->getFaces();
			int n_vertices = (int)vertices.size();
			int n_triangles = (int)faces.size() / 3;

			// Generate Rigid Body .vtk files
			std::vector<Vector3r> vertices_new;
			vertices_new.resize(n_vertices);
			/* ignition::math::Matrix3d rotation = ignition::math::Matrix3d(
				scene.boundaryModels[i]->rigidBody->GetLink()->WorldPose().Rot());
			 */	ignition::math::Matrix3d rotation = ignition::math::Matrix3d(
				scene.boundaryModels[i]->rigidBody->WorldPose().Rot());
			Matrix3r R_WF_LF;
			R_WF_LF << rotation(0, 0), rotation(0, 1), rotation(0, 2),
				rotation(1, 0), rotation(1, 1), rotation(1, 2),
				rotation(2, 0), rotation(2, 1), rotation(2, 2);

			//auto linkPos = scene.boundaryModels[i]->rigidBody->GetLink()->WorldPose().Pos();
			auto linkPos = scene.boundaryModels[i]->rigidBody->WorldPose().Pos();

			Vector3r rboPos_WF = Vector3r(linkPos.X(), linkPos.Y(), linkPos.Z());

			for (int vertex = 0; vertex < n_vertices; vertex++)
			{
				vertices_new[vertex] = rboPos_WF + R_WF_LF * m_vertixPositionsRbo_LF[i][vertex];
			}

			// Vertices
			{
				std::vector<Vector3r> positions;
				positions.reserve(n_vertices);
				for (int j = 0u; j < n_vertices; j++)
				{
					Vector3r x = vertices_new[j];
					swapByteOrder(&x[0]);
					swapByteOrder(&x[1]);
					swapByteOrder(&x[2]);
					positions.emplace_back(x);
				}
				// export to vtk
				outfile << "POINTS " << n_vertices << real_str;
				outfile.write(reinterpret_cast<char *>(positions[0].data()), 3 * n_vertices * sizeof(Real));
				outfile << "\n";
			}

			// Connectivity
			{
				std::vector<int> connectivity_to_write;
				connectivity_to_write.reserve(4 * n_triangles);
				for (int tri_i = 0; tri_i < n_triangles; tri_i++)
				{
					int val = 3;
					swapByteOrder(&val);
					connectivity_to_write.push_back(val);
					val = faces[3 * tri_i + 0];
					swapByteOrder(&val);
					connectivity_to_write.push_back(val);
					val = faces[3 * tri_i + 1];
					swapByteOrder(&val);
					connectivity_to_write.push_back(val);
					val = faces[3 * tri_i + 2];
					swapByteOrder(&val);
					connectivity_to_write.push_back(val);
				}
				// export to vtk
				outfile << "CELLS " << n_triangles << " " << 4 * n_triangles << "\n";
				outfile.write(reinterpret_cast<char *>(&connectivity_to_write[0]), connectivity_to_write.size() * sizeof(int));
				outfile << "\n";
			}

			// Cell types
			{
				outfile << "CELL_TYPES " << n_triangles << "\n";
				int cell_type_swapped = 5;
				swapByteOrder(&cell_type_swapped);
				std::vector<int> cell_type_arr(n_triangles, cell_type_swapped);
				outfile.write(reinterpret_cast<char *>(&cell_type_arr[0]), cell_type_arr.size() * sizeof(int));
				outfile << "\n";
			}
			outfile.close();
		}
		//}
	}

	m_isFirstFrame = false;
}

/* void GazeboSimulatorBase::initDensityMap(std::vector<Vector3r> &x, std::vector<unsigned int> &faces, const Utilities::GazeboSceneLoader::GazeboBoundaryData *boundaryData, const bool md5, const bool isDynamic, BoundaryModel_Koschier2017 *boundaryModel)
{
	Simulation *sim = Simulation::getCurrent();
	const std::string &sceneFile = GazeboSceneConfiguration::getCurrent()->getSceneFile();
	const Utilities::GazeboSceneLoader::Scene &scene = GazeboSceneConfiguration::getCurrent()->getScene();
	const Real supportRadius = sim->getSupportRadius();
	std::string scene_path = FileSystem::getFilePath(sceneFile);
	std::string scene_file_name = FileSystem::getFileName(sceneFile);
	const bool useCache = getUseParticleCaching();
	Discregrid::CubicLagrangeDiscreteGrid *densityMap;

	// if a map file is given, use this one
	if (boundaryData->mapFile != "")
	{
		std::string mapFileName = boundaryData->mapFile;
		if (FileSystem::isRelativePath(mapFileName))
			mapFileName = FileSystem::normalizePath(scene_path + "/" + mapFileName);
		densityMap = new Discregrid::CubicLagrangeDiscreteGrid(mapFileName);
		boundaryModel->setMap(densityMap);
		LOG_INFO << "Loaded density map: " << mapFileName;
		return;
	}

	string cachePath = scene_path + "/Cache";

	// Cache map
	std::string mesh_base_path = FileSystem::getFilePath(boundaryData->meshFile);
	std::string mesh_file_name = FileSystem::getFileName(boundaryData->meshFile);

	Eigen::Matrix<unsigned int, 3, 1> resolutionSDF = boundaryData->mapResolution;
	const string scaleStr = "s" + real2String(boundaryData->scale[0]) + "_" + real2String(boundaryData->scale[1]) + "_" + real2String(boundaryData->scale[2]);
	const string resStr = "r" + to_string(resolutionSDF[0]) + "_" + to_string(resolutionSDF[1]) + "_" + to_string(resolutionSDF[2]);
	const string invertStr = "i" + to_string((int)boundaryData->mapInvert);
	const string thicknessStr = "t" + real2String(boundaryData->mapThickness);
	const string kernelStr = "k" + to_string(sim->getKernel());
	string densityMapFileName = "";
	if (isDynamic)
		densityMapFileName = FileSystem::normalizePath(cachePath + "/" + mesh_file_name + "_db_dm_" + real2String(scene.particleRadius) + "_" + scaleStr + "_" + resStr + "_" + invertStr + "_" + thicknessStr + "_" + kernelStr + ".cdm");
	else
		densityMapFileName = FileSystem::normalizePath(cachePath + "/" + mesh_file_name + "_sb_dm_" + real2String(scene.particleRadius) + "_" + scaleStr + "_" + resStr + "_" + invertStr + "_" + thicknessStr + "_" + kernelStr + ".cdm");

	// check MD5 if cache file is available
	bool foundCacheFile = false;

	if (useCache)
		foundCacheFile = FileSystem::fileExists(densityMapFileName);

	if (useCache && foundCacheFile && md5)
	{
		densityMap = new Discregrid::CubicLagrangeDiscreteGrid(densityMapFileName);
		boundaryModel->setMap(densityMap);
		LOG_INFO << "Loaded cached density map: " << densityMapFileName;
		return;
	}

	if (!useCache || !foundCacheFile || !md5)
	{
		//////////////////////////////////////////////////////////////////////////
		// Generate distance field of object using Discregrid
		//////////////////////////////////////////////////////////////////////////
#ifdef USE_DOUBLE
		Discregrid::TriangleMesh sdfMesh(&x[0][0], faces.data(), x.size(), faces.size() / 3);
#else
		// if type is float, copy vector to double vector
		std::vector<double> doubleVec;
		doubleVec.resize(3 * x.size());
		for (unsigned int i = 0; i < x.size(); i++)
			for (unsigned int j = 0; j < 3; j++)
				doubleVec[3 * i + j] = x[i][j];
		Discregrid::TriangleMesh sdfMesh(&doubleVec[0], faces.data(), x.size(), faces.size() / 3);
#endif

		Discregrid::MeshDistance md(sdfMesh);
		Eigen::AlignedBox3d domain;
		for (auto const &x_ : x)
		{
			domain.extend(x_.cast<double>());
		}
		const Real tolerance = boundaryData->mapThickness;
		domain.max() += (4.0 * supportRadius + tolerance) * Eigen::Vector3d::Ones();
		domain.min() -= (4.0 * supportRadius + tolerance) * Eigen::Vector3d::Ones();

		LOG_INFO << "Domain - min: " << domain.min()[0] << ", " << domain.min()[1] << ", " << domain.min()[2];
		LOG_INFO << "Domain - max: " << domain.max()[0] << ", " << domain.max()[1] << ", " << domain.max()[2];

		LOG_INFO << "Set SDF resolution: " << resolutionSDF[0] << ", " << resolutionSDF[1] << ", " << resolutionSDF[2];
		densityMap = new Discregrid::CubicLagrangeDiscreteGrid(domain, std::array<unsigned int, 3>({resolutionSDF[0], resolutionSDF[1], resolutionSDF[2]}));
		auto func = Discregrid::DiscreteGrid::ContinuousFunction{};

		Real sign = 1.0;
		if (boundaryData->mapInvert)
			sign = -1.0;
		func = [&md, &sign, &tolerance](Eigen::Vector3d const &xi) { return sign * (md.signedDistanceCached(xi) - tolerance); };

		LOG_INFO << "Generate SDF";
		START_TIMING("SDF Construction");
		densityMap->addFunction(func, false);
		STOP_TIMING_AVG

		const bool sim2D = sim->is2DSimulation();

		//////////////////////////////////////////////////////////////////////////
		// Generate density map of object using Discregrid
		//////////////////////////////////////////////////////////////////////////
		if (sim2D)
			SimpleQuadrature::determineSamplePointsInCircle(supportRadius, 30);

		auto int_domain = Eigen::AlignedBox3d(Eigen::Vector3d::Constant(-supportRadius), Eigen::Vector3d::Constant(supportRadius));
		Real factor = 5.0;
		if (sim2D)
			factor = 1.75;
		auto density_func = [&](Eigen::Vector3d const &x) {
			auto d = densityMap->interpolate(0u, x);
			if (d > (1.0 + 1.0 / factor) * supportRadius)
			{
				return 0.0;
			}

			auto integrand = [&](Eigen::Vector3d const &xi) {
				if (xi.squaredNorm() > supportRadius * supportRadius)
					return 0.0;

				auto dist = densityMap->interpolate(0u, x + xi);

				// Linear function gamma
				if (dist > 1.0 / factor * supportRadius)
					return 0.0;
				return static_cast<double>((1.0 - factor * dist / supportRadius) * sim->W(xi.cast<Real>()));
			};

			double res = 0.0;
			if (sim2D)
				res = 0.8 * SimpleQuadrature::integrate(integrand);
			else
				res = 0.8 * GaussQuadrature::integrate(integrand, int_domain, 50);

			return res;
		};

		auto cell_diag = densityMap->cellSize().norm();
		std::cout << "Generate density map..." << std::endl;
		const bool no_reduction = true;
		START_TIMING("Density Map Construction");
		densityMap->addFunction(density_func, false, [&](Eigen::Vector3d const &x_) {
			if (no_reduction)
			{
				return true;
			}
			auto x = x_.cwiseMax(densityMap->domain().min()).cwiseMin(densityMap->domain().max());
			auto dist = densityMap->interpolate(0u, x);
			if (dist == std::numeric_limits<double>::max())
			{
				return false;
			}

			return fabs(dist) < 2.5 * supportRadius;
		});
		STOP_TIMING_PRINT;

		// reduction
		if (!no_reduction)
		{
			std::cout << "Reduce discrete fields...";
			densityMap->reduceField(0u, [&](const Eigen::Vector3d &, double v) {
				return fabs(v) < 2.5 * supportRadius;
			});
			densityMap->reduceField(1u, [&](const Eigen::Vector3d &, double v) -> double {
				if (v == std::numeric_limits<double>::max())
					return false;
				return true;
			});
			std::cout << "DONE" << std::endl;
		}

		boundaryModel->setMap(densityMap);

		// Store cache file
		if (useCache && (FileSystem::makeDir(cachePath) == 0))
		{
			LOG_INFO << "Save density map: " << densityMapFileName;
			densityMap->save(densityMapFileName);
		}
	}
} */

/* void GazeboSimulatorBase::initVolumeMap(std::vector<Vector3r> &x, std::vector<unsigned int> &faces, const Utilities::GazeboSceneLoader::GazeboBoundaryData *boundaryData, const bool md5, const bool isDynamic, BoundaryModel_Bender2019 *boundaryModel)
{
	Simulation *sim = Simulation::getCurrent();
	const std::string &sceneFile = GazeboSceneConfiguration::getCurrent()->getSceneFile();
	const Utilities::GazeboSceneConfiguration::Scene &scene = GazeboSceneConfiguration::getCurrent()->getScene();
	const Real supportRadius = sim->getSupportRadius();
	std::string scene_path = FileSystem::getFilePath(sceneFile);
	std::string scene_file_name = FileSystem::getFileName(sceneFile);
	const bool useCache = getUseParticleCaching();
	Discregrid::CubicLagrangeDiscreteGrid *volumeMap;

	// if a map file is given, use this one
	if (boundaryData->mapFile != "")
	{
		std::string mapFileName = boundaryData->mapFile;
		if (FileSystem::isRelativePath(mapFileName))
			mapFileName = FileSystem::normalizePath(scene_path + "/" + mapFileName);
		volumeMap = new Discregrid::CubicLagrangeDiscreteGrid(mapFileName);
		boundaryModel->setMap(volumeMap);
		LOG_INFO << "Loaded volume map: " << mapFileName;
		return;
	}

	string cachePath = scene_path + "/Cache";

	// Cache map
	std::string mesh_base_path = FileSystem::getFilePath(boundaryData->meshFile);
	std::string mesh_file_name = FileSystem::getFileName(boundaryData->meshFile);

	Eigen::Matrix<unsigned int, 3, 1> resolutionSDF = boundaryData->mapResolution;
	const string scaleStr = "s" + real2String(boundaryData->scale[0]) + "_" + real2String(boundaryData->scale[1]) + "_" + real2String(boundaryData->scale[2]);
	const string resStr = "r" + to_string(resolutionSDF[0]) + "_" + to_string(resolutionSDF[1]) + "_" + to_string(resolutionSDF[2]);
	const string invertStr = "i" + to_string((int)boundaryData->mapInvert);
	const string thicknessStr = "t" + real2String(boundaryData->mapThickness);
	string volumeMapFileName = "";
	if (isDynamic)
		volumeMapFileName = FileSystem::normalizePath(cachePath + "/" + mesh_file_name + "_db_vm_" + real2String(scene.particleRadius) + "_" + scaleStr + "_" + resStr + "_" + invertStr + "_" + thicknessStr + ".cdm");
	else
		volumeMapFileName = FileSystem::normalizePath(cachePath + "/" + mesh_file_name + "_sb_vm_" + real2String(scene.particleRadius) + "_" + scaleStr + "_" + resStr + "_" + invertStr + "_" + thicknessStr + ".cdm");

	// check MD5 if cache file is available
	bool foundCacheFile = false;

	if (useCache)
		foundCacheFile = FileSystem::fileExists(volumeMapFileName);

	if (useCache && foundCacheFile && md5)
	{
		volumeMap = new Discregrid::CubicLagrangeDiscreteGrid(volumeMapFileName);
		boundaryModel->setMap(volumeMap);
		LOG_INFO << "Loaded cached volume map: " << volumeMapFileName;
		return;
	}

	if (!useCache || !foundCacheFile || !md5)
	{
		//////////////////////////////////////////////////////////////////////////
		// Generate distance field of object using Discregrid
		//////////////////////////////////////////////////////////////////////////
#ifdef USE_DOUBLE
		Discregrid::TriangleMesh sdfMesh(&x[0][0], faces.data(), x.size(), faces.size() / 3);
#else
		// if type is float, copy vector to double vector
		std::vector<double> doubleVec;
		doubleVec.resize(3 * x.size());
		for (unsigned int i = 0; i < x.size(); i++)
			for (unsigned int j = 0; j < 3; j++)
				doubleVec[3 * i + j] = x[i][j];
		Discregrid::TriangleMesh sdfMesh(&doubleVec[0], faces.data(), x.size(), faces.size() / 3);
#endif

		Discregrid::MeshDistance md(sdfMesh);
		Eigen::AlignedBox3d domain;
		for (auto const &x_ : x)
		{
			domain.extend(x_.cast<double>());
		}
		const Real tolerance = boundaryData->mapThickness;
		domain.max() += (4.0 * supportRadius + tolerance) * Eigen::Vector3d::Ones();
		domain.min() -= (4.0 * supportRadius + tolerance) * Eigen::Vector3d::Ones();

		LOG_INFO << "Domain - min: " << domain.min()[0] << ", " << domain.min()[1] << ", " << domain.min()[2];
		LOG_INFO << "Domain - max: " << domain.max()[0] << ", " << domain.max()[1] << ", " << domain.max()[2];

		LOG_INFO << "Set SDF resolution: " << resolutionSDF[0] << ", " << resolutionSDF[1] << ", " << resolutionSDF[2];
		volumeMap = new Discregrid::CubicLagrangeDiscreteGrid(domain, std::array<unsigned int, 3>({resolutionSDF[0], resolutionSDF[1], resolutionSDF[2]}));
		auto func = Discregrid::DiscreteGrid::ContinuousFunction{};

		//volumeMap->setErrorTolerance(0.001);

		Real sign = 1.0;
		if (boundaryData->mapInvert)
			sign = -1.0;
		const Real particleRadius = sim->getParticleRadius();
		// subtract 0.5 * particle radius to prevent penetration of particles and the boundary
		func = [&md, &sign, &tolerance, &particleRadius](Eigen::Vector3d const &xi) { return sign * (md.signedDistanceCached(xi) - tolerance - 0.5 * particleRadius); };

		LOG_INFO << "Generate SDF";
		START_TIMING("SDF Construction");
		volumeMap->addFunction(func, false);
		STOP_TIMING_PRINT

		//////////////////////////////////////////////////////////////////////////
		// Generate volume map of object using Discregrid
		//////////////////////////////////////////////////////////////////////////

		Simulation *sim = Simulation::getCurrent();
		const bool sim2D = sim->is2DSimulation();

		if (sim2D)
			SimpleQuadrature::determineSamplePointsInCircle(supportRadius, 30);
		auto int_domain = Eigen::AlignedBox3d(Eigen::Vector3d::Constant(-supportRadius), Eigen::Vector3d::Constant(supportRadius));
		Real factor = 1.0;
		if (sim2D)
			factor = 1.0;
		auto volume_func = [&](Eigen::Vector3d const &x) {
			auto dist = volumeMap->interpolate(0u, x); */
//if (dist > (1.0 + 1.0 /*/ factor*/) * supportRadius)
/* 	{
				return 0.0;
			}

			auto integrand = [&volumeMap, &x, &supportRadius, &factor, &sim, &sim2D](Eigen::Vector3d const &xi) -> double {
				if (xi.squaredNorm() > supportRadius * supportRadius)
					return 0.0;

				auto dist = volumeMap->interpolate(0u, x + xi);

				if (dist <= 0.0)
					return 1.0 - 0.001 * dist / supportRadius;
				if (dist < 1.0 / factor * supportRadius)
					return static_cast<double>(CubicKernel::W(factor * static_cast<Real>(dist)) / CubicKernel::W_zero());
				return 0.0;
			};

			double res = 0.0;
			if (sim2D)
				res = 1.2 * SimpleQuadrature::integrate(integrand);
			else
				res = 1.2 * GaussQuadrature::integrate(integrand, int_domain, 30);

			return res;
		};

		auto cell_diag = volumeMap->cellSize().norm();
		std::cout << "Generate volume map..." << std::endl;
		const bool no_reduction = true;
		START_TIMING("Volume Map Construction");
		volumeMap->addFunction(volume_func, false, [&](Eigen::Vector3d const &x_) {
			if (no_reduction)
			{
				return true;
			}
			auto x = x_.cwiseMax(volumeMap->domain().min()).cwiseMin(volumeMap->domain().max());
			auto dist = volumeMap->interpolate(0u, x);
			if (dist == std::numeric_limits<double>::max())
			{
				return false;
			}

			return fabs(dist) < 4.0 * supportRadius;
		});
		STOP_TIMING_PRINT;

		// reduction
		if (!no_reduction)
		{
			std::cout << "Reduce discrete fields...";
			volumeMap->reduceField(0u, [&](const Eigen::Vector3d &, double v) {
				return fabs(v) < 4.0 * supportRadius;
			});
			volumeMap->reduceField(1u, [&](const Eigen::Vector3d &, double v) -> double {
				if (v == std::numeric_limits<double>::max())
					return false;
				return true;
			});
			std::cout << "DONE" << std::endl;
		}

		boundaryModel->setMap(volumeMap);

		// Store cache file
		if (useCache && (FileSystem::makeDir(cachePath) == 0))
		{
			LOG_INFO << "Save volume map: " << volumeMapFileName;
			volumeMap->save(volumeMapFileName);
		}
	}

	// store maximal distance of a point to center of mass for CFL
	if (boundaryData->dynamic)
	{
		// determine center of mass
		Vector3r com;
		com.setZero();
		for (unsigned int i = 0; i < x.size(); i++)
		{
			com += x[i];
		}
		com /= static_cast<Real>(x.size());

		// determine point with maximal distance to center of mass
		Real maxDist = 0.0;
		for (unsigned int i = 0; i < x.size(); i++)
		{
			const Vector3r diff = x[i] - com;
			const Real dist = diff.norm();
			if (dist > maxDist)
			{
				maxDist = dist;
			}
		}
		boundaryModel->setMaxDist(maxDist);
	}
}
 */
