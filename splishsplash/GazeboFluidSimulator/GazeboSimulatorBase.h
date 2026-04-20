#ifndef __GazeboSimulatorBase_h__
#define __GazeboSimulatorBase_h__

#include "SPlisHSPlasH/Common.h"
#include "SPlisHSPlasH/TimeStep.h"
#include "SPlisHSPlasH/FluidModel.h"
#include "ParameterObject.h"
#include "SPlisHSPlasH/BoundaryModel_Akinci2012.h"
#include "SPlisHSPlasH/BoundaryModel_Koschier2017.h"
#include "SPlisHSPlasH/BoundaryModel_Bender2019.h"
#include "SPlisHSPlasH/TriangleMesh.h"
#include "GazeboSceneConfiguration.h"
#include "GazeboRigidBody.h"
#include "GazeboSceneLoader.h"
#include "GazeboBoundarySimulator.h"
#include "gazebo/physics/PhysicsTypes.hh"
#include "gazebo/common/CommonTypes.hh"
#include "gazebo/physics/physics.hh"

using namespace gazebo;

namespace SPH
{
	class ExporterBase;

	class GazeboSimulatorBase : public GenParam::ParameterObject
	{
	public:
		struct SimulationMethod
		{
			short simulationMethod = 0;
			TimeStep *simulation = NULL;
			FluidModel model;
		};

		struct Exporter
		{
			std::string m_key;
			std::string m_name;
			std::string m_description;
			ExporterBase *m_exporter;
			int m_id;
		};

	protected:
		int m_simulation_steps;
		std::string m_exePath;
		std::string m_stateFile;
		std::string m_outputPath;
		bool m_useParticleCaching;
		bool m_isStaticScene;
		bool m_enableRigidBodyVTKExport;
		bool m_enableRigidBodyExport;
		bool m_enableStateExport;
		Real m_framesPerSecond;
		Real m_framesPerSecondState;
		std::string m_particleAttributes;
		std::unique_ptr<Utilities::GazeboSceneLoader> m_sceneLoader;
		Real m_nextFrameTime;
		Real m_nextFrameTimeState;
		bool m_firstState;
		unsigned int m_frameCounter;
		bool m_isFirstFrame;
		bool m_isFirstFrameVTK;
		GazeboBoundarySimulator *m_boundarySimulator;
		std::vector<std::string> m_paramTokens;
		std::function<void()> m_timeStepCB;
		std::vector<Exporter> m_particleExporters;
		std::vector<Exporter> m_rbExporters;
		std::vector<std::vector<Vector3r>> m_vertixPositionsRbo_LF;
		void writeRigidBodies(const unsigned int frame);
#ifdef DL_OUTPUT
		Real m_nextTiming;
#endif

		virtual void initParameters();

		void initFluidData();
		void createFluidBlocks(std::map<std::string, unsigned int> &fluidIDs, std::vector<std::vector<Vector3r>> &fluidParticles, std::vector<std::vector<Vector3r>> &fluidVelocities);
		//void createEmitters();
		void createExporters();
		void cleanupExporters();
		void initExporters();

	public:
		static int DATA_EXPORT_FPS;
		static int PARTICLE_EXPORT_ATTRIBUTES;
		static int STATE_EXPORT;
		static int STATE_EXPORT_FPS;
		static int RENDER_WALLS;
		GazeboSimulatorBase();
		GazeboSimulatorBase(const GazeboSimulatorBase &) = delete;
		GazeboSimulatorBase &operator=(const GazeboSimulatorBase &) = delete;
		virtual ~GazeboSimulatorBase();

		void run();
		void buildModel();
		void init(sdf::ElementPtr &worldPluginSDF);
		void initSimulation();
		void cleanup();
		void processBoundary(physics::CollisionPtr collision, std::string objFilePath);
		void reset();
		bool timeStepNoGUI();

		void setTimeStepCB(std::function<void()> const &callBackFct) { m_timeStepCB = callBackFct; }

		static void particleInfo(std::vector<std::vector<unsigned int>> &particles);

		std::string real2String(const Real r);
		void initDensityMap(std::vector<Vector3r> &x, std::vector<unsigned int> &faces, const Utilities::GazeboSceneLoader::GazeboBoundaryData *boundaryData, const bool md5, const bool isDynamic, BoundaryModel_Koschier2017 *boundaryModel);
		void initVolumeMap(std::vector<Vector3r> &x, std::vector<unsigned int> &faces, const Utilities::GazeboSceneLoader::GazeboBoundaryData *boundaryData, const bool md5, const bool isDynamic, BoundaryModel_Bender2019 *boundaryModel);

		void readParameters();

		void step();

		void saveState(const std::string &stateFile = "");
		void writeFluidParticlesState(const std::string &fileName, FluidModel *model);
		void readFluidParticlesState(const std::string &fileName, FluidModel *model);
		void writeBoundaryState(const std::string &fileName, BoundaryModel *model);
		void readBoundaryState(const std::string &fileName, BoundaryModel *model);
		void writeParameterState(BinaryFileWriter &binWriter);
		void readParameterState(BinaryFileReader &binReader);
		void writeParameterObjectState(BinaryFileWriter &binWriter, GenParam::ParameterObject *paramObj);
		void readParameterObjectState(BinaryFileReader &binReader, GenParam::ParameterObject *paramObj);

		void updateBoundaryParticles(const bool forceUpdate);
		void updateDMVelocity();
		void updateVMVelocity();

		static void loadObj(const std::string &filename, TriangleMesh &mesh, const Vector3r &scale);

		Utilities::GazeboSceneLoader *getSceneLoader() { return m_sceneLoader.get(); }

		const std::string &getExePath() const { return m_exePath; }

		bool getUseParticleCaching() const { return m_useParticleCaching; }
		void setUseParticleCaching(bool val) { m_useParticleCaching = val; }

		std::string getOutputPath() const { return m_outputPath; }

		std::string getStateFile() const { return m_stateFile; }
		void setStateFile(std::string val) { m_stateFile = val; }

		GazeboBoundarySimulator *getBoundarySimulator() const { return m_boundarySimulator; }
		void setBoundarySimulator(GazeboBoundarySimulator *val) { m_boundarySimulator = val; }
		bool isStaticScene() const { return m_isStaticScene; }

		void addParticleExporter(const std::string &key, const std::string &name, const std::string &description, ExporterBase *exporter) { m_particleExporters.push_back({key, name, description, exporter, -1}); }
		std::vector<Exporter> &getParticleExporters() { return m_particleExporters; }

		void addRigidBodyExporter(const std::string &key, const std::string &name, const std::string &description, ExporterBase *exporter) { m_rbExporters.push_back({key, name, description, exporter, -1}); }
		std::vector<Exporter> &getRigidBodyExporters() { return m_rbExporters; }
		std::vector<std::vector<Vector3r>> getInitialRboVertices() { return m_vertixPositionsRbo_LF; };
		void setInitialRboVertices(std::vector<std::vector<Vector3r>> vertixPositionsRbo_LF) { m_vertixPositionsRbo_LF = vertixPositionsRbo_LF; };
		void initRbVertixPositions();

		void activateExporter(const std::string &exporterName, const bool active);
		void initBoundaryData();

		// VTK expects big endian
		template <typename T>
		inline void swapByteOrder(T *v)
		{
			constexpr size_t n = sizeof(T);
			uint8_t *bytes = reinterpret_cast<uint8_t *>(v);
			for (unsigned int c = 0u; c < n / 2; c++)
				std::swap(bytes[c], bytes[n - c - 1]);
		}
	};
}

#endif
