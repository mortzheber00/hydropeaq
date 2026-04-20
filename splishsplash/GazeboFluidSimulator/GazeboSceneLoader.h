#ifndef __GazeboSceneLoader_h__
#define __GazeboSceneLoader_h__

#include "SPlisHSPlasH/Common.h"
#include "gazebo/gazebo.hh"
#include <vector>
#include "ParameterObject.h"

namespace Utilities
{
	/** \brief Importer of Gazebo SPlisHSPlasH scene files. 
	*/
	class GazeboSceneLoader
	{

	public:
		/** \brief Struct for an AABB */
		struct Box
		{
			Vector3r m_minX;
			Vector3r m_maxX;
		};

		/** \brief Struct to store a fluid object */
		struct FluidData
		{
			std::string id;
			std::string samplesFile;
			Vector3r translation;
			Matrix3r rotation;
			Vector3r scale;
			Vector3r initialVelocity;
			unsigned char mode;
			bool invert;
			std::array<unsigned int, 3> resolutionSDF;
		};

		/** \brief Struct to store a fluid block */
		struct FluidBlock
		{
			std::string id;
			Box box;
			unsigned char mode;
			Vector3r initialVelocity;
		};

		/** \brief Struct to store an emitter object */
		struct EmitterData
		{
			std::string id;
			unsigned int width;
			unsigned int height;
			Vector3r x;
			Real velocity; // emission velocity
			Matrix3r rotation;
			Real emitStartTime;
			Real emitEndTime;
			unsigned int type;
		};

		struct GazeboBoundaryData
		{
			std::string collisionName;
			std::string objFilePath;
			Vector3r translation;
			Matrix3r rotation;
			Vector3r scale;
			bool dynamic;
			bool isWall;
			Real density = 1000.0;
			gazebo::physics::CollisionPtr rigidBody;
			unsigned int samplingMode;
		};

		/** \brief Struct to store scene information */
		struct Scene
		{
			std::vector<GazeboBoundaryData *> boundaryModels;
			std::vector<GazeboSceneLoader::FluidData *> fluidModels;
			std::vector<GazeboSceneLoader::FluidBlock *> fluidBlocks;
			std::vector<GazeboSceneLoader::EmitterData *> emitters;
			Real particleRadius;
			Real timeStepSize;
			std::string outputPath;
		};

		template <typename T>
		bool getSDFParameter(const sdf::ElementPtr sdf, T &parameter, const std::string &parameterName, const T &defaultValue);
		void readScene(sdf::ElementPtr fluidSceneSDF, Scene &scene);
		void readParameterObject(const std::string &key, GenParam::ParameterObject *paramObj);
		bool getVector3rParameter(const sdf::ElementPtr sdf, Vector3r &parameter, const ::std::string &parameterName, const Vector3r &defaultValue);
		void processBoundary(Scene &scene, const gazebo::physics::CollisionPtr &collision, const std::string &objFilePath);
		//void getVector3iParameter(const sdf::ElementPtr sdf, Eigen::Matrix<unsigned int, 3, 1> &parameter, const ::std::string &parameterName, const Eigen::Matrix<unsigned int, 3, 1> &defaultValue)
		//void getSDFParameter(Scene &scene, sdf::ElementPtr sdf, const std::string &parameterName);

	private:
		sdf::ElementPtr fluidSceneSDF;
		void processFluidModels(Scene &scene, const sdf::ElementPtr &sdf);
		void processFluidBlocks(Scene &scene, const sdf::ElementPtr &sdf);
		void processFluidEmmiters(Scene &scene, const sdf::ElementPtr &sdf);
	};

	/* template <>
	bool GazeboSceneLoader::readValue<bool>(const nlohmann::json &j, bool &v); */

} // namespace Utilities

#endif
