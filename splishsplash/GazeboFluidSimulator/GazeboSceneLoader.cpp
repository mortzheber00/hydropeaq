#include "GazeboSceneLoader.h"
#include "Utilities/Logger.h"
#include <iostream>
#include "gazebo/physics/physics.hh"
#include "gazebo/gazebo.hh"
#include "gazebo/util/system.hh"
#include "gazebo/physics/PhysicsTypes.hh"
using namespace Utilities;
using namespace GenParam;
using namespace std;
using namespace gazebo;

template <typename T>
bool GazeboSceneLoader::getSDFParameter(const sdf::ElementPtr sdf, T &parameter, const std::string &parameterName, const T &defaultValue)
{
	bool result = false;
	if (sdf->HasElement(parameterName))
	{
		result = true;
		//parameter = sdf->GetElement(parameterName)->Get<T>();
		parameter = sdf->Get<T>(parameterName);
	}
	else
	{
		parameter = defaultValue;
	}
	return result;
}

bool GazeboSceneLoader::getVector3rParameter(const sdf::ElementPtr sdf, Vector3r &parameter, const ::std::string &parameterName, const Vector3r &defaultValue)
{
	bool result = false;
	if (sdf->HasElement(parameterName))
	{
		result = true;
		ignition::math::Vector3d value = sdf->Get<ignition::math::Vector3d>(parameterName);
		parameter = Vector3r(value.X(), value.Y(), value.Z());
	}
	else
	{
		parameter = defaultValue;
	}
	return result;
}

/* void GazeboSceneLoader::getVector3iParameter(const sdf::ElementPtr sdf, Eigen::Matrix<unsigned int, 3, 1> &parameter, const ::std::string &parameterName, const Eigen::Matrix<unsigned int, 3, 1> &defaultValue)
{
	if (sdf->HasElement(parameterName))
	{
		ignition::math::Vector value = sdf->GetElement(parameterName)->Get<ignition::math::Vector3d>();
		parameter = Eigen::Matrix<unsigned int, 3, 1>(value.X(), value.Y(), value.Z());
	}
	else
	{
		parameter = defaultValue;
	}
} */

void GazeboSceneLoader::processFluidModels(Scene &scene, const sdf::ElementPtr &sdf)
{
	if (sdf->HasElement("fluidModels"))
	{
		sdf::ElementPtr fluidModelsElement = sdf->GetElement("fluidModels");
		while (fluidModelsElement)
		{
			std::string particleFile;
			getSDFParameter<std::string>(fluidModelsElement, particleFile, "particleFile", "");
			if (particleFile != "")
			{
				GazeboSceneLoader::FluidData *data = new GazeboSceneLoader::FluidData();
				data->samplesFile = particleFile;

				getSDFParameter<std::string>(fluidModelsElement, data->id, "id", "Fluid");
				getVector3rParameter(fluidModelsElement, data->translation, "translation", Vector3r::Zero());
				getVector3rParameter(fluidModelsElement, data->scale, "scale", Vector3r::Ones());

				data->rotation = Matrix3r::Identity();
				Vector3r rotationAxis;
				Real rotationAngle;
				getVector3rParameter(fluidModelsElement, rotationAxis, "rotationAxis", Vector3r(1.0, 0.0, 0.0));
				getSDFParameter<Real>(fluidModelsElement, rotationAngle, "rotationAngle", 0.0);
				data->rotation = AngleAxisr(rotationAngle, rotationAxis);

				getVector3rParameter(fluidModelsElement, data->initialVelocity, "initialVelocity", Vector3r::Zero());
				getSDFParameter<bool>(fluidModelsElement, data->invert, "invert", false);
	
				/* Eigen::Matrix<unsigned int, 3, 1> resolutionSDF;
				getVector3iParameter(fluidModelsElement,resolutionSDF ,"resolutionSDF", Eigen::Matrix<unsigned int, 3, 1>(20,20,20) );
				data->resolutionSDF[0] = resolutionSDF[0];
				data->resolutionSDF[1] = resolutionSDF[1];
				data->resolutionSDF[2] = resolutionSDF[2]; */

				//getSDFParameter<unsigned char>(fluidModelsElement, data->mode, "denseMode", 0);

				scene.fluidModels.push_back(data);
			}
			fluidModelsElement = fluidModelsElement->GetNextElement("fluidModels");
		}
	}
}

void GazeboSceneLoader::processFluidBlocks(Scene &scene, const sdf::ElementPtr &sdf)
{
	if (sdf->HasElement("fluidBlock"))
	{
		sdf::ElementPtr fluidBlockElement = sdf->GetElement("fluidBlock");
		while (fluidBlockElement)
		{
			Vector3r translation, scale;
			getVector3rParameter(fluidBlockElement, translation, "translation", Vector3r::Zero());
			getVector3rParameter(fluidBlockElement, scale, "scale", Vector3r::Ones());

			Vector3r minX, maxX;
			getVector3rParameter(fluidBlockElement, minX, "start", Vector3r::Zero());
			getVector3rParameter(fluidBlockElement, maxX, "end", Vector3r::Ones());

			GazeboSceneLoader::FluidBlock *block = new GazeboSceneLoader::FluidBlock();
			block->box.m_minX[0] = scale[0] * minX[0] + translation[0];
			block->box.m_minX[1] = scale[1] * minX[1] + translation[1];
			block->box.m_minX[2] = scale[2] * minX[2] + translation[2];
			block->box.m_maxX[0] = scale[0] * maxX[0] + translation[0];
			block->box.m_maxX[1] = scale[1] * maxX[1] + translation[1];
			block->box.m_maxX[2] = scale[2] * maxX[2] + translation[2];

			//getSDFParameter<std::string>(fluidBlockElement, block->id, "id", "Fluid");
			//getSDFParameter<unsigned char>(fluidBlockElement, block->mode, "denseMode", 0);
			getVector3rParameter(fluidBlockElement, block->initialVelocity, "initialVelocity", Vector3r::Zero());
			
			scene.fluidBlocks.push_back(block);
			fluidBlockElement = fluidBlockElement->GetNextElement("fluidBlock");
		}
	}
}

void GazeboSceneLoader::processFluidEmmiters(Scene &scene, const sdf::ElementPtr &sdf)
{
	if (sdf->HasElement("fluidEmmiter"))
	{
		sdf::ElementPtr fluidEmmiterElement = sdf->GetElement("fluidEmmiter");
		while (fluidEmmiterElement)
		{
			GazeboSceneLoader::EmitterData *data = new GazeboSceneLoader::EmitterData();

			getSDFParameter<std::string>(fluidEmmiterElement, data->id, "id", "Fluid");
			getSDFParameter<unsigned int>(fluidEmmiterElement, data->width, "width", 5);
			getSDFParameter<unsigned int>(fluidEmmiterElement, data->height, "height", 5);
			getVector3rParameter(fluidEmmiterElement, data->x, "translation", Vector3r::Zero());

			Vector3r axis;
			Real angle;
			data->rotation = Matrix3r::Identity();
			getVector3rParameter(fluidEmmiterElement, axis, "rotationAxis", Vector3r(0, 0, 1));
			getSDFParameter<Real>(fluidEmmiterElement, angle, "rotationAngle", 0.0);
			axis.normalize();
			data->rotation = AngleAxisr(angle, axis).toRotationMatrix();

			getSDFParameter<Real>(fluidEmmiterElement, data->velocity, "velocity", 1.0);
			getSDFParameter<Real>(fluidEmmiterElement, data->emitStartTime, "emitStartTime", 0.0);
			getSDFParameter<Real>(fluidEmmiterElement, data->emitEndTime, "emitStartTime", std::numeric_limits<Real>::max());

			// type: 0 = rectangular, 1 = circle
			getSDFParameter<unsigned int>(fluidEmmiterElement, data->type, "type", 0);

			scene.emitters.push_back(data);
			fluidEmmiterElement = fluidEmmiterElement->GetNextElement("fluidEmmiter");
		}
	}
}

void GazeboSceneLoader::processBoundary(Scene &scene, const physics::CollisionPtr &collision, const std::string &objFilePath)
{
	GazeboBoundaryData *data = new GazeboBoundaryData();

	data->objFilePath = objFilePath;
	if (collision->GetModel()->GetSDF()->HasElement("static"))
	{
		data->dynamic = !collision->GetModel()->GetSDF()->Get<bool>("static");
	}
	data->collisionName = collision->GetModel()->GetSDF()->GetAttribute("name")->GetAsString() + "_" + collision->GetName() + ".obj";
	ignition::math::Pose3d linkPose = collision->GetLink()->WorldPose();
	ignition::math::Pose3d collisionPose = collision->WorldPose();
	
	
	// TODO scale the mesh if necessary
	data->scale = Vector3r::Ones();

	// translation
	ignition::math::Vector3d translation = collisionPose.Pos();
	Vector3r fluidObjectTranslation = Vector3r(translation.X(), translation.Y(), translation.Z());
	data->translation = fluidObjectTranslation;

	// set the orientation of the geom as float4
	ignition::math::Matrix3d orientation = ignition::math::Matrix3d(collisionPose.Rot()); //.GetAsMatrix3();
	Matrix3r fluidObjectOrientation;
	fluidObjectOrientation << orientation(0, 0), orientation(0, 1), orientation(0, 2),
		orientation(1, 0), orientation(1, 1), orientation(1, 2),
		orientation(2, 0), orientation(2, 1), orientation(2, 2);
	data->rotation = fluidObjectOrientation;
	data->rigidBody = collision;
	scene.boundaryModels.push_back(data);
}

void GazeboSceneLoader::readScene(sdf::ElementPtr fluidSceneSDF, Scene &scene)
{
	LOG_INFO << "Load scene file from sdf: ";
	if (fluidSceneSDF != nullptr)
	{
		this->fluidSceneSDF = fluidSceneSDF;
	}

	if (fluidSceneSDF->HasElement("fluidConfiguration"))
	{
		sdf::ElementPtr fluidConfiguration = fluidSceneSDF->GetElement("fluidConfiguration");
		getSDFParameter<Real>(fluidConfiguration, scene.timeStepSize, "timeStepSize", 0.001);
		getSDFParameter<Real>(fluidConfiguration, scene.particleRadius, "particleRadius", 0.025);
		getSDFParameter<std::string>(fluidConfiguration, scene.outputPath, "outputPath", "/tmp");
			
	}
	else
	{
		std::cout << "Fluid configuration missing from sdf, using default values" << std::endl;
	}

	processFluidModels(scene, fluidSceneSDF);
	processFluidBlocks(scene, fluidSceneSDF);
	processFluidEmmiters(scene, fluidSceneSDF);
}

void GazeboSceneLoader::readParameterObject(const std::string &key, ParameterObject *paramObj)
{
	if (paramObj == nullptr)
		return;

	const unsigned int numParams = paramObj->numParameters();

	//////////////////////////////////////////////////////////////////////////
	// read configuration
	//////////////////////////////////////////////////////////////////////////
	if (this->fluidSceneSDF->HasElement(key))
	{
		sdf::ElementPtr config = this->fluidSceneSDF->GetElement(key);
		for (unsigned int i = 0; i < numParams; i++)
		{
			ParameterBase *paramBase = paramObj->getParameter(i);

			if (paramBase->getType() == RealParameterType)
			{
				Real val;
				if (getSDFParameter<Real>(config, val, paramBase->getName(), 0.0))
					static_cast<NumericParameter<Real> *>(paramBase)->setValue(val);
			}
			else if (paramBase->getType() == ParameterBase::UINT32)
			{
				unsigned int val;
				if (getSDFParameter<unsigned int>(config, val, paramBase->getName(), 0))
					static_cast<NumericParameter<unsigned int> *>(paramBase)->setValue(val);
			}
			/* else if (paramBase->getType() == ParameterBase::UINT16)
			{
				unsigned short val;
				if (getSDFParameter<unsigned short>(config, val, paramBase->getName(), 0))
					static_cast<NumericParameter<unsigned short>*>(paramBase)->setValue(val);
			} */

			/*  else if (paramBase->getType() == ParameterBase::UINT8)
			{
				unsigned char val;
				if (getSDFParameter<unsigned char>(config, val, paramBase->getName(), 0))
					static_cast<NumericParameter<unsigned char> *>(paramBase)->setValue(val);
			}  */

			else if (paramBase->getType() == ParameterBase::INT32)
			{
				int val;
				if (getSDFParameter<int>(config, val, paramBase->getName(), 0))
					static_cast<NumericParameter<int> *>(paramBase)->setValue(val);
			}

			/* else if (paramBase->getType() == ParameterBase::INT16)
			{
				short val;
				if (getSDFParameter<short>(config, val, paramBase->getName(), 0))
					static_cast<NumericParameter<short> *>(paramBase)->setValue(val);
			} */

			/* else if (paramBase->getType() == ParameterBase::INT8)
			{
				char val;
				if (getSDFParameter<char>(config, val, paramBase->getName(), 0))
					static_cast<NumericParameter<char> *>(paramBase)->setValue(val);
			} */

			else if (paramBase->getType() == ParameterBase::ENUM)
			{
				int val;
				if (getSDFParameter<int>(config, val, paramBase->getName(), 0))
					static_cast<EnumParameter *>(paramBase)->setValue(val);
			}

			else if (paramBase->getType() == ParameterBase::BOOL)
			{
				bool val;
				if (getSDFParameter<bool>(config, val, paramBase->getName(), 0))
					static_cast<BoolParameter *>(paramBase)->setValue(val);
			}

			else if (paramBase->getType() == RealVectorParameterType)
			{
				if (static_cast<VectorParameter<Real> *>(paramBase)->getDim() == 3)
				{
					Vector3r val;
					if (getVector3rParameter(config, val, paramBase->getName(), Vector3r::Zero()))
						static_cast<VectorParameter<Real> *>(paramBase)->setValue(val.data());
				}
			}
			else if (paramBase->getType() == ParameterBase::STRING)
			{
				std::string val;
				if (getSDFParameter<std::string>(config, val, paramBase->getName(), ""))
					static_cast<StringParameter *>(paramBase)->setValue(val);
			}
		}
	}
}
