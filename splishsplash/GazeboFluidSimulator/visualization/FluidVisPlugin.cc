/* Desc: System plugin for rendering the particles from Fluidix
 * Author: Andrei Haidu
 */

#include "FluidVisPlugin.hh"

using namespace gazebo;

//////////////////////////////////////////////////
FluidVisPlugin::FluidVisPlugin()
{
}

//////////////////////////////////////////////////
FluidVisPlugin::~FluidVisPlugin()
{
}

//////////////////////////////////////////////////
void FluidVisPlugin::Load(int _argc, char **_argv)
{
	// check for the given arguments
	std::cout << "Visualization plugin loaded" << std::endl;
}

//////////////////////////////////////////////////
void FluidVisPlugin::Init()
{
	this->node = transport::NodePtr(new transport::Node());

	this->newFluidMsgReceived = false;

	// Event to check that the rendering engine is loaded
	this->updateConnection = event::Events::ConnectPreRender(
		boost::bind(&FluidVisPlugin::InitAtRenderEvent, this));
}

//////////////////////////////////////////////////
void FluidVisPlugin::InitAtRenderEvent()
{
	// Initialize node only after the rendering engine has been loaded
	this->node->Init();

	// subscribe to the fluid topic
	this->fluidSub = this->node->Subscribe("~/fluid_pos",
										   &FluidVisPlugin::OnFluidMsg, this);

	this->rigidsSub = this->node->Subscribe("~/rigids_pos",
											&FluidVisPlugin::OnRigidMsg, this);

	if (!this->userCam)
	{
		// Get a pointer to the active user camera
		this->userCam = gui::get_active_camera();

		this->userCam->OgreViewport()->setVisibilityMask(GZ_VISIBILITY_ALL & ~GZ_VISIBILITY_SELECTABLE);

		// Enable saving frames
	/* 	this->userCam->EnableSaveFrame(true);
		this->userCam->SetRenderRate(1.0/60);

		// Specify the path to save frames into
		this->userCam->SetSaveFramePathname("/home/dyck/Documents/gazebo_frames_ScaledDensities3"); */
	} 

	this->manager = this->userCam->GetScene()->OgreSceneManager();

	this->updateConnection = event::Events::ConnectPreRender(
		boost::bind(&FluidVisPlugin::RenderAsPointsUpdate, this));
}

/////////////////////////////////////////////////
void FluidVisPlugin::RenderAsPointsUpdate()
{
	// render fluid if new message received
	if (this->newFluidMsgReceived)
	{
		FluidVisPlugin::RenderParticles(this->fluidParticlePositions, "fluid1",
		                                Ogre::ColourValue(0.2f, 0.6f, 1.0f), 6.0f);
		FluidVisPlugin::RenderParticles(this->rigidsParticlePositions, "rigid1",
		                                Ogre::ColourValue(1.0f, 0.4f, 0.1f), 4.0f);
		this->newFluidMsgReceived = false;
	}
}
/////////////////////////////////////////////////
void FluidVisPlugin::OnFluidMsg(
	const boost::shared_ptr<msgs::Fluid const> &_msg)
{
	this->newFluidMsgReceived = true;
	this->fluidParticlePositions.resize(_msg->position_size());

	for (int i = 0; i < _msg->position_size(); ++i)
	{
		this->fluidParticlePositions[i] = Ogre::Vector3(
			_msg->position(i).x(), _msg->position(i).y(), _msg->position(i).z());
	}
}

void FluidVisPlugin::OnRigidMsg(
	const boost::shared_ptr<msgs::Fluid const> &_msg)
{

	this->newFluidMsgReceived = true;
	this->rigidsParticlePositions.resize(_msg->position_size());

	for (int i = 0; i < _msg->position_size(); ++i)
	{
		this->rigidsParticlePositions[i] = Ogre::Vector3(
			_msg->position(i).x(), _msg->position(i).y(), _msg->position(i).z());
	}
}
/////////////////////////////////////////////////
void FluidVisPlugin::RenderParticles(std::vector<Ogre::Vector3> &_particles, std::string _name,
                                     Ogre::ColourValue _color, float _pointSize)
{
	// Create a per-name material with the requested color and point size
	std::string matName = _name + "_mat";
	if (!Ogre::MaterialManager::getSingleton().resourceExists(matName))
	{
		Ogre::MaterialPtr mat = Ogre::MaterialManager::getSingleton().create(
			matName, Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
		Ogre::Pass *pass = mat->getTechnique(0)->getPass(0);
		pass->setLightingEnabled(false);
		pass->setSelfIllumination(_color);
		pass->setPointSize(_pointSize);
	}

	Ogre::SceneNode *sceneNode = NULL;
	Ogre::ManualObject *obj = NULL;
	bool attached = false;

	if (this->manager->hasManualObject(_name))
	{
		sceneNode = this->manager->getSceneNode(_name);
		obj = this->manager->getManualObject(_name);
		attached = true;
	}
	else
	{
		sceneNode = this->manager->getRootSceneNode()->createChildSceneNode(_name);
		obj = this->manager->createManualObject(_name);
	}

	sceneNode->setVisible(true);
	obj->setVisible(true);
	obj->clear();
	obj->begin(matName, Ogre::RenderOperation::OT_POINT_LIST);

	for (unsigned int i = 0; i < _particles.size(); ++i)
	{
		obj->colour(_color);
		obj->position(_particles[i]);
	}

	obj->end();

	if (!attached)
		sceneNode->attachObject(obj);
}

/////////////////////////////////////////////////
void FluidVisPlugin::RenderParticlesAsEntities(std::vector<Ogre::Vector3> &_particles, std::string _name, const std::string& colour)
{
	//std::cout << "OnUpdate: Rendering new positions.." << std::endl;
	for (unsigned int i = 0; i < _particles.size(); ++i)
	{
		Ogre::SceneNode *sceneNode = NULL;
		Ogre::Entity *entity = NULL;
		bool attached = false;
		std::ostringstream name_ss;
		std::string name;

		name_ss << _name << "_" << i;
		name = name_ss.str();

		if (this->manager->hasEntity(name))
		{
			sceneNode = this->manager->getSceneNode(name);
			entity = this->manager->getEntity(name);
			attached = true;
		}
		else
		{
			sceneNode = this->manager->getRootSceneNode()->createChildSceneNode(name);
			entity = this->manager->createEntity(name,
												 Ogre::SceneManager::PT_SPHERE);
			entity->setMaterialName(colour);
		}

		sceneNode->setVisible(true);
		entity->setVisible(true);

		sceneNode->setScale(0.0001, 0.0001, 0.0001);

		sceneNode->setPosition(_particles[i]);

		if (!attached)
		{
			sceneNode->attachObject(entity);
		}
	}
}

// Register this plugin with the simulator
GZ_REGISTER_SYSTEM_PLUGIN(FluidVisPlugin)
