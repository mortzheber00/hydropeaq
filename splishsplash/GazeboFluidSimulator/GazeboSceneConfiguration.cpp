#include "GazeboSceneConfiguration.h"

using namespace SPH;
using namespace std;

GazeboSceneConfiguration* GazeboSceneConfiguration::m_current = nullptr;

GazeboSceneConfiguration::GazeboSceneConfiguration () 
{
	m_sceneFile = "";
}

GazeboSceneConfiguration::~GazeboSceneConfiguration () 
{
	for (unsigned int i = 0; i < m_scene.boundaryModels.size(); i++)
		delete m_scene.boundaryModels[i];
	m_scene.boundaryModels.clear();

	for (unsigned int i = 0; i < m_scene.fluidModels.size(); i++)
		delete m_scene.fluidModels[i];
	m_scene.fluidModels.clear();

	for (unsigned int i = 0; i < m_scene.fluidBlocks.size(); i++)
		delete m_scene.fluidBlocks[i];
	m_scene.fluidBlocks.clear();

	for (unsigned int i = 0; i < m_scene.emitters.size(); i++)
		delete m_scene.emitters[i];
	m_scene.emitters.clear();

	m_current = nullptr;
}

GazeboSceneConfiguration* GazeboSceneConfiguration::getCurrent ()
{
	if (m_current == nullptr)
	{
		m_current = new GazeboSceneConfiguration ();
	}
	return m_current;
}

void GazeboSceneConfiguration::setCurrent (GazeboSceneConfiguration* sc)
{
	m_current = sc;
}

bool GazeboSceneConfiguration::hasCurrent()
{
	return (m_current != nullptr);
}

