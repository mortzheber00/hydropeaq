#ifndef __GazeboSceneConfiguration_h__
#define __GazeboSceneConfiguration_h__

#include "GazeboSceneLoader.h"

namespace SPH
{
	/** \brief Class to store the scene configuration that is imported from the scene file.
	*/
	class GazeboSceneConfiguration
	{
	private:
		static GazeboSceneConfiguration *m_current;

	protected:
		Utilities::GazeboSceneLoader::Scene m_scene;
		std::string m_sceneFile;

	public:
		GazeboSceneConfiguration();
		GazeboSceneConfiguration(const GazeboSceneConfiguration&) = delete;
		GazeboSceneConfiguration& operator=(const GazeboSceneConfiguration&) = delete;
		~GazeboSceneConfiguration();

		// Singleton
		static GazeboSceneConfiguration* getCurrent ();
		static void setCurrent (GazeboSceneConfiguration* sc);
		static bool hasCurrent();

		void setSceneFile(const std::string& file) { m_sceneFile = file; }
		const std::string& getSceneFile() const { return m_sceneFile; }

		Utilities::GazeboSceneLoader::Scene& getScene() { return m_scene; }

	};
}

#endif
