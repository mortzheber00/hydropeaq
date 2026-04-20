#include "RigidBodyExporter_VTK.h"
#include <Utilities/Logger.h>
#include <Utilities/FileSystem.h>
#include "SPlisHSPlasH/Simulation.h"

using namespace SPH;
using namespace Utilities;

RigidBodyExporter_VTK::RigidBodyExporter_VTK(GazeboSimulatorBase *base) : ExporterBase(base)
{
	m_isFirstFrame = true;
}

RigidBodyExporter_VTK::~RigidBodyExporter_VTK(void)
{
}

void RigidBodyExporter_VTK::init(const std::string &outputPath)
{
	m_exportPath = FileSystem::normalizePath(outputPath + "/vtk");
}

void RigidBodyExporter_VTK::step(const unsigned int frame)
{
	if (!m_active)
		return;

	writeRigidBodies(frame);
}

void RigidBodyExporter_VTK::reset()
{
	m_isFirstFrame = true;
}

void RigidBodyExporter_VTK::setActive(const bool active)
{
	ExporterBase::setActive(active);
	if (m_active)
		FileSystem::makeDirs(m_exportPath);
}

void RigidBodyExporter_VTK::writeRigidBodies(const unsigned int frame)
{
	Simulation *sim = Simulation::getCurrent();
	const Utilities::GazeboSceneLoader::Scene &scene = GazeboSceneConfiguration::getCurrent()->getScene();
	const unsigned int nBoundaryModels = sim->numberOfBoundaryModels();
	std::string exportPath = FileSystem::normalizePath(scene.outputPath + "/vtk");
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
			GazeboRigidBody *rbo = dynamic_cast<GazeboRigidBody *>(bm->getRigidBodyObject());
			const std::vector<Vector3r> &vertices =  rbo->getUpdatedVertices();
			const std::vector<unsigned int> &faces = rbo->getFaces();
			int n_vertices = (int)vertices.size();
			int n_triangles = (int)faces.size() / 3;

			// Generate Rigid Body .vtk files
			std::vector<Vector3r> vertices_new;
			vertices_new.resize(n_vertices);
			ignition::math::Matrix3d rotation = ignition::math::Matrix3d(
				scene.boundaryModels[i]->rigidBody->GetLink()->WorldPose().Rot());
			Matrix3r R_WF_LF;
			R_WF_LF << rotation(0, 0), rotation(0, 1), rotation(0, 2),
				rotation(1, 0), rotation(1, 1), rotation(1, 2),
				rotation(2, 0), rotation(2, 1), rotation(2, 2);

			//auto linkPos = scene.boundaryModels[i]->rigidBody->GetLink()->WorldPose().Pos();
			auto collisionPos = scene.boundaryModels[i]->rigidBody->GetLink()->WorldPose().Pos();

			Vector3r rboPos_WF = Vector3r(collisionPos.X(), collisionPos.Y(), collisionPos.Z());

			/* for (int vertex = 0; vertex < n_vertices; vertex++)
			{
				vertices_new[vertex] = rboPos_WF + R_WF_LF * m_vertixPositionsRbo_LF[i][vertex];
			} */

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

/* void RigidBodyExporter_VTK::writeRigidBodies(const unsigned int frame)
{
	Simulation* sim = Simulation::getCurrent();
	const unsigned int nBoundaryModels = sim->numberOfBoundaryModels();

	// check if we have a static model
	bool isStatic = true;
	for (unsigned int i = 0; i < sim->numberOfBoundaryModels(); i++)
	{
		BoundaryModel* bm = sim->getBoundaryModel(i);
		if (bm->getRigidBodyObject()->isDynamic())
		{
			isStatic = false;
			break;
		}
	}

#ifdef USE_DOUBLE
	const char* real_str = " double\n";
#else 
	const char* real_str = " float\n";
#endif
	//if (m_isFirstFrame || !isStatic)
	if (m_isFirstFrame || true)
	{
		for (unsigned int i = 0; i < sim->numberOfBoundaryModels(); i++)
		{
			std::string fileName = "rb_data_";
			fileName = fileName + std::to_string(i) + "_" + std::to_string(frame) + ".vtk";
			std::string exportFileName = FileSystem::normalizePath(m_exportPath + "/" + fileName);

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

			BoundaryModel* bm = sim->getBoundaryModel(i);
			const std::vector<Vector3r>& vertices = bm->getRigidBodyObject()->getVertices();
			const std::vector<unsigned int>& faces = bm->getRigidBodyObject()->getFaces();
			int n_vertices = (int)vertices.size();
			int n_triangles = (int)faces.size() / 3;

			// Vertices
			{
				std::vector<Vector3r> positions;
				positions.reserve(n_vertices);
				for (int j = 0u; j < n_vertices; j++)
				{
					Vector3r x = vertices[j];
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
		}
	}

	m_isFirstFrame = false;
} */