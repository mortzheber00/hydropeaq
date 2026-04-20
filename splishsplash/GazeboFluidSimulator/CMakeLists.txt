set(SIMULATION_LINK_LIBRARIES SPlisHSPlasH)
set(SIMULATION_DEPENDENCIES SPlisHSPlasH)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

set(EXPORTER_SOURCE_FILES
	Exporter/ParticleExporter_Partio.cpp
	Exporter/ParticleExporter_VTK.cpp
	Exporter/RigidBodyExporter_BIN.cpp
	Exporter/RigidBodyExporter_OBJ.cpp
	Exporter/RigidBodyExporter_VTK.cpp
)

set(EXPORTER_HEADER_FILES
	Exporter/ExporterBase.h
	Exporter/ParticleExporter_Partio.h
	Exporter/ParticleExporter_VTK.h
	Exporter/RigidBodyExporter_BIN.h
	Exporter/RigidBodyExporter_OBJ.h
	Exporter/RigidBodyExporter_VTK.h
)


#eigen
find_package( Eigen3 REQUIRED )
include_directories( ${EIGEN3_INCLUDE_DIR} )
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Gazebo
find_package(gazebo REQUIRED)
include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

add_library(FluidSimulator SHARED
	FluidSimulator.h
	FluidSimulator.cpp
	GazeboExporterRegistration.cpp
	GazeboSceneConfiguration.cpp
	GazeboSceneConfiguration.h
	GazeboSimulatorBase.cpp
	GazeboSimulatorBase.h
	GazeboBoundarySimulator.cpp
	GazeboBoundarySimulator.h
	GazeboSceneLoader.h
	GazeboSceneLoader.cpp
	
	${EXPORTER_SOURCE_FILES}
	${EXPORTER_HEADER_FILES}
)
set_property(TARGET FluidSimulator PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(FluidSimulator PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
set_target_properties(FluidSimulator PROPERTIES RELWITHDEBINFO_POSTFIX ${CMAKE_RELWITHDEBINFO_POSTFIX})
set_target_properties(FluidSimulator PROPERTIES MINSIZEREL_POSTFIX ${CMAKE_MINSIZEREL_POSTFIX})
add_dependencies(FluidSimulator ${SIMULATION_DEPENDENCIES})

target_include_directories(FluidSimulator PUBLIC
	${PROJECT_SOURCE_DIR}
	${GENERICPARAMETERS_INCLUDE_DIR}
	${DISCREGRID_INCLUDE_DIR}
	${NEIGHBORHOOD_SEARCH_INCLUDE_DIR}
)
target_compile_definitions(FluidSimulator PRIVATE -DTW_NO_LIB_PRAGMA -DTW_STATIC)
set_property(TARGET FluidSimulator PROPERTY POSITION_INDEPENDENT_CODE ON)

set_target_properties(FluidSimulator PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
set_target_properties(FluidSimulator PROPERTIES RELWITHDEBINFO_POSTFIX ${CMAKE_RELWITHDEBINFO_POSTFIX})
set_target_properties(FluidSimulator PROPERTIES MINSIZEREL_POSTFIX ${CMAKE_MINSIZEREL_POSTFIX})
add_dependencies(FluidSimulator ${SIMULATION_DEPENDENCIES})
target_link_libraries(FluidSimulator ${SIMULATION_LINK_LIBRARIES} ${GAZEBO_LIBRARIES} ${IGNITION-MATH_LIBRARIES})

add_library(FluidVisPlugin SHARED visualization/FluidVisPlugin.cc)
target_link_libraries(FluidVisPlugin gazebo_msgs ${OGRE_LIBRARIES} gazebo_gui)

set(GAZEBO_PLUGIN_INSTALL_DIR
  ${CMAKE_INSTALL_PREFIX}/lib/gazebo-${GAZEBO_MAJOR_VERSION}/plugins
)

string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_tolower)
if( cmake_build_type_tolower STREQUAL "debug")
	install(FILES ${CMAKE_BINARY_DIR}/lib/libFluidSimulator${CMAKE_DEBUG_POSTFIX}.so DESTINATION ${GAZEBO_PLUGIN_INSTALL_DIR})
	install(FILES ${CMAKE_BINARY_DIR}/lib/libFluidVisPlugin${CMAKE_DEBUG_POSTFIX}.so DESTINATION ${GAZEBO_PLUGIN_INSTALL_DIR})
else()
	install(FILES ${CMAKE_BINARY_DIR}/lib/libFluidSimulator.so DESTINATION ${GAZEBO_PLUGIN_INSTALL_DIR})
	install(FILES ${CMAKE_BINARY_DIR}/lib/libFluidVisPlugin.so DESTINATION ${GAZEBO_PLUGIN_INSTALL_DIR})
endif()
