cmake_minimum_required(VERSION 2.8.3)

project(sph_replay)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  trajectory_msgs
)

find_package(gazebo REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
)

link_directories(${GAZEBO_LIBRARY_DIRS})

add_library(trajectory_replay_plugin src/trajectory_replay_plugin.cpp)
target_link_libraries(trajectory_replay_plugin
  ${catkin_LIBRARIES}
  ${GAZEBO_LIBRARIES}
)

catkin_install_python(PROGRAMS
  scripts/replay_trajectory.py
  scripts/generate_boundary_obj.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS trajectory_replay_plugin
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)
