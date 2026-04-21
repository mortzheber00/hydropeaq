cmake_minimum_required(VERSION 2.8.3)

project(amph)

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

find_package(roslaunch)

foreach(dir config launch meshes urdf worlds)
  install(DIRECTORY ${dir}/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)

install(TARGETS trajectory_replay_plugin
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)
