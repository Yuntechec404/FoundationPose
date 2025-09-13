cmake_minimum_required(VERSION 3.0.2)
project(ros_foundationpose)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  message_filters
  image_transport
  cv_bridge
  tf
  message_generation
)

##############
## Messages ##
##############

add_message_files(
  FILES
  Confidence.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    message_filters
    image_transport
    cv_bridge
    tf
    message_runtime
)


include_directories(${catkin_INCLUDE_DIRS})

# catkin_install_python(PROGRAMS
#   scripts/foundationpose_tracker.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

