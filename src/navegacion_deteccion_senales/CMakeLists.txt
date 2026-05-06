cmake_minimum_required(VERSION 3.8)
project(navegacion_deteccion_senales)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/lane_follower.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()