cmake_minimum_required(VERSION 3.0)
# project(run_graspabiltiy)

set(SOURCES
  src/main.cpp
  src/Graspability.cpp
)

project(run_graspability)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(include)
add_executable(run_graspability ${SOURCES})
target_link_libraries( run_graspability ${OpenCV_LIBS} )
