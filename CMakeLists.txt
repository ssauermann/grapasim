cmake_minimum_required(VERSION 3.10)

project(GraPaSim CXX)
if(SETUP_COMMAND)
    EXECUTE_PROCESS(COMMAND ${SETUP_COMMAND})
endif()

set( CMAKE_VERBOSE_MAKEFILE ON )
set(CUDA_VERBOSE_BUILD ON)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# let ccmake and cmake-gui offer the default build type options
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug;Release;RelWithDebInfo;MinSizeRel")

# set Release as the default build type if it is not yet set.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
            "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules/)

# include vectorization module
include(vectorization)
# include OpenMP module
include(OpenMP)
include(cuda)
# include options module
#include(options)
# include mpi module
#include(mpi)
# include vtk module
include_directories(SYSTEM lib/libxsd)
include(vtk)

# add mardyn
ADD_SUBDIRECTORY(src)