cmake_minimum_required(VERSION 3.10)
project(GraPaSim)

set(CMAKE_CXX_STANDARD 17)

add_executable(GraPaSim src/main.cpp src/Simulation.cpp src/Simulation.h src/Particles/Particle.h src/Constants.h src/Integration/Integrator.h src/Integration/Leapfrog.cpp src/Integration/Leapfrog.h src/Particles/ParticleContainer.h src/Particles/LinkedCells.cpp src/Particles/LinkedCells.h src/Forces/Forces.h src/Forces/VelocityFieldOnly.cpp src/Forces/VelocityFieldOnly.h src/Output/OutputWriter.h src/Output/XYZWriter.h src/Output/XYZWriter.cpp src/Output/VTKWriter.cpp src/Output/VTKWriter.h)


find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include(FindXercesC)
find_package(XercesC REQUIRED)
include_directories( ${XercesC_INCLUDE_DIR} )


target_link_libraries ( ${PROJECT_NAME} ${XercesC_LIBRARY} )