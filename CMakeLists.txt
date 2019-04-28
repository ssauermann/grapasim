# cmake_minimum_required(VERSION <specify CMake version here>)
project(GraPaSim)

set(CMAKE_CXX_STANDARD 17)

add_executable(GraPaSim main.cpp Simulation.cpp Simulation.h Particles/Particle.h Constants.h Integration/Integrator.h Integration/Leapfrog.cpp Integration/Leapfrog.h Particles/ParticleContainer.h Particles/LinkedCells.cpp Particles/LinkedCells.h Forces/Forces.h Forces/VelocityFieldOnly.cpp Forces/VelocityFieldOnly.h Output/OutputWriter.h Output/XYZWriter.h Output/XYZWriter.cpp)