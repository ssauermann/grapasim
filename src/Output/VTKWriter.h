#pragma once


#include "OutputWriter.h"
#include "VTK/vtk-unstructured.h"
#include "VTK/vtk-punstructured.h"

class VTKWriter : public OutputWriter {

    VTKFile_t &_vtkFile;

    VTKFile_t &_parallelVTKFile;

    int _rank;


public:
    VTKWriter(const std::vector<Particle> &particles, const std::string &filename) : OutputWriter(particles,
                                                                                                  filename) {}

    void write(unsigned long iteration) override;

    void plotParticle(Particle& particle);

    void initializeVTKFile();


    void initializeParallelVTKFile(const std::vector<std::string>& fileNames);

};
