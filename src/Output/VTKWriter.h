#pragma once

#include <memory>
#include "OutputWriter.h"

class VTKFile_t;
struct VTKFile_tDeleter
{
    void operator()(VTKFile_t *p);
};

class VTKWriter : public OutputWriter {

    std::unique_ptr<VTKFile_t, VTKFile_tDeleter> vtkFile;

    std::unique_ptr<VTKFile_t, VTKFile_tDeleter> parallelVTKFile;

    unsigned int rank;


public:
    VTKWriter(const std::vector<Particle> &particles, const std::string &filename, const unsigned int rank)
            : OutputWriter(particles, filename), rank(rank) {}

    void write(unsigned long iteration) override;

    void plotParticle(const Particle &particle);

    void initializeVTKFile();


    void initializeParallelVTKFile(const std::vector<std::string> &fileNames);

};
