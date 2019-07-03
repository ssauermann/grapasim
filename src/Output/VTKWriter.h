#pragma once

#include <memory>
#include "OutputWriter.h"

class VTKFile_t;

struct VTKFile_tDeleter {
    void operator()(VTKFile_t *p);
};

class VTKWriter : public OutputWriter {

    std::unique_ptr<VTKFile_t, VTKFile_tDeleter> vtkFile;

    std::unique_ptr<VTKFile_t, VTKFile_tDeleter> parallelVTKFile;

    unsigned int rank;

    std::string fileName;

    void initializeVTKFile();

    void initializeParallelVTKFile(const std::vector<std::string> &fileNames);

public:
    VTKWriter(const std::string &filename, const unsigned int rank)
            : OutputWriter(filename), rank(rank) {}

    void writeBegin(unsigned long iteration, int numParticles) override;

    void plotParticle(const Particle &particle) override;

    void writeFinalize() override;


};
