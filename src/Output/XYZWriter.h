#pragma once

#include <fstream>
#include "../Constants.h"
#include "OutputWriter.h"

class XYZWriter : public OutputWriter {
    std::ofstream file;

public:

    XYZWriter(const std::string &filename) : OutputWriter(filename) {}

    void writeBegin(unsigned long iteration, int numParticles) override;

    void plotParticle(const Particle &p) override;

    void writeFinalize() override;

};

