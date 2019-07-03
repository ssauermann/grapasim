#pragma once

#include <string>
#include <vector>
#include "../Constants.h"
#include "../Particles/Particle.h"

class OutputWriter {
protected:
    const std::string &filename;

    OutputWriter(const std::string &filename) : filename(filename) {};

public:
    virtual void writeBegin(unsigned long iteration, int numParticles) = 0;

    virtual void plotParticle(const Particle &p) = 0;

    virtual void writeFinalize() = 0;
};

