#pragma once

#include <string>
#include <vector>
#include "../Constants.h"
#include "../Particles/Particle.h"

class OutputWriter {
protected:
    const std::string &filename;
    const std::vector<Particle> &particles;

    OutputWriter(const std::vector<Particle> &particles, const std::string &filename) : particles(particles),
                                                                                        filename(filename) {}

public:
    virtual void write(unsigned long iteration) = 0;
};

