#pragma once

#include <vector>
#include <Particles/Particle.h>

class Generator {

public:
    unsigned int dimensions = DIMENSIONS;
    PRECISION mesh = 1;
    PRECISION mass = 1;
    Vector initialVelocity{0};

    virtual void generate(std::vector<Particle> &particles) = 0;
};
