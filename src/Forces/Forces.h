#pragma once

#include "../Particles/Particle.h"

class Forces {

public:
    virtual void calculate(Particle& particle) = 0;

    virtual void interact(Particle& particle1, Particle& particle2) = 0;

};