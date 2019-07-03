#pragma once

#include "Forces.h"

class ForceFieldOnly : public Forces {
    PRECISION k = 1; // spring constant
    PRECISION gamma = 1; // dumping coefficient
    PRECISION mu = 1; // friction coefficient

public:
    void calculate(Particle &particle) override;

    void interact(Particle &particle1, Particle &particle2) override;

};