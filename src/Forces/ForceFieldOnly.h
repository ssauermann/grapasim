#pragma once

#include "Forces.h"

class ForceFieldOnly : public Forces {
    PRECISION k = 1; // spring constant
    PRECISION gamma = 1; // dumping coefficient

    PRECISION l2Square(VECTOR a, VECTOR b);

public:
    void calculate(Particle &particle) override;

    void interact(Particle &particle1, Particle &particle2) override;

};