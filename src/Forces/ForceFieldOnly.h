#pragma once

#include "Forces.h"

class ForceFieldOnly : public Forces {
    PRECISION k = 4.0e+03; // spring constant
    PRECISION gamma = 4.0e-01; // damping coefficient

   // k_normal = 4.0e+03
   // c_normal = 4.0e-01
   // k_tangent = 1.6e+03
   // c_tangent = 2.5e-01


    PRECISION mu = 1; // friction coefficient

public:
    void calculate(Particle &particle) override;

    void interact(Particle &particle1, Particle &particle2) override;

};