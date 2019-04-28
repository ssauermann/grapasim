#pragma once

#include "Forces.h"

class VelocityFieldOnly : public Forces {


public:

    void calculate(Particle &particle) override;

    void interact(Particle &particle1, Particle &particle2) override;
};


