#pragma once

#include <vector>
#include "../Constants.h"
#include "../Particles/Particle.h"

class Integrator {
protected:
    PRECISION stepsize;

    explicit Integrator(const PRECISION stepsize) : stepsize(stepsize) {
    }

public:

    void reverse() {
        this->stepsize = -this->stepsize;
    }

    virtual void doStepPreForce(Particle &particle) = 0;

    virtual void doStepPostForce(Particle &particle) = 0;

};
