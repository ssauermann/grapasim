#pragma once

#include "Integrator.h"

class Leapfrog : public Integrator {


public:
    explicit Leapfrog(const PRECISION stepSize) : Integrator(stepSize) {}

    void doStepPreForce(Particle &particle) override;

    void doStepPostForce(Particle &particle) override;

};