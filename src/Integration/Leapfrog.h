#pragma once

#include <Particles/Particle.h>
#include "Constants.h"

class Leapfrog {

public:
    static constexpr PRECISION stepSize = 1.25e-06;

    DEVICE static void doStepPreForce(Particle &particle);

    DEVICE static void doStepPostForce(Particle &particle);

};