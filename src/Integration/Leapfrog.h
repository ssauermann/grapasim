#pragma once

#include <Particles/Particle.h>
#include "Constants.h"

class Leapfrog {

public:
    static PRECISION stepSize;

    static void doStepPreForce(Particle &particle);

    static void doStepPostForce(Particle &particle);

};