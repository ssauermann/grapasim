#pragma once

#include <Particles/Particle.h>
#include "Constants.h"

static class Leapfrog {

public:
    static PRECISION stepSize;

    DEVICE static void doStepPreForce(Particle &particle){
        // Half-step velocity to get v(t+0.5)
        particle.v += 0.5 * stepSize * particle.F / particle.mass;

        // Update positions
        particle.x += stepSize * particle.v;

        particle.F = {0};
    }

    DEVICE static void doStepPostForce(Particle &particle){

        // Half-step velocity to get v(t+1)
        particle.v += 0.5 * stepSize * particle.F / particle.mass;
    }

};