#include "Leapfrog.h"

PRECISION Leapfrog::stepSize = 1e-5;

__device__ void Leapfrog::doStepPreForce(Particle &particle) {
    // Half-step velocity to get v(t+0.5)
    particle.v += 0.5 * stepSize * particle.F / particle.mass;

    // Update positions
    particle.x += stepSize * particle.v;

    particle.F = {0};
}

__device__ void Leapfrog::doStepPostForce(Particle &particle) {

    // Half-step velocity to get v(t+1)
    particle.v += 0.5 * stepSize * particle.F / particle.mass;
}
