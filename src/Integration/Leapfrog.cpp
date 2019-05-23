#include "Leapfrog.h"

void Leapfrog::doStepPreForce(Particle &particle) {
    // Half-step velocity to get v(t+0.5)
    for (unsigned int i = 0; i < DIMENSIONS; ++i) {
        particle.v[i] += 0.5 * this->stepsize * particle.F[i] / particle.mass;;
    }

    // Update positions
    for (unsigned int i = 0; i < DIMENSIONS; ++i) {
        particle.x[i] += this->stepsize * particle.v[i];
    }
}

void Leapfrog::doStepPostForce(Particle &particle) {
    // Half-step velocity to get v(t+1)
    for (unsigned int i = 0; i < DIMENSIONS; ++i) {
        particle.v[i] += 0.5 * this->stepsize * particle.F[i] / particle.mass;
    }
}
