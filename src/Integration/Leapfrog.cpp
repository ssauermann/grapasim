#include "Leapfrog.h"

void Leapfrog::doStepPreForce(Particle &particle) {
    // Half-step velocity to get v(t+0.5)
    particle.v += 0.5 * this->stepsize * particle.F / particle.mass;

    // Update positions
    particle.x += this->stepsize * particle.v;

}

void Leapfrog::doStepPostForce(Particle &particle) {
    // Half-step velocity to get v(t+1)
    particle.v += 0.5 * this->stepsize * particle.F / particle.mass;

}
