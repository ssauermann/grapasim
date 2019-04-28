#include "VelocityFieldOnly.h"

void VelocityFieldOnly::calculate(Particle &particle) {
    VECTOR newVelocity = {particle.x[1] - particle.x[0], -particle.x[0] - particle.x[1]};

    // Update forces based on velocity field
    for (unsigned int i = 0; i < DIMENSIONS; ++i) {
        particle.F[i] = newVelocity[i] - particle.v[i];
    }
}

void VelocityFieldOnly::interact(Particle &particle1, Particle &particle2) {
    // No collision
}
