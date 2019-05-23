#include "VelocityFieldOnly.h"

void VelocityFieldOnly::calculate(Particle &particle) {
    //VECTOR newVelocity = {particle.x[1] - particle.x[0], -particle.x[0] - particle.x[1]};
    /*VECTOR newVelocity = {particle.x[1] * particle.x[1] * particle.x[1] - 9 * particle.x[1],
                          particle.x[0] * particle.x[0] * particle.x[0] - 6 * particle.x[0]};*/
    //VECTOR newVelocity = {particle.x[1], -particle.x[0]};

    auto sos = particle.x[0] * particle.x[0] + particle.x[1] * particle.x[1];
    VECTOR newVelocity = {-particle.x[1] / sos, particle.x[0] / sos};

    if (sos == 0) {
        newVelocity[0] = newVelocity[1] = 0;
    }


    // Update forces based on velocity field
    for (unsigned int i = 0; i < DIMENSIONS; ++i) {
        //particle.F[i] = (newVelocity[i] - particle.v[i]);

        // particle.v[i] += 0.5 * this->stepsize * particle.F[i];
        // particle.v[i] += 0.5 * this->stepsize * (newVelocity[i] - particle.v[i]);

        particle.v[i] = newVelocity[i];
    }
}

void VelocityFieldOnly::interact(Particle &particle1, Particle &particle2) {
    // No collision
}
