#include "VelocityFieldOnly.h"

void VelocityFieldOnly::calculate(Particle &particle) {
    //VECTOR newVelocity = {particle.x[1] - particle.x[0], -particle.x[0] - particle.x[1]};
    /*VECTOR newVelocity = {particle.x[1] * particle.x[1] * particle.x[1] - 9 * particle.x[1],
                          particle.x[0] * particle.x[0] * particle.x[0] - 6 * particle.x[0]};*/
    //VECTOR newVelocity = {particle.x[1], -particle.x[0]};

    auto sos = particle.x.x * particle.x.x + particle.x.y * particle.x.y;
    Vector newVelocity = {-particle.x.y / sos, particle.x.y / sos};

    if (sos == 0) {
        newVelocity.x = newVelocity.y = 0;
    }

    particle.v = newVelocity;
}

void VelocityFieldOnly::interact(Particle &particle1, Particle &particle2) {
    // No collision
}
