#include <cmath>
#include "ForceFieldOnly.h"

void ForceFieldOnly::calculate(Particle &particle) {

}

void ForceFieldOnly::interact(Particle &particle1, Particle &particle2) {

    auto penetrationDepth = particle1.radius + particle2.radius - sqrt(l2Square(particle1.x, particle2.x));

    if(penetrationDepth > 0){
        for (int i = 0; i < DIMENSIONS; ++i) {
            particle1.F[i] += - this->k * penetrationDepth * distanceOfPenetration[i] - this->gamma * velocityOfPenetration[i];
        }
    }

}

PRECISION ForceFieldOnly::l2Square(VECTOR a, VECTOR b) {

    PRECISION sum = 0;

    for (int i = 0; i < DIMENSIONS; ++i) {
        auto diff = (a[i] - b[i]);
        sum += diff * diff;
    }

    return sum;
}
