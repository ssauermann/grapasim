#include <cmath>
#include <algorithm>    // std::max
#include "ForceFieldOnly.h"

void ForceFieldOnly::calculate(Particle &particle) {
    // Apply gravity
    particle.F.y -= 9.80665;
}

void ForceFieldOnly::interact(Particle &particle1, Particle &particle2) {

    auto l2norm = (particle2.x - particle1.x).l2norm();

    // penetration depth
    auto xi = std::max((PRECISION)0, particle1.radius + particle2.radius - l2norm);

    // Normal vector
    auto N = (particle2.x - particle1.x) / l2norm;
    // Relative velocity
    auto V = particle1.v - particle2.v;

    // Relative velocity in normal direction
    auto xiDot = V * N;

    auto fn = - k * xi - gamma * xiDot;
    auto Fn = fn * N;

    particle1.F += Fn;

#ifdef SHEAR_FORCES
    // Tangential velocity
    auto Vt = V - xiDot * N;
    auto Ft = -mu * fn * Vt / Vt.l2norm();
    particle1.F += Ft;
#endif
}