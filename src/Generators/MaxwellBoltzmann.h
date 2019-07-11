#pragma once


#include <cstdlib>
#include <Particles/Particle.h>
#include <random>

class MaxwellBoltzmannDistribution {
    std::random_device rd;
    std::seed_seq ssq{rd()};
    std::mt19937 gen{rd()};
    std::normal_distribution<> gauss{0, 1};

    double factor;
    int dimensions;
public:
    explicit MaxwellBoltzmannDistribution(PRECISION factor, int dimensions) : factor(factor), dimensions(dimensions) {

    }

    void apply(Particle &p) {

        p.v.x += factor * gauss(gen);
        p.v.y += factor * gauss(gen);

        if (dimensions > 2) {
            p.v.z += factor * gauss(gen);
        }

    }

};

