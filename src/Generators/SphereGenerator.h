#pragma once

#include <memory>
#include "Generator.h"

class SphereGenerator : public Generator {

    void doGenerate(const Vector &corner,
                    const std::array<unsigned int, 3> &particleNumbers,
                    std::vector<Particle> &particles,
                    const std::shared_ptr<std::array<int, 3>> &indices,
                    unsigned int nesting = 0);

public:

    unsigned int radius = 1;
    Vector center{0};
    PRECISION size = 1;
    int type = 0;

    explicit SphereGenerator(unsigned int radius) : radius(radius) {
    }

    void generate(std::vector<Particle> &particles) override;


};