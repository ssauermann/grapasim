#pragma once

#include "ParticleContainer.h"

class LinkedCells : public ParticleContainer {

    std::vector<Particle>& particles;

public:
    explicit LinkedCells(std::vector<Particle>& particles) : particles(particles){
    }

    void iteratePairs(const std::function<void(Particle&, Particle&)> &function) override;

    void iterate(const std::function<void(Particle&)> &function) override;
};

