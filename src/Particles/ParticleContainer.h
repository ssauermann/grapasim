#pragma once

#include <vector>
#include <functional>
#include "Particle.h"

class ParticleContainer {

public:
    virtual void updateContainer() = 0;

    virtual void iteratePairs(const std::function<void(Particle&, Particle&)>&) = 0;

    virtual void iterate(const std::function<void(Particle&)>&) = 0;

    virtual void iterateAll(const std::function<void(Particle&)>&) = 0;

    virtual int particleCount(bool includeVirtual) = 0;


};