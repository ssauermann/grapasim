#pragma once

#include "LinkedCells.h"

class LinkedCellsImpl : public LinkedCells {

public:
    explicit LinkedCellsImpl(Domain &domain, Vector cellSizeTarget, std::vector<Particle> &particles)
    : LinkedCells(domain, cellSizeTarget, particles) {}

    void iteratePairs() override;
    void iterate() override;
    void preStep() override;
    void postStep() override;
    void prepareComputation() override;
    void finalizeComputation() override;

};