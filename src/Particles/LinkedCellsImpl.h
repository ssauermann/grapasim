#pragma once

#include "LinkedCells.h"

class LinkedCellsImpl : public LinkedCells {

    Particle *deviceParticles = nullptr;
    Particle *deviceHaloParticles = nullptr;
    int *deviceInner = nullptr;
    int *devicePairOffsets = nullptr;
    Cell *deviceCells = nullptr;

public:
    explicit LinkedCellsImpl(Domain &domain, Vector cellSizeTarget, std::vector<Particle> &particles);

    ~LinkedCellsImpl();

    void iteratePairs() override;
    void iterate() override;
    void preStep() override;
    void postStep() override;
    void prepareComputation() override;
    void finalizeComputation() override;

};