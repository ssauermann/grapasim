#pragma once

#include "LinkedCells.h"

struct GPULayout;

class LinkedCellsImpl : public LinkedCells {

    GPULayout *layout = nullptr;

    int GPU_N = 0;

protected:

    void updateDecomp() override;

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