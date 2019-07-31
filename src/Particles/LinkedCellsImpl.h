#pragma once

#include <Domain/SFC.h>
#include "LinkedCells.h"

struct GPULayout;

class LinkedCellsImpl : public LinkedCells {

    SFC *decomp= nullptr;
    GPULayout *layout = nullptr;

    int GPU_N = 0;

protected:


    void init() override;

public:
    explicit LinkedCellsImpl(Domain &domain, Vector cellSizeTarget, std::vector<Particle> &particles);

    ~LinkedCellsImpl();

    void updateDecomp() override;
    void iteratePairs() override;
    void iterate() override;
    void preStep() override;
    void postStep() override;
    void prepareComputation() override;
    void finalizeComputation() override;

};