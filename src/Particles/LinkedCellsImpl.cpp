#include "LinkedCellsImpl.h"

struct GPULayout {

};

void LinkedCellsImpl::prepareComputation() {
    // Nothing to prepare
}

void LinkedCellsImpl::finalizeComputation() {
    // Nothing to finalize
}

void LinkedCellsImpl::iteratePairs() {
#pragma omp parallel for // schedule(dynamic, 1)
    for (auto it = this->inner.begin(); it < this->inner.end(); ++it) {
        auto &cell = this->cells.at(*it);
        for (int offset: pairOffsets) {
            for (int i=0; i<cell.size; ++i) {
                auto pIdx = cell.data[i];
                auto& otherCell = cells.at(*it + offset);
                for (int j=0; j<otherCell.size; ++j) {
                    auto qIdx = otherCell.data[j];
                    if (pIdx != qIdx) {
                        Particle *p;
                        Particle *q;
                        if (pIdx >= 0) {
                            p = &this->particles[pIdx];
                        } else {
                            p = &this->haloParticles[-pIdx-1];
                        }
                        if (qIdx >= 0) {
                            q = &this->particles[qIdx];
                        } else {
                            q = &this->haloParticles[-qIdx-1];
                        }
                        SpringForce::interact(*p, *q);
                    }
                }
            }
        }
    }
}

void LinkedCellsImpl::iterate() {
#pragma omp parallel for
    for (auto it = this->particles.begin(); it < this->particles.end(); ++it) {
        SpringForce::calculate(*it);
    }
}


void LinkedCellsImpl::preStep() {
#pragma omp parallel for
    for (auto it = this->particles.begin(); it < this->particles.end(); ++it) {
        Leapfrog::doStepPreForce(*it);
    }
}

void LinkedCellsImpl::postStep() {
#pragma omp parallel for
    for (auto it = this->particles.begin(); it < this->particles.end(); ++it) {
        Leapfrog::doStepPostForce(*it);
    }
}

LinkedCellsImpl::LinkedCellsImpl(Domain &domain, Vector cellSizeTarget, std::vector<Particle> &particles) : LinkedCells(
        domain, cellSizeTarget, particles) {
    // Nothing to setup
}


LinkedCellsImpl::~LinkedCellsImpl() = default;

void LinkedCellsImpl::updateDecomp() {
    // No decomposition for CPU variant
}

void LinkedCellsImpl::init() {
    //haloParticles = std::vector<Particle>;
}