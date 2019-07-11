#include "LinkedCellsImpl.h"


void LinkedCellsImpl::prepareComputation() {
    // Nothing to prepare
}

void LinkedCellsImpl::finalizeComputation() {
    // Nothing to finalize
}

void LinkedCellsImpl::iteratePairs() {
#pragma omp parallel for schedule(dynamic, 1)
    for(auto it = this->inner.begin(); it < this->inner.end(); ++it){
        auto& cell = *it;
        for (int offset: pairOffsets) {
            for (Particle *p: cell->second) {
                for (Particle *q: cells.at(cell->first + offset)->second) {
                    if(p != q) {
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


void  LinkedCellsImpl::preStep(){
#pragma omp parallel for
    for (auto it = this->particles.begin(); it < this->particles.end(); ++it) {
        Leapfrog::doStepPreForce(*it);
    }
}

void  LinkedCellsImpl::postStep(){
#pragma omp parallel for
    for (auto it = this->particles.begin(); it < this->particles.end(); ++it) {
        Leapfrog::doStepPostForce(*it);
    }
}