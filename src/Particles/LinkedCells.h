#pragma once

#include <Domain/Domain.h>
#include <tuple>
#include <iostream>
#include <cassert>
#include <vector>
#include <Forces/SpringForce.h>
#include "Vector.h"

#include "IntVector.h"
#include "Particle.h"
#include <functional>
#include <Integration/Leapfrog.h>

#define MAXCELLPARTICLE 8

struct Cell {
    int size = 0;
    int data[MAXCELLPARTICLE] = {0};

    void pushBack(int x) {
        assert(size < MAXCELLPARTICLE - 1);
        data[size++] = x;
    }
};

class LinkedCells {
protected:
    std::vector<Particle> particles;

    std::vector<Particle> haloParticles;
    Domain domain;
    Vector cellSize = {0, 0, 0};
    IntVector numCells = {0, 0, 0};

    //typedef std::vector<int> Cell;


    std::vector<Cell> cells;
    std::vector<int> halo;
    std::vector<int> boundary;
    std::vector<int> inner;

    std::vector<int> pairOffsets;

    PRECISION fitCells(PRECISION domainSpace, PRECISION cellSizeTarget, int *numberOfCells) {
        int numCellsTarget = (int) (domainSpace / cellSizeTarget);
        PRECISION freeDomain = domainSpace - (numCellsTarget * cellSizeTarget);
        *numberOfCells = numCellsTarget + 2; // Two halo cells in each dimension
        return cellSizeTarget + freeDomain / numCellsTarget;
    }

    virtual void init() = 0;

    IntVector cellIndex(Particle &p) {
        IntVector idx = {0};
        idx.x = (int) ((p.x.x - domain.x.first) / cellSize.x + 1);
        idx.y = (int) ((p.x.y - domain.y.first) / cellSize.y + 1);
        idx.z = (int) ((p.x.z - domain.z.first) / cellSize.z + 1);

#ifndef NDEBUG
        if (p.type >= 0) {
            assert(idx.x > 0 && idx.x < numCells.x - 1);
            assert(idx.y > 0 && idx.y < numCells.y - 1);
            assert(idx.z > 0 && idx.z < numCells.z - 1);
        } else {
            assert(idx.x == 0 || idx.x == numCells.x - 1 ||
                   idx.y == 0 || idx.y == numCells.y - 1 ||
                   idx.z == 0 || idx.z == numCells.z - 1);
        }
#endif

        return idx;
    }

    int to1dIndex(IntVector multiIndex) {
        auto idx = multiIndex.x * numCells.y * numCells.z + multiIndex.y * numCells.z + multiIndex.z;
        return idx;
    }



public:

    LinkedCells(const LinkedCells &) = delete;

    explicit LinkedCells(Domain &domain, Vector cellSizeTarget, std::vector<Particle> &particles) : particles(
            particles), domain(domain) {
        // initialize cells
        this->cellSize.x = fitCells(domain.x.second - domain.x.first, cellSizeTarget.x, &numCells.x);
        this->cellSize.y = fitCells(domain.y.second - domain.y.first, cellSizeTarget.y, &numCells.y);
        this->cellSize.z = fitCells(domain.z.second - domain.z.first, cellSizeTarget.z, &numCells.z);

        for (int i = 0; i < numCells.x; ++i) {
            for (int j = 0; j < numCells.y; ++j) {
                for (int k = 0; k < numCells.z; ++k) {
                    Cell cell{};
                    int idx = this->cells.size();
                    assert(to1dIndex({i, j, k}) == idx);
                    this->cells.push_back(cell);

                    // Test if halo block and add to halo subvector
                    if (i == 0 || j == 0 || k == 0 ||
                        i == (numCells.x - 1) || j == (numCells.y - 1) || k == (numCells.z - 1)) {
                        halo.push_back(idx);
                    } else {
                        // Test if boundary block and add to boundary subvector
                        if (i == 1 || j == 1 || k == 1 ||
                            i == (numCells.x - 2) || j == (numCells.y - 2) || k == (numCells.z - 2)) {
                            boundary.push_back(idx);
                        }
                        // Inner cell == non halo cell
                        inner.push_back(idx);

                    }

                }
            }
        }

        pairOffsets = calculatePairs();

        std::cout << "Linked cells initialized with (" << numCells.x << ", " << numCells.y << ", " << numCells.z
                  << ") cells\n";

        // Sanity check
        for (auto &p: particles) {
            // Assert that no particle is outside of the domain (not even in halo)
            cellIndex(p); //happens in here
        }

    }

    std::vector<int> calculatePairs() {
        // Precompute the cell offsets in 1d
        auto pairs = std::vector<int>();
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                for (int k = -1; k <= 1; ++k) {
                    pairs.push_back(to1dIndex({i, j, k}));
                }
            }
        }
        return pairs;
    }

    void updateContainer();
    virtual void updateDecomp() = 0;

    void output(const std::function<void(Particle &)> &, bool includeHalo = false);

    int particleCount(bool includeVirtual);

    virtual void iteratePairs() = 0;

    virtual void iterate() = 0;

    virtual void preStep() = 0;

    virtual void postStep() = 0;

    virtual void prepareComputation() = 0;

    virtual void finalizeComputation() = 0;
};

