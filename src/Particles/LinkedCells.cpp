#include "LinkedCells.h"

void LinkedCells::iteratePairs(const std::function<void(Particle &, Particle &)> &function) {

    for (auto &cell: inner) {
        for (int offset: pairOffsets) {
            for (Particle *p: cell.second) {
                for (Particle *q: cells.at(cell.first + offset).second) {
                    function(*p, *q);
                }
            }
        }
    }

    /* for (int i = 0; i < this->particles.size(); ++i) {
         for (int j = 0; j < this->particles.size(); ++j) {
             if (i != j) {
                 function(this->particles.at(i), this->particles.at(j));
             }
         }
     }*/
}

void LinkedCells::iterate(const std::function<void(Particle &)> &function) {
#pragma omp parallel for
    for (auto it = this->particles.begin(); it < this->particles.end(); ++it) {
        function(*it);
    }
}

void LinkedCells::updateContainer() {
    // Clear cells
    for (auto &cell : this->cells) {
        cell.second.clear();
    }
    haloParticles.clear();

    // Re-sort particles into cells
    for (auto &p : this->particles) {
        auto idx = to1dIndex(cellIndex(p));
        cells.at(idx).second.push_back(&p);
    }

    // Mirror boundary particles into halo cells for reflecting boundary
    // Updating positions and velocities is enough as forces do not matter
    for (auto &bc : this->boundary) {
        for (Particle *p: bc.second) {
            // if is boundary in +x
            if (p->x.x >= domain.x.second - cellSize.x) {
                Particle copy = *p;
                copy.id = -1;
                copy.x.x += 2 * (domain.x.second - p->x.x);
                copy.v.x *= -1;
                this->haloParticles.push_back(copy);
            }
            // if is boundary in -x
            if (p->x.x < domain.x.first + cellSize.x) {
                Particle copy = *p;
                copy.id = -1;
                copy.x.x -= 2 * (p->x.x - domain.x.first);
                copy.v.x *= -1;
                this->haloParticles.push_back(copy);
            }

            // if is boundary in +y
            if (p->x.y >= domain.y.second - cellSize.y) {
                Particle copy = *p;
                copy.id = -1;
                copy.x.y += 2 * (domain.y.second - p->x.y);
                copy.v.y *= -1;
                this->haloParticles.push_back(copy);
            }
            // if is boundary in -y
            if (p->x.y < domain.y.first + cellSize.y) {
                Particle copy = *p;
                copy.id = -1;
                copy.x.y -= 2 * (p->x.y - domain.y.first);
                copy.v.y *= -1;
                this->haloParticles.push_back(copy);
            }

            // if is boundary in +z
            if (p->x.z >= domain.z.second - cellSize.z) {
                Particle copy = *p;
                copy.id = -1;
                copy.x.z += 2 * (domain.z.second - p->x.z);
                copy.v.z *= -1;
                this->haloParticles.push_back(copy);
            }
            // if is boundary in -z
            if (p->x.z < domain.z.first + cellSize.z) {
                Particle copy = *p;
                copy.id = -1;
                copy.x.z -= 2 * (p->x.z - domain.z.first);
                copy.v.z *= -1;
                this->haloParticles.push_back(copy);
            }

            // TODO Diagonal mirroring might be necessary?
        }
    }

    // Re-sort halo particles into cells
    for (auto &p : this->haloParticles) {
        auto idx = to1dIndex(cellIndex(p));
        cells.at(idx).second.push_back(&p);
    }

}

void LinkedCells::iterateAll(const std::function<void(Particle &)> &function) {
    for (auto it = this->particles.begin(); it < this->particles.end(); ++it) {
        function(*it);
    }
    for (auto it = this->haloParticles.begin(); it < this->haloParticles.end(); ++it) {
        function(*it);
    }
}

int LinkedCells::particleCount(bool includeVirtual) {
    int count = particles.size();
    if (includeVirtual) {
        count += haloParticles.size();
    }
    return count;
}
