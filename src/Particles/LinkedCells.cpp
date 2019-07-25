#include "LinkedCells.h"

void LinkedCells::updateContainer() {
    // Clear cells
    for (auto &cell : this->cells) {
        cell.size = 0;
    }
    haloParticles.clear();

    // Re-sort particles into cells
    for (int i = 0; i < this->particles.size(); ++i) {
        auto &p = this->particles[i];
        const auto &cellIdx = cellIndex(p);

        assert(cellIdx.x > 0 && cellIdx.x < numCells.x - 1);
        assert(cellIdx.y > 0 && cellIdx.y < numCells.y - 1);
        assert(cellIdx.z > 0 && cellIdx.z < numCells.z - 1);

        auto idx = to1dIndex(cellIdx);
        cells.at(idx).pushBack(i);
    }

    // Mirror boundary particles into halo cells for reflecting boundary
    // Updating positions and velocities is enough as forces do not matter
    for (auto &bc : this->boundary) {
        auto &cell = this->cells.at(bc);
        for (int i = 0; i < cell.size; ++i) {
            int pIdx = cell.data[i];
            Particle *p = &this->particles[pIdx];
            // if is boundary in +x
            if (p->x.x > domain.x.second - cellSize.x + 1e-5) {
                Particle copy = *p;
                copy.type = -1;
                copy.x.x += 2 * (domain.x.second - p->x.x);
                copy.v.x *= -1;
                this->haloParticles.push_back(copy);
            }
            // if is boundary in -x
            if (p->x.x < domain.x.first + cellSize.x - 1e-5) {
                Particle copy = *p;
                copy.type = -1;
                copy.x.x -= 2 * (p->x.x - domain.x.first);
                copy.v.x *= -1;
                this->haloParticles.push_back(copy);
            }

            // if is boundary in +y
            if (p->x.y > domain.y.second - cellSize.y + 1e-5) {
                Particle copy = *p;
                copy.type = -1;
                copy.x.y += 2 * (domain.y.second - p->x.y);
                copy.v.y *= -1;
                this->haloParticles.push_back(copy);
            }
            // if is boundary in -y
            if (p->x.y < domain.y.first + cellSize.y - 1e-5) {
                Particle copy = *p;
                copy.type = -1;
                copy.x.y -= 2 * (p->x.y - domain.y.first);
                copy.v.y *= -1;
                this->haloParticles.push_back(copy);
            }

            if(this->numCells.z > 3) {
                // if is boundary in +z
                if (p->x.z > domain.z.second - cellSize.z + 1e-5) {
                    Particle copy = *p;
                    copy.type = -1;
                    copy.x.z += 2 * (domain.z.second - p->x.z);
                    copy.v.z *= -1;
                    this->haloParticles.push_back(copy);
                }
                // if is boundary in -z
                if (p->x.z < domain.z.first + cellSize.z - 1e-5) {
                    Particle copy = *p;
                    copy.type = -1;
                    copy.x.z -= 2 * (p->x.z - domain.z.first);
                    copy.v.z *= -1;
                    this->haloParticles.push_back(copy);
                }
            }
            // TODO Diagonal mirroring might be necessary?
        }
        updateDecomp();
    }

    // Re-sort halo particles into cells
    for (int pIdx = 0; pIdx < this->haloParticles.size(); ++pIdx) {
        Particle &p = this->haloParticles[pIdx];
        auto idx = to1dIndex(cellIndex(p));
        cells.at(idx).pushBack(-pIdx-1);
    }

}

void LinkedCells::output(const std::function<void(Particle &)> &function, bool includeHalo) {
    for (auto it = this->particles.begin(); it < this->particles.end(); ++it) {
        function(*it);
    }
    if (includeHalo) {
        for (auto it = this->haloParticles.begin(); it < this->haloParticles.end(); ++it) {
            function(*it);
        }
    }
}

int LinkedCells::particleCount(bool includeVirtual) {
    int count = particles.size();
    if (includeVirtual) {
        count += haloParticles.size();
    }
    return count;
}
