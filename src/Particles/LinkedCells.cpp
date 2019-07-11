#include "LinkedCells.h"

void LinkedCells::updateContainer() {
    // Clear cells
    for (auto &cell : this->cells) {
        cell->second.clear();
    }
    haloParticles.clear();

    // Re-sort particles into cells
    for (auto &p : this->particles) {
        auto idx = to1dIndex(cellIndex(p));
        cells.at(idx)->second.push_back(&p);
    }

    // Mirror boundary particles into halo cells for reflecting boundary
    // Updating positions and velocities is enough as forces do not matter
    for (auto &bc : this->boundary) {
        for (Particle *p: bc->second) {
            // if is boundary in +x
            if (p->x.x > domain.x.second - cellSize.x + 1e-5 ) {
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
/*
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
*/
            // TODO Diagonal mirroring might be necessary?
        }
    }

    // Re-sort halo particles into cells
    for (auto &p : this->haloParticles) {
        auto idx = to1dIndex(cellIndex(p));
        cells.at(idx)->second.push_back(&p);
    }

}

void LinkedCells::output(const std::function<void(Particle &)> & function, bool includeHalo) {
    for (auto it = this->particles.begin(); it < this->particles.end(); ++it) {
        function(*it);
    }
    if(includeHalo) {
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
