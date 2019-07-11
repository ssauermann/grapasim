#include "LinkedCells.h"
#ifdef ENABLE_CUDA
#include "LinkedCells.cu"
#endif

void LinkedCells::prepareComputation() {
#ifdef ENABLE_CUDA
    int N = this->particles.size();
// Copy host to device
    HANDLE_ERROR( cudaMalloc( (void**)&this->device_particles, sizeof(Particle) ) );
    HANDLE_ERROR( cudaMemcpy(
            this->device_particles,
            this->particles.data(),
            sizeof(Particle) * N,
            cudaMemcpyHostToDevice ) );
#endif
}

void LinkedCells::finalizeComputation() {
    int N = this->particles.size();
#ifdef ENABLE_CUDA
    HANDLE_ERROR( cudaMemcpy(
            this->particles,
            this->device_particles,
            sizeof(Particle) * N,
            cudaMemcpyDeviceToHost ) );
    cudaFree(this->device_particles);
#endif
}

void LinkedCells::iteratePairs() {
#ifdef ENABLE_CUDA

#else
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
#endif
}

void LinkedCells::iterate() {
#ifdef ENABLE_CUDA
    int N = this->particles.size();


  vec2,
  vec1.data(),
  numMoments * sizeof(double));

    iterateKernel<<<(N + 511) / 512, 512>>>(N, particles);

#else
#pragma omp parallel for
    for (auto it = this->particles.begin(); it < this->particles.end(); ++it) {
        SpringForce::calculate(*it);
    }
#endif
}


void  LinkedCells::preStep(){
#pragma omp parallel for
    for (auto it = this->particles.begin(); it < this->particles.end(); ++it) {
        Leapfrog::doStepPreForce(*it);
    }
}

void  LinkedCells::postStep(){
#pragma omp parallel for
    for (auto it = this->particles.begin(); it < this->particles.end(); ++it) {
        Leapfrog::doStepPostForce(*it);
    }
}

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
