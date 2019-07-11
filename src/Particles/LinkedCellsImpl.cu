#include "LinkedCellsImpl.h"


void LinkedCells::prepareComputation() {
    int N = this->particles.size();
// Copy host to device
    HANDLE_ERROR( cudaMalloc( (void**)&this->device_particles, sizeof(Particle) ) );
    HANDLE_ERROR( cudaMemcpy(
            this->device_particles,
            this->particles.data(),
            sizeof(Particle) * N,
            cudaMemcpyHostToDevice ) );
}

void LinkedCells::finalizeComputation() {
    int N = this->particles.size();
    HANDLE_ERROR( cudaMemcpy(
            this->particles,
            this->device_particles,
            sizeof(Particle) * N,
            cudaMemcpyDeviceToHost ) );
    cudaFree(this->device_particles);
}

void LinkedCells::iteratePairs() {
//TODO
}

void LinkedCells::iterate() {
    int N = this->particles.size();


  vec2,
  vec1.data(),
  numMoments * sizeof(double));

    iterateKernel<<<(N + 511) / 512, 512>>>(N, particles);

}


void  LinkedCells::preStep(){
//TODO
}

void  LinkedCells::postStep(){
//TODO
}