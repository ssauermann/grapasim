#include "LinkedCellsImpl.h"

#include "Integration/Leapfrog.h"

__global__ void calculateKernel(int N, Particle* particles){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    /*if(idx < numParticles) {
        SpringForce::interact(particle[idx]);
    }*/
    // Do nothing for now //TODO
}

__global__ void iterateKernel(int N, Particle* particles){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < N) {
        SpringForce::calculate(particle[idx]);
    }
}


__global__ preKernel(int N, Particle* particles){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < N) {
        Leapfrog::doStepPreForce(particle[idx]);
    }
}


__global__ postKernel(Particle* particles){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < N) {
        Leapfrog::doStepPostForce(particle[idx]);
    }
}

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