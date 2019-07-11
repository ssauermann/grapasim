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
        SpringForce::calculate(particles[idx]);
    }
}


__global__ void preKernel(int N, Particle* particles){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < N) {
        Leapfrog::doStepPreForce(particles[idx]);
    }
}


__global__ void postKernel(int N, Particle* particles){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < N) {
        Leapfrog::doStepPostForce(particles[idx]);
    }
}

void LinkedCellsImpl::prepareComputation() {
    int N = this->particles.size();
// Copy host to device
    CudaSafeCall(  cudaMalloc( (void**)&this->deviceParticles, sizeof(Particle) ) );
    CudaSafeCall(  cudaMemcpy(
            this->deviceParticles,
            this->particles.data(),
            sizeof(Particle) * N,
            cudaMemcpyHostToDevice ) );
}

void LinkedCellsImpl::finalizeComputation() {
    int N = this->particles.size();
    CudaSafeCall( cudaMemcpy(
            this->particles.data(),
            this->deviceParticles,
            sizeof(Particle) * N,
            cudaMemcpyDeviceToHost ) );
    cudaFree(this->deviceParticles);
}

void LinkedCellsImpl::iteratePairs() {
//TODO
}

void LinkedCellsImpl::iterate() {
    int N = this->particles.size();

    iterateKernel<<<(N + 511) / 512, 512>>>(N, this->deviceParticles);

}


void  LinkedCellsImpl::preStep(){
//TODO
}

void  LinkedCellsImpl::postStep(){
//TODO
}