#pragma once

#include "Particle.h"
#include "Forces/SpringForce.h"
#include "Integration/Leapfrog.h"

__global__ calculateKernel(int N, Particle* particles){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    /*if(idx < numParticles) {
        SpringForce::interact(particle[idx]);
    }*/
    // Do nothing for now //TODO
}

__global__ iterateKernel(int N, Particle* particles){
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