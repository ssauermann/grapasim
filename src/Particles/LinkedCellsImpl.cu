#include "LinkedCellsImpl.h"

#include "Integration/Leapfrog.h"
/*
__device__ void calculateKernelInner(int NA, int NB, Particle *cellA, Particle *cellB) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < NA && idy < NB) {
        auto p = cellA[idx];
        auto q = cellB[idy];

        if (p != q) {
            SpringForce::interact(p, q);
        }
    }
}*/

__global__ void
calculateKernel(Cell *cells, Particle *particles, Particle *haloParticles, int *inner, int *pairOffsets) {

    int cellIdx = inner[blockIdx.x];
    auto &cell = cells[cellIdx];
    int offset = pairOffsets[blockIdx.y];
    auto &otherCell = cells[cellIdx + offset];

    int idx = threadIdx.x;
    int idy = threadIdx.y;
    if (idx < cell.size && idy < otherCell.size) {
        int pi = cell.data[idx];
        int qi = otherCell.data[idy];
        if (pi != qi) {
            Particle *p;
            Particle *q;
            if (pi >= 0) {
                p = &particles[pi];
            } else {
                p = &haloParticles[-pi-1];
            }
            if (qi >= 0) {
                q = &particles[qi];
            } else {
                q = &haloParticles[-qi-1];
            }
            SpringForce::interact(*p, *q);
        }
    }

}

__global__ void iterateKernel(int N, Particle *particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        SpringForce::calculate(particles[idx]);
    }
}


__global__ void preKernel(int N, Particle *particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        Leapfrog::doStepPreForce(particles[idx]);
    }
}


__global__ void postKernel(int N, Particle *particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        Leapfrog::doStepPostForce(particles[idx]);
    }
}

void LinkedCellsImpl::prepareComputation() {

    int N = this->cells.size();
    // Copy cells to device
    CudaSafeCall(cudaMemcpy(this->deviceCells, this->cells.data(), sizeof(Cell) * N, cudaMemcpyHostToDevice));

    N = this->haloParticles.size();
    // Copy halo particles to device
    CudaSafeCall(cudaMalloc((void **) &this->deviceHaloParticles, sizeof(Particle) * N));
    CudaSafeCall(cudaMemcpy(this->deviceHaloParticles, this->haloParticles.data(), sizeof(Particle) * N,
                            cudaMemcpyHostToDevice));


}

void LinkedCellsImpl::finalizeComputation() {
    int N = this->particles.size();
// Copy particles from device to host to allow output to access the data
    CudaSafeCall(cudaMemcpy(
            this->particles.data(),
            this->deviceParticles,
            sizeof(Particle) * N,
            cudaMemcpyDeviceToHost));

    // Clear halos again
    CudaSafeCall(cudaFree(this->deviceHaloParticles));
}

void LinkedCellsImpl::iteratePairs() {

    int NInner = this->inner.size();
    int NPairOffsets = this->pairOffsets.size();

    dim3 blocks(NInner, NPairOffsets);
    dim3 threadsPerBlock(MAXCELLPARTICLE, MAXCELLPARTICLE);

    calculateKernel << < blocks, threadsPerBlock >> >(this->deviceCells, this->deviceParticles, this->deviceHaloParticles,
            this->deviceInner, this->devicePairOffsets);
    CudaCheckError();
}

void LinkedCellsImpl::iterate() {
    int N = this->particles.size();

    iterateKernel << < (N + 511) / 512, 512 >> > (N, this->deviceParticles);
    CudaCheckError();
}


void LinkedCellsImpl::preStep() {
    int N = this->particles.size();

    preKernel << < (N + 511) / 512, 512 >> > (N, this->deviceParticles);
    CudaCheckError();
}

void LinkedCellsImpl::postStep() {

    int N = this->particles.size();

    postKernel << < (N + 511) / 512, 512 >> > (N, this->deviceParticles);
    CudaCheckError();
}

LinkedCellsImpl::LinkedCellsImpl(Domain &domain, Vector cellSizeTarget, std::vector<Particle> &particles) : LinkedCells(
        domain, cellSizeTarget, particles) {

    int N = this->particles.size();
    // Copy particles to device
    CudaSafeCall(cudaMalloc((void **) &this->deviceParticles, sizeof(Particle) * N));
    CudaSafeCall(
            cudaMemcpy(this->deviceParticles, this->particles.data(), sizeof(Particle) * N, cudaMemcpyHostToDevice));

    N = this->inner.size();
    // Copy inner cell indices to device
    CudaSafeCall(cudaMalloc((void **) &this->deviceInner, sizeof(int) * N));
    CudaSafeCall(cudaMemcpy(this->deviceInner, this->inner.data(), sizeof(int) * N, cudaMemcpyHostToDevice));

    N = this->pairOffsets.size();
    // Copy pairOffsets to device
    CudaSafeCall(cudaMalloc((void **) &this->devicePairOffsets, sizeof(int) * N));
    CudaSafeCall(cudaMemcpy(this->devicePairOffsets, this->pairOffsets.data(), sizeof(int) * N, cudaMemcpyHostToDevice));


    CudaSafeCall(cudaMalloc((void **) &this->deviceCells, sizeof(Cell) * this->cells.size()));

}

LinkedCellsImpl::~LinkedCellsImpl() {
    CudaSafeCall(cudaFree(this->deviceParticles));
    CudaSafeCall(cudaFree(this->deviceInner));
    CudaSafeCall(cudaFree(this->devicePairOffsets));
    CudaSafeCall(cudaFree(this->deviceCells));
}
