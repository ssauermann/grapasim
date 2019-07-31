#include "LinkedCellsImpl.h"

#include "Integration/Leapfrog.h"
#include <cmath>
#include <Domain/Hilbert.h>
#include <Domain/Hilbert3D.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>


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

struct GPULayout {
    Particle *deviceParticles = nullptr;
    Particle *deviceHaloParticles = nullptr;
    int *deviceInner = nullptr;
    int *devicePairOffsets = nullptr;
    Cell *deviceCells = nullptr;
    cudaStream_t stream = 0;
    int size = 0;
    Particle *resultParticles = nullptr;
};

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
                p = &haloParticles[-pi - 1];
            }
            if (qi >= 0) {
                q = &particles[qi];
            } else {
                q = &haloParticles[-qi - 1];
            }
            SpringForce::interact(*p, *q);
            p->modified = true;
        }
    }

}

__global__ void iterateKernel(Cell *cells, Particle *particles, int *inner) {

    int cellIdx = inner[blockIdx.x];
    auto &cell = cells[cellIdx];
    int idx = threadIdx.x;

    if (idx < cell.size) {
        int pi = cell.data[idx];
        SpringForce::calculate(particles[pi]);
    }

}


__global__ void preKernel(Cell *cells, Particle *particles, int *inner) {

    int cellIdx = inner[blockIdx.x];
    auto &cell = cells[cellIdx];
    int idx = threadIdx.x;

    if (idx < cell.size) {
        int pi = cell.data[idx];
        Leapfrog::doStepPreForce(particles[pi]);
    }

}


__global__ void postKernel(Cell *cells, Particle *particles, int *inner) {
    int cellIdx = inner[blockIdx.x];
    auto &cell = cells[cellIdx];
    int idx = threadIdx.x;

    if (idx < cell.size) {
        int pi = cell.data[idx];
        Leapfrog::doStepPostForce(particles[pi]);
    }

}

void LinkedCellsImpl::prepareComputation() {
#pragma omp parallel for
    for (int devId = 0; devId < GPU_N; ++devId) {
        CudaSafeCall(cudaSetDevice(devId));

        int N = this->cells.size();
        CudaSafeCall(cudaMalloc((void **) &this->layout[devId].deviceHaloParticles, sizeof(Particle) * N));

        // Copy cells to device
        CudaSafeCall(cudaMemcpyAsync(this->layout[devId].deviceCells, this->cells.data(), sizeof(Cell) * N,
                                     cudaMemcpyHostToDevice, this->layout[devId].stream));
        N = this->haloParticles.size();
        // Copy halo particles to device
        CudaSafeCall(cudaMemcpyAsync(this->layout[devId].deviceHaloParticles, this->haloParticles.data(),
                                     sizeof(Particle) * N,
                                     cudaMemcpyHostToDevice, this->layout[devId].stream));

    }
}

void LinkedCellsImpl::finalizeComputation() {

#pragma omp parallel for
    for (int devId = 0; devId < GPU_N; ++devId) {
        CudaSafeCall(cudaSetDevice(devId));
        int N = this->particles.size();

        // Copy particles from device to host to allow output to access the data
        CudaSafeCall(cudaMemcpyAsync(
                //this->particles.data(),
                this->layout[devId].resultParticles,
                this->layout[devId].deviceParticles,
                sizeof(Particle) * N,
                cudaMemcpyDeviceToHost, this->layout[devId].stream));

        CudaSafeCall(cudaFree(this->layout[devId].deviceHaloParticles));
    }

    for (int devId = 0; devId < GPU_N; ++devId) {
        CudaSafeCall(cudaSetDevice(devId));
        CudaSafeCall(cudaDeviceSynchronize());
    }


    // reduce result
    for (int devId = 0; devId < GPU_N; ++devId) {
        for (unsigned long i = 0; i < this->particles.size(); ++i) {
            if (this->layout[devId].resultParticles[i].modified) {
                this->particles[i].x = this->layout[devId].resultParticles[i].x;
                this->particles[i].v = this->layout[devId].resultParticles[i].v;
            }
        }
    }
}

void LinkedCellsImpl::iteratePairs() {

#pragma omp parallel for
    for (int devId = 0; devId < GPU_N; ++devId) {
        CudaSafeCall(cudaSetDevice(devId));

        int NInner = this->layout[devId].size;
        int NPairOffsets = this->pairOffsets.size();

        dim3 blocks(NInner, NPairOffsets);
        dim3 threadsPerBlock(MAXCELLPARTICLE, MAXCELLPARTICLE);

        calculateKernel << < blocks, threadsPerBlock, 0, this->layout[devId].stream >> >
        (this->layout[devId].deviceCells, this->layout[devId].deviceParticles, this->layout[devId].deviceHaloParticles,
                this->layout[devId].deviceInner, this->layout[devId].devicePairOffsets);
        CudaCheckError();
    }
}

void LinkedCellsImpl::iterate() {

#pragma omp parallel for
    for (int devId = 0; devId < GPU_N; ++devId) {
        CudaSafeCall(cudaSetDevice(devId));

        int NInner = this->layout[devId].size;
        iterateKernel << < NInner, MAXCELLPARTICLE, 0, this->layout[devId].stream >> >
        (this->layout[devId].deviceCells, this->layout[devId].deviceParticles, this->layout[devId].deviceInner);
        CudaCheckError();
    }

}


void LinkedCellsImpl::preStep() {
#pragma omp parallel for
    for (int devId = 0; devId < GPU_N; ++devId) {
        CudaSafeCall(cudaSetDevice(devId));

        int NInner = this->layout[devId].size;
        preKernel << < NInner, MAXCELLPARTICLE, 0, this->layout[devId].stream >> >
        (this->layout[devId].deviceCells, this->layout[devId].deviceParticles, this->layout[devId].deviceInner);
        CudaCheckError();
    }
}

void LinkedCellsImpl::postStep() {
#pragma omp parallel for
    for (int devId = 0; devId < GPU_N; ++devId) {
        CudaSafeCall(cudaSetDevice(devId));

        int NInner = this->layout[devId].size;
        postKernel << < NInner, MAXCELLPARTICLE, 0, this->layout[devId].stream >> >
                                                    (this->layout[devId].deviceCells, this->layout[devId].deviceParticles, this->layout[devId].deviceInner);
        CudaCheckError();
    }
}

LinkedCellsImpl::LinkedCellsImpl(Domain &domain, Vector cellSizeTarget, std::vector<Particle> &particles) : LinkedCells(
        domain, cellSizeTarget, particles) {

    // Get number of available devices
    CudaSafeCall(cudaGetDeviceCount(&GPU_N));
    //GPU_N -= GPU_N % 2;
    printf("CUDA-capable device count: %i\n", GPU_N);

    this->layout = new GPULayout[GPU_N];

    IntVector numInner = numCells;
    numInner.x -= 2;
    numInner.y -= 2;
    numInner.z -= 2;

    if(numInner.z == 1){
        std::cout << "Init 2D Hilbert\n";
        this->decomp = new Hilbert(inner, numInner);
    } else {
        std::cout << "Init 3D Hilbert\n";
        this->decomp = new Hilbert3D(inner, numInner);
    }


#pragma omp parallel for
    for (int devId = 0; devId < GPU_N; ++devId) {
        std::cout << "Init stream " << devId << "\n";
        CudaSafeCall(cudaSetDevice(devId));
        CudaSafeCall(cudaStreamCreate(&this->layout[devId].stream));

        // Copy particles to device
        std::cout << "Copy particles\n";
        int N = this->particles.size();
        CudaSafeCall(cudaMalloc((void **) &this->layout[devId].deviceParticles, sizeof(Particle) * N));
        CudaSafeCall(
                cudaMemcpy(this->layout[devId].deviceParticles, this->particles.data(), sizeof(Particle) * N,
                           cudaMemcpyHostToDevice));

        CudaSafeCall(cudaMallocHost((void **) &this->layout[devId].resultParticles, sizeof(Particle) * N));

        std::cout << "Copy inner\n";
        N = this->inner.size();
        // Copy inner cell indices to device
        CudaSafeCall(cudaMalloc((void **) &this->layout[devId].deviceInner, sizeof(int) * N));
        CudaSafeCall(cudaMemcpy(this->layout[devId].deviceInner, this->inner.data(), sizeof(int) * N,
                                cudaMemcpyHostToDevice));
        this->layout[devId].size = N;

        std::cout << "Copy offsets\n";
        N = this->pairOffsets.size();
        // Copy pairOffsets to device
        CudaSafeCall(cudaMalloc((void **) &this->layout[devId].devicePairOffsets, sizeof(int) * N));
        CudaSafeCall(
                cudaMemcpy(this->layout[devId].devicePairOffsets, this->pairOffsets.data(), sizeof(int) * N,
                           cudaMemcpyHostToDevice));


        std::cout << "Malloc cells\n";
        CudaSafeCall(cudaMalloc((void **) &this->layout[devId].deviceCells, sizeof(Cell) * this->cells.size()));
    }
}

LinkedCellsImpl::~LinkedCellsImpl() {
    for (int devId = 0; devId < GPU_N; ++devId) {
        CudaSafeCall(cudaSetDevice(devId));
        CudaSafeCall(cudaFree(this->layout[devId].deviceParticles));
        CudaSafeCall(cudaFree(this->layout[devId].resultParticles));
        CudaSafeCall(cudaFree(this->layout[devId].deviceInner));
        CudaSafeCall(cudaFree(this->layout[devId].devicePairOffsets));
        CudaSafeCall(cudaFree(this->layout[devId].deviceCells));
    }
    free(this->layout);
}

void LinkedCellsImpl::updateDecomp() {
    // Assert square boundary
    // assert(this->numCells.x % 2 == 0 && this->numCells.y == this->numCells.x && this->numCells.z == this->numCells.x);



// TODO Set inner cells for each device
#ifdef DYNDD

    auto ordered = this->decomp->ordered();
    int partSize = ceil(1.0 * this->particles.size() / GPU_N);

    int offset = 0;

    for (int devId = 0; devId < GPU_N; ++devId) {
        CudaSafeCall(cudaSetDevice(devId));

        CudaSafeCall(cudaFree(this->layout[devId].deviceInner));

        int upperCellIndex = offset;
        int particleCount = 0;
        while(particleCount < partSize && upperCellIndex < ordered.size()){

            auto cellIdx = ordered.at(upperCellIndex);
            particleCount += cells.at(cellIdx).size;
            upperCellIndex++;
        }
        int N = upperCellIndex-offset;

        // Copy inner cell indices to device
        CudaSafeCall(cudaMalloc((void **) &this->layout[devId].deviceInner, sizeof(int) * N));
        CudaSafeCall(cudaMemcpyAsync(this->layout[devId].deviceInner, &ordered.data()[offset], sizeof(int) * N,
                                     cudaMemcpyHostToDevice, this->layout[devId].stream));
        this->layout[devId].size = N;

        offset = upperCellIndex;
    }

#else
    int partSize = ceil(1.0 * this->inner.size() / GPU_N);
    int offset = 0;
    int remaining = this->inner.size();

    for (int devId = 0; devId < GPU_N; ++devId) {
        int N = partSize < remaining ? partSize : remaining;
        CudaSafeCall(cudaSetDevice(devId));
        // Copy inner cell indices to device
        CudaSafeCall(cudaMalloc((void **) &this->layout[devId].deviceInner, sizeof(int) * N));
        CudaSafeCall(cudaMemcpyAsync(this->layout[devId].deviceInner, &this->inner.data()[offset], sizeof(int) * N,
                                cudaMemcpyHostToDevice, this->layout[devId].stream));
        this->layout[devId].size = N;
        offset += N;
        remaining -= N;
    }
    /*for (int devId = 0; devId < GPU_N; ++devId) {
        cudaStreamSynchronize(this->layout[devId].stream);
    }*/
    assert(remaining == 0);
#endif

}


void LinkedCellsImpl::init() {

}