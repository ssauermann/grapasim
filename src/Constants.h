#pragma once

#include "HappyClion.h"


#define DIMENSIONS 3

#ifdef __CUDACC__
#ifndef ENABLE_CUDA
#error Set ENABLE_CUDA to enable a cuda build
#endif

#include "CudaError.h"
#define DEVICE __device__
#define DEVICE_HOST __device__ __host__
#else
#define DEVICE
#define DEVICE_HOST
#endif

// #define SHEAR_FORCES // Uncomment to enable shear forces

// #define DOREVERSE

#define DYNDD

typedef float PRECISION;

