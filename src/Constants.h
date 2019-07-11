#pragma once

#define DIMENSIONS 3

#ifdef ENABLE_CUDA
#define DEVICE __device__ __host__
#else
#define DEVICE
#endif

// #define SHEAR_FORCES // Uncomment to enable shear forces

// #define DOREVERSE

typedef float PRECISION;
