#pragma once

#define DIMENSIONS 3

#ifdef ENABLE_CUDA
#define DEVICE __device__
#define DEVICE_HOST __device__ __host__
#else
#define DEVICE
#define DEVICE_HOST
#endif

// #define SHEAR_FORCES // Uncomment to enable shear forces

// #define DOREVERSE

typedef float PRECISION;
