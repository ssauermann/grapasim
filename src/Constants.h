#pragma once

#include <array>

#ifdef __JETBRAINS_IDE__
// Stuff that only clion will see goes here
#define ENABLE_VTK
#endif

//TODO ifdef etc.
#define DIMENSIONS 2

typedef float PRECISION;

typedef std::array<PRECISION, DIMENSIONS> VECTOR;
