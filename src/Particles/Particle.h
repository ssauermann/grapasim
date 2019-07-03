#pragma once

#include <tuple>
#include "../Constants.h"

struct Particle {

    Particle(VECTOR x, VECTOR v, PRECISION mass, PRECISION radius, unsigned long id) :
            x(x), v(v), mass(mass), radius(radius), id(id) {
        this->F.fill(0);
    }

    VECTOR x;

    VECTOR F;

    VECTOR v;

    PRECISION mass;

    PRECISION radius;

    unsigned long id;

};