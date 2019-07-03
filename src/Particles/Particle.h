#pragma once

#include <tuple>
#include <Vector.h>
#include "../Constants.h"

struct Particle {

    Particle(Vector x, Vector v, PRECISION mass, PRECISION radius, unsigned long id) :
            x(x), F({0}) , v(v), mass(mass), radius(radius), id(id){;
    }

    Vector x;

    Vector F;

    Vector v;

    PRECISION mass;

    PRECISION radius;

    unsigned long id;

};