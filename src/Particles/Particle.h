#pragma once

#include <tuple>
#include <Vector.h>
#include "../Constants.h"

struct Particle {

    Particle(Vector x, Vector v, PRECISION mass, PRECISION radius, unsigned long id, int type) :
            x(x), F({0}) , v(v), mass(mass), radius(radius), id(id), type(type){;
    }

    Vector x;

    Vector F;

    Vector v;

    PRECISION mass;

    PRECISION radius;

    int type;

    unsigned long id;

};