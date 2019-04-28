#pragma once

#include <tuple>
#include "../Constants.h"

struct Particle {

    Particle(VECTOR x, VECTOR v, PRECISION mass, unsigned long id){
        this->x = x;
        this->F.fill(0);
        this->v = v;
        this->mass = mass;
        this->id = id;
    }

    VECTOR x;

    VECTOR F;

    VECTOR v;

    PRECISION mass;

    unsigned long id;

};