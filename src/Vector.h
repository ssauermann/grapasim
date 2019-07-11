#pragma once

#include "Constants.h"

struct Vector {
    PRECISION x;
    PRECISION y;
    PRECISION z;

    DEVICE_HOST PRECISION l2norm();

    DEVICE_HOST friend PRECISION operator*(const Vector& lhs, const Vector& rhs);

    DEVICE_HOST Vector& operator+=(const Vector& rhs);

    DEVICE_HOST Vector& operator-=(const Vector& rhs);

    DEVICE_HOST Vector& operator*=(PRECISION rhs);

    DEVICE_HOST Vector& operator/=(PRECISION rhs);

};


DEVICE_HOST Vector operator+(Vector lhs, const Vector& rhs);

DEVICE_HOST Vector operator-(Vector lhs, const Vector& rhs);

DEVICE_HOST Vector operator*(Vector lhs, PRECISION rhs);

DEVICE_HOST Vector operator/(Vector lhs, PRECISION rhs);

DEVICE_HOST Vector operator*(PRECISION lhs, Vector rhs);