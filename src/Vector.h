#pragma once

#include <cmath>
#include "Constants.h"

struct Vector {
    PRECISION x;
    PRECISION y;
    PRECISION z;

    DEVICE PRECISION l2norm() {
        return sqrt((*this)*(*this));
    }

    DEVICE friend PRECISION operator*(const Vector& lhs, const Vector& rhs)
    {
        PRECISION sum = 0;

        sum += lhs.x * rhs.x;
        sum += lhs.y * rhs.y;
        sum += lhs.z * rhs.z;

        return sum;
    }

    DEVICE Vector& operator+=(const Vector& rhs){
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }

    DEVICE Vector& operator-=(const Vector& rhs){
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        return *this;
    }

    DEVICE Vector& operator*=(const PRECISION rhs){
        x *= rhs;
        y *= rhs;
        z *= rhs;
        return *this;
    }

    DEVICE Vector& operator/=(const PRECISION rhs){
        x /= rhs;
        y /= rhs;
        z /= rhs;
        return *this;
    }

};


DEVICE inline Vector operator+(Vector lhs, const Vector& rhs)
{
    lhs += rhs;
    return lhs;
}

DEVICE inline Vector operator-(Vector lhs, const Vector& rhs)
{
    lhs -= rhs;
    return lhs;
}

DEVICE inline Vector operator*(Vector lhs, const PRECISION rhs)
{
    lhs *= rhs;
    return lhs;
}

DEVICE inline Vector operator/(Vector lhs, const PRECISION rhs)
{
    lhs /= rhs;
    return lhs;
}

DEVICE inline Vector operator*(const PRECISION lhs, Vector rhs){
    rhs *= lhs;
    return rhs;
}