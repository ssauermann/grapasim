#pragma once

#include <cmath>
#include "Constants.h"

struct Vector {
    PRECISION x;
    PRECISION y;
#if DIMENSIONS > 2
    PRECISION z;
#endif

    PRECISION l2norm() {
        return sqrt((*this)*(*this));
    }

    friend PRECISION operator*(const Vector& lhs, const Vector& rhs)
    {
        PRECISION sum = 0;

        sum += lhs.x * rhs.x;
        sum += lhs.y * rhs.y;

#if DIMENSIONS > 2
        sum += lhs.z * rhs.z;
#endif

        return sum;
    }

    Vector& operator+=(const Vector& rhs){
        x += rhs.x;
        y += rhs.y;
#if DIMENSIONS > 2
        z += rhs.z;
#endif
        return *this;
    }

    Vector& operator-=(const Vector& rhs){
        x -= rhs.x;
        y -= rhs.y;
#if DIMENSIONS > 2
        z -= rhs.z;
#endif
        return *this;
    }

    Vector& operator*=(const PRECISION rhs){
        x *= rhs;
        y *= rhs;
#if DIMENSIONS > 2
        z *= rhs;
#endif
        return *this;
    }

    Vector& operator/=(const PRECISION rhs){
        x /= rhs;
        y /= rhs;
#if DIMENSIONS > 2
        z /= rhs;
#endif
        return *this;
    }

};


inline Vector operator+(Vector lhs, const Vector& rhs)
{
    lhs += rhs;
    return lhs;
}

inline Vector operator-(Vector lhs, const Vector& rhs)
{
    lhs -= rhs;
    return lhs;
}

inline Vector operator*(Vector lhs, const PRECISION rhs)
{
    lhs *= rhs;
    return lhs;
}

inline Vector operator/(Vector lhs, const PRECISION rhs)
{
    lhs /= rhs;
    return lhs;
}

inline Vector operator*(const PRECISION lhs, Vector rhs){
    rhs *= lhs;
    return rhs;
}