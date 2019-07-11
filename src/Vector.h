#pragma once

#include "Constants.h"

struct Vector {
    PRECISION x;
    PRECISION y;
    PRECISION z;

     PRECISION l2norm();

     friend PRECISION operator*(const Vector& lhs, const Vector& rhs);

     Vector& operator+=(const Vector& rhs);

     Vector& operator-=(const Vector& rhs);

     Vector& operator*=(PRECISION rhs);

     Vector& operator/=(PRECISION rhs);

};


 Vector operator+(Vector lhs, const Vector& rhs);

 Vector operator-(Vector lhs, const Vector& rhs);

 Vector operator*(Vector lhs, PRECISION rhs);

 Vector operator/(Vector lhs, PRECISION rhs);

 Vector operator*(PRECISION lhs, Vector rhs);