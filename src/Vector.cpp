#include "Vector.h"
#include <cmath>

PRECISION Vector::l2norm() {
    return sqrt((*this)*(*this));
}

PRECISION operator*(const Vector &lhs, const Vector &rhs) {
    PRECISION sum = 0;

    sum += lhs.x * rhs.x;
    sum += lhs.y * rhs.y;
    sum += lhs.z * rhs.z;

    return sum;
}

Vector &Vector::operator+=(const Vector &rhs) {
    x += rhs.x;
    y += rhs.y;
    z += rhs.z;
    return *this;
}

Vector &Vector::operator-=(const Vector &rhs) {
    x -= rhs.x;
    y -= rhs.y;
    z -= rhs.z;
    return *this;
}

Vector &Vector::operator*=(const PRECISION rhs) {
    x *= rhs;
    y *= rhs;
    z *= rhs;
    return *this;
}

Vector &Vector::operator/=(const PRECISION rhs) {
    x /= rhs;
    y /= rhs;
    z /= rhs;
    return *this;
}

Vector operator+(Vector lhs, const Vector &rhs) {
    lhs += rhs;
    return lhs;
}

Vector operator-(Vector lhs, const Vector &rhs) {
    lhs -= rhs;
    return lhs;
}

Vector operator*(Vector lhs, const PRECISION rhs) {
    lhs *= rhs;
    return lhs;
}

Vector operator/(Vector lhs, const PRECISION rhs) {
    lhs /= rhs;
    return lhs;
}

Vector operator*(const PRECISION lhs, Vector rhs) {
    rhs *= lhs;
    return rhs;
}
