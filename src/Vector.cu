#include "Vector.h"

__device__ __host__ PRECISION Vector::l2norm() {
    return sqrt((*this)*(*this));
}

__device__ __host__ PRECISION operator*(const Vector &lhs, const Vector &rhs) {
    PRECISION sum = 0;

    sum += lhs.x * rhs.x;
    sum += lhs.y * rhs.y;
    sum += lhs.z * rhs.z;

    return sum;
}

__device__ __host__ Vector &Vector::operator+=(const Vector &rhs) {
    x += rhs.x;
    y += rhs.y;
    z += rhs.z;
    return *this;
}

__device__ __host__ Vector &Vector::operator-=(const Vector &rhs) {
    x -= rhs.x;
    y -= rhs.y;
    z -= rhs.z;
    return *this;
}

__device__ __host__ Vector &Vector::operator*=(const PRECISION rhs) {
    x *= rhs;
    y *= rhs;
    z *= rhs;
    return *this;
}

__device__ __host__ Vector &Vector::operator/=(const PRECISION rhs) {
    x /= rhs;
    y /= rhs;
    z /= rhs;
    return *this;
}

__device__ __host__ Vector operator+(Vector lhs, const Vector &rhs) {
    lhs += rhs;
    return lhs;
}

__device__ __host__ Vector operator-(Vector lhs, const Vector &rhs) {
    lhs -= rhs;
    return lhs;
}

__device__ __host__ Vector operator*(Vector lhs, const PRECISION rhs) {
    lhs *= rhs;
    return lhs;
}

__device__ __host__ Vector operator/(Vector lhs, const PRECISION rhs) {
    lhs /= rhs;
    return lhs;
}

__device__ __host__ Vector operator*(const PRECISION lhs, Vector rhs) {
    rhs *= lhs;
    return rhs;
}
