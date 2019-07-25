#include "Hilbert.h"

#define back _back();
#define front _front();
#define up _up();
#define down _down();
#define left _left();
#define right _right();

#define A _A(depth - 1);
#define B _B(depth - 1);
#define C _C(depth - 1);
#define H _H(depth - 1);

void Hilbert::_H(int depth) {
    if (depth == 0) {
        execute();
    } else {
        A
        up
        H
        right
        H
        down
        B
    }
}

void Hilbert::_A(int depth) {
    if (depth == 0) {
        execute();
    } else {

        H
        right
        A
        up
        A
        left
        C
    }
}

void Hilbert::_B(int depth) {
    if (depth == 0) {
        execute();
    } else {
        C
        left
        B
        down
        B
        right
        H
    }
}

void Hilbert::_C(int depth) {
    if (depth == 0) {
        execute();
    } else {
        B
        down
        C
        left
        C
        up
        A
    }
}