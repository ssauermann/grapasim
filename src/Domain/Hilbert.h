#pragma once

#include "SFC.h"

class Hilbert : public SFC {

    void H(int depth) {
        if (depth == 0) {
            execute();
        } else {
            A(depth - 1);
            up();
            H(depth - 1);
            right();
            H(depth - 1);
            down();
            B(depth - 1);
        }
    }

    void A(int depth) {
        if (depth == 0) {
            execute();
        } else {

            H(depth - 1);
            right();
            A(depth - 1);
            up();
            A(depth - 1);
            left();
            C(depth - 1);
        }
    }

    void B(int depth) {
        if (depth == 0) {
            execute();
        } else {
            C(depth - 1);
            left();
            B(depth - 1);
            down();
            B(depth - 1);
            right();
            H(depth - 1);
        }
    }

    void C(int depth) {
        if (depth == 0) {
            execute();
        } else {
            B(depth - 1);
            down();
            C(depth - 1);
            left();
            C(depth - 1);
            up();
            A(depth - 1);
        }
    }

public:
    Hilbert(std::vector<int> &cells, IntVector &numCells) : SFC(cells, numCells) {
        assert(isPowerOfTwo(numCells.x) && numCells.y == numCells.x && numCells.z == 1);

        H(targetDepth);
    }

};