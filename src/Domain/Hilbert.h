#pragma once

#include "SFC.h"

class Hilbert : public SFC {

    void _H(int depth);

    void _A(int depth);

    void _B(int depth);

    void _C(int depth);

public:
    Hilbert(std::vector<int> &cells, IntVector &numCells) : SFC(cells, numCells) {
        assert(isPowerOfTwo(numCells.x) && numCells.y == numCells.x && numCells.z == 1);

        _H(targetDepth);
    }

};