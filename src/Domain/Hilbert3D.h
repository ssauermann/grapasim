#pragma once

#include "SFC.h"

class Hilbert3D : public SFC {

    void _xyz(int depth);
    void _xYZ(int depth);
    void _yzx(int depth);
    void _yZX(int depth);
    void _zxy(int depth);
    void _zXY(int depth);
    void _XyZ(int depth);
    void _XYz(int depth);
    void _YzX(int depth);
    void _YZx(int depth);
    void _ZxY(int depth);
    void _ZXy(int depth);


public:
    Hilbert3D(std::vector<int> &cells, IntVector &numCells) : SFC(cells, numCells) {
        assert(isPowerOfTwo(numCells.x) && numCells.y == numCells.x && numCells.z == numCells.x);

        _xyz(targetDepth);
    }

};