#pragma once

#include <vector>
#include <IntVector.h>
#include <cassert>
#include <cmath>

class SFC {
    std::vector<int> &cells;
    IntVector &numCells;

    IntVector activeIndex;
    std::vector<int> order;

protected:
    int targetDepth = 0;

    void execute() {
        std::cout << activeIndex.x << ", " << activeIndex.y << ", " << activeIndex.z << "\n";
        auto idx = activeIndex.x * numCells.y * numCells.z + activeIndex.y * numCells.z + activeIndex.z;
        order.push_back(cells.at(idx));
    }

    void up() {
        activeIndex.y++;
    }

    void down() {
        activeIndex.y--;
    }

    void left() {
        activeIndex.x--;
    }

    void right() {
        activeIndex.x++;
    }

    void front() {
        activeIndex.z--;
    }

    void back() {
        activeIndex.z++;
    }

    // https://stackoverflow.com/a/108360
    static bool isPowerOfTwo(unsigned int n){
        return n && !(n & (n - 1));
    }

public:
    SFC(std::vector<int> &cells, IntVector &numCells) : cells(cells), numCells(numCells), activeIndex({0, 0, 0}) {
        // target level is integer log2 of numCells.x
        unsigned int index = numCells.x;
        while (index >>= 1u) ++targetDepth;
    }

    std::vector<int> ordered(){
        return order;
    }

};