#include <iostream>
#include "Simulation.h"

int main() {
    std::cout << "Hello from GraPaSim" << std::endl;

    auto sim = Simulation();

    sim.run();

    return 0;
}