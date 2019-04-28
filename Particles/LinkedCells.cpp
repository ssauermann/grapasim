#include "LinkedCells.h"

void LinkedCells::iteratePairs(const std::function<void(Particle &, Particle &)> &function) {
    // TODO Implement cell based optimization

    for (int i = 0; i < this->particles.size(); ++i) {
        for (int j = i + 1; j < this->particles.size(); ++j) {
            function(this->particles.at(i), this->particles.at(j));
        }
    }
}

void LinkedCells::iterate(const std::function<void(Particle &)> &function) {
    for (auto &p : this->particles) {
        function(p);
    }
}
