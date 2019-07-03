#include <cassert>
#include <cmath>
#include "SphereGenerator.h"

void SphereGenerator::generate(std::vector<Particle> &particles) {
    assert(DIMENSIONS >= this->dimensions);

    // Generate a cube and remove all particles which do not fulfill the sphere equation.

    // Choose corner position of the cube so that the center is at (0,0,0)
    // -1 if dimension is used (and cube has to be moved in this dimension) else 0.
    Vector corner{0};

    corner.x = -(this->radius * this->mesh);
    if(this->dimensions > 1){
        corner.y = corner.x;
    }
#if DIMENSIONS > 2
    if(this->dimensions > 2){
        corner.z = corner.x;
    }
#endif


    // Cuboid with one particle at (0,0,0) and 2*r particles per dimension
    std::array<unsigned int, DIMENSIONS> particleNumbers{};

    for (int i = 0; i < DIMENSIONS; ++i) {
        if (this->dimensions >= i) {
            particleNumbers[i] = this->radius * 2 + 1;
        } else {
            particleNumbers[i] = 1;
        }
    }

    auto indices = std::make_shared<std::array<int, DIMENSIONS>>();
    doGenerate(corner, particleNumbers, particles, indices);
}

void SphereGenerator::doGenerate(const Vector &corner,
                                 const std::array<unsigned int, DIMENSIONS> &particleNumbers,
                                 std::vector<Particle> &particles,
                                 const std::shared_ptr<std::array<int, DIMENSIONS>> &indices,
                                 unsigned int nesting) {

    if (nesting < this->dimensions) {
        for (int i = 0; i < particleNumbers[nesting]; ++i) {
            indices->at(nesting) = i;
            this->doGenerate(corner, particleNumbers, particles, indices, nesting + 1);
        }
    } else {
        auto newId = particles.empty() ? 0 : particles.back().id + 1;

        // Calculate position in grid
        Vector position = corner;

        position.x += indices->at(0) * this->mesh;
        position.y += indices->at(1) * this->mesh;
#if DIMENSIONS > 2
        position.z += indices->at(2) * this->mesh;
#endif

        //L2 Norm
        PRECISION norm = position.l2norm();

        if (norm <= this->radius * this->mesh) {
            // In circle

            // Move position to new center
            position += this->center;


            Particle p(position, this->initialVelocity, this->mass, this->size, newId);

            particles.push_back(p);
        }
    }

}
