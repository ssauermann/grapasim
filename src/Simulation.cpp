#include <vector>
#include <iostream>
#include <cmath>
#include <Output/VTKWriter.h>
#include <Generators/SphereGenerator.h>
#include "Simulation.h"
#include "Integration/Leapfrog.h"
#include "Particles/Particle.h"
#include "Particles/LinkedCells.h"
#include "Forces/Forces.h"
#include "Forces/VelocityFieldOnly.h"
#include "Output/XYZWriter.h"

void Simulation::run() {

    unsigned int simSteps = 200;
    PRECISION stepSize = 0.005;
    auto integrator = Leapfrog(stepSize);
    auto force = VelocityFieldOnly();

    auto generator = SphereGenerator(100);
    generator.mesh = 0.1;
    generator.center[1] = 5;

    std::vector<Particle> particles;

    generator.generate(particles);

    std::string filename = "sim";
    auto output = VTKWriter(particles, filename, 0);
    auto particleContainer = LinkedCells(particles);

    auto preStep = std::bind(&Integrator::doStepPreForce, std::ref(integrator), std::placeholders::_1);
    auto postStep = std::bind(&Integrator::doStepPostForce, std::ref(integrator), std::placeholders::_1);
    auto forces = std::bind(&Forces::calculate, std::ref(force), std::placeholders::_1);

    auto print = [](Particle &p) {
        std::string velocity = "[";
        std::string position = "[";

        for (int i = 0; i < DIMENSIONS; ++i) {
            velocity += std::to_string(p.v[i]);
            position += std::to_string(p.x[i]);
            if (i < DIMENSIONS - 1) {
                velocity += ", ";
                position += ", ";
            }
        }
        velocity += "]";
        position += "]";

        std::cout << "Particle(" << position << " - " << velocity << ")" << std::endl;
    };

    for (unsigned int step = 0; step < simSteps; ++step) {

        particleContainer.iterate(preStep);

        particleContainer.iterate(forces);

        particleContainer.iterate(postStep);

        // particleContainer.iterate(print);


        // TODO Remove when boundaries are implemented
        for (auto it = particles.begin(); it < particles.end();) {
            if (std::isnan(it->x[0]) || std::isnan(it->x[1]) ||
                std::isnan(it->F[0]) || std::isnan(it->F[1]) ||
                std::isnan(it->v[0]) || std::isnan(it->v[1]) ||
                std::abs(it->x[0]) > 10000 || std::abs(it->x[1]) > 10000 ||
                std::abs(it->F[0]) > 10000 || std::abs(it->F[1]) > 10000 ||
                std::abs(it->v[0]) > 10000 || std::abs(it->v[1]) > 10000) {
                particles.erase(it);
                std::cout << "Deleted particle: " << it->id << std::endl;
            } else {
                ++it;
            }
        }


        if (step % 1 == 0)
            output.write(step);

    }


    output.write(simSteps);

    std::cout << "Finished simulation" << std::endl;


}
