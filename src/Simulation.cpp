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
#include "Forces/ForceFieldOnly.h"
#include "Output/XYZWriter.h"

void Simulation::run() {

    unsigned int simSteps = 100;
    PRECISION stepSize = 0.0001;
    unsigned int writeFrequency = 1;

    auto integrator = Leapfrog(stepSize);
    auto force = ForceFieldOnly();

    auto generator = SphereGenerator(5);
    generator.mesh = 1.1;
    generator.size = 1;
    generator.center.y = 8;

    std::vector<Particle> particles;

    generator.generate(particles);


    //particles.push_back(Particle({0, 0}, {0, 0}, 100, 5, -1));

    std::string filename = "sim";
    auto output = VTKWriter(particles, filename, 0);

    auto particleContainer = LinkedCells(particles);

    auto preStep = std::bind(&Integrator::doStepPreForce, std::ref(integrator), std::placeholders::_1);
    auto postStep = std::bind(&Integrator::doStepPostForce, std::ref(integrator), std::placeholders::_1);
    auto forces = std::bind(&Forces::calculate, std::ref(force), std::placeholders::_1);
    auto forcePairs = std::bind(&Forces::interact, std::ref(force), std::placeholders::_1,std::placeholders::_2);

    /*auto print = [](Particle &p) {
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
    };*/

    std::cout << "Simulating " << particles.size() << " particles\n";
    //Calculate starting forces
    particleContainer.iterate(forces);
    particleContainer.iteratePairs(forcePairs);


    for (unsigned int step = 0; step < simSteps; ++step) {

        particleContainer.iterate(preStep);

        particleContainer.iterate(forces);
        particleContainer.iteratePairs(forcePairs);

        particleContainer.iterate(postStep);

        // particleContainer.iterate(print);


        // TODO Remove when boundaries are implemented
        for (auto it = particles.begin(); it < particles.end();) {
            if (std::isnan(it->x.x) || std::isnan(it->x.y) ||
                std::isnan(it->F.x) || std::isnan(it->F.y) ||
                std::isnan(it->v.x) || std::isnan(it->v.y) ||
                std::abs(it->x.x) > 10000 || std::abs(it->x.y) > 10000 ||
                std::abs(it->F.x) > 10000 || std::abs(it->F.y) > 10000 ||
                std::abs(it->v.x) > 10000 || std::abs(it->v.y) > 10000) {
                particles.erase(it);
                std::cout << "Deleted particle: " << it->id << std::endl;
            } else {
                ++it;
            }
        }

#ifdef DOREVERSE
        if ( step == simSteps / 2) {
            std::cout << "Inverse now" << std::endl;
            integrator.reverse();
        }
#endif

        if (step % writeFrequency == 0)
            output.write(step);

    }


    output.write(simSteps);

    std::cout << "Finished simulation" << std::endl;


}
