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
    PRECISION stepSize = 0.01;
    unsigned int writeFrequency = 1;

    auto integrator = Leapfrog(stepSize);
    auto force = ForceFieldOnly();

    auto generator = SphereGenerator(5);
    generator.mass = 1;
    generator.mesh = 0.45;
    generator.size = 1;
    generator.center.y = 8;

    Domain domain = {.x = std::make_pair(0, 50), .y = std::make_pair(0, 50), .z = std::make_pair(0, 50)};
    Vector cellSizeTarget = {5, 5, 5};

    std::vector<Particle> particles;

    generator.generate(particles);


    //particles.push_back(Particle({0, 0}, {0, 0}, 100, 5, -1));

    std::string filename = "sim";
    auto output = VTKWriter(particles, filename, 0);

    auto particleContainer = LinkedCells(domain, cellSizeTarget, particles);

    auto preStep = std::bind(&Integrator::doStepPreForce, std::ref(integrator), std::placeholders::_1);
    auto postStep = std::bind(&Integrator::doStepPostForce, std::ref(integrator), std::placeholders::_1);
    auto forces = std::bind(&Forces::calculate, std::ref(force), std::placeholders::_1);
    auto forcePairs = std::bind(&Forces::interact, std::ref(force), std::placeholders::_1, std::placeholders::_2);

    std::cout << "Simulating " << particles.size() << " particles\n";
    //Calculate starting forces
    particleContainer.iterate(forces);
    particleContainer.iteratePairs(forcePairs);
    output.write(0);


    for (unsigned int step = 1; step <= simSteps; ++step) {

        particleContainer.updateContainer();

        particleContainer.iterate(preStep);

        particleContainer.iterate(forces);
        particleContainer.iteratePairs(forcePairs);

        particleContainer.iterate(postStep);

        // particleContainer.iterate(print);



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
