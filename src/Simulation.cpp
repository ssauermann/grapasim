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

    auto generator = SphereGenerator(2);
    generator.mass = 1;
    generator.mesh = 1;
    generator.dimensions = 2;
    generator.size = 0.5;
    generator.center.y = 5;
    generator.center.x = 5;

    Domain domain = {.x = std::make_pair(0, 10), .y = std::make_pair(0, 10), .z = std::make_pair(0, 1)};
    Vector cellSizeTarget = {2, 2, 1};

    std::vector<Particle> particles;

    generator.generate(particles);

    std::string filename = "sim";
    auto output = VTKWriter(filename, 0);

    auto particleContainer = LinkedCells(domain, cellSizeTarget, particles);

    auto preStep = std::bind(&Integrator::doStepPreForce, std::ref(integrator), std::placeholders::_1);
    auto postStep = std::bind(&Integrator::doStepPostForce, std::ref(integrator), std::placeholders::_1);
    auto forces = std::bind(&Forces::calculate, std::ref(force), std::placeholders::_1);
    auto forcePairs = std::bind(&Forces::interact, std::ref(force), std::placeholders::_1, std::placeholders::_2);

    auto outputW = std::bind(&OutputWriter::plotParticle, std::ref(output), std::placeholders::_1);

    std::cout << "Simulating " << particles.size() << " particles\n";
    //Calculate starting forces
    particleContainer.iterate(forces);
    particleContainer.iteratePairs(forcePairs);

    output.writeBegin(0, particleContainer.particleCount(true));
    particleContainer.iterateAll(outputW);
    output.writeFinalize();


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

        if (step % writeFrequency == 0 || step == simSteps) {
            output.writeBegin(step, particleContainer.particleCount(true));
            particleContainer.iterateAll(outputW);
            output.writeFinalize();
        }
    }

    std::cout << "Finished simulation" << std::endl;


}
