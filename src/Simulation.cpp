#include <vector>
#include <iostream>
#include <cmath>
#include <Output/VTKWriter.h>
#include <Generators/SphereGenerator.h>
#include <Generators/MaxwellBoltzmann.h>
#include "Simulation.h"
#include "Integration/Leapfrog.h"
#include "Particles/Particle.h"
#include "Particles/LinkedCellsImpl.h"
#include "Forces/SpringForce.h"
#include "Output/XYZWriter.h"

void Simulation::run() {

    unsigned int simSteps = 100000;
    unsigned int writeFrequency = 1000;

    bool includeHaloInOutput = true;

    Domain domain = {.x = std::make_pair(-0.064, 0.064), .y = std::make_pair(0, 0.256), .z = std::make_pair(-0.5, 0.5)};
    Vector cellSizeTarget = {0.0005, 0.001, 1};
    std::vector<Particle> particles;

    SphereGenerator generator(30);
    generator.mass = 0.001;
    generator.mesh = 0.0011;
    generator.dimensions = 2;
    generator.size = 0.0005; //0.0005;
    generator.center.y = 0.04;

    generator.generate(particles);
    MaxwellBoltzmannDistribution mwb(0.001, 2);
    for (auto &p: particles) {
        // p.v.y = -5;
        mwb.apply(p);
    }

    std::string filename = "sim";
    auto output = VTKWriter(filename, 0);

    LinkedCellsImpl particleContainer(domain, cellSizeTarget, particles);

    auto outputW = std::bind(&OutputWriter::plotParticle, std::ref(output), std::placeholders::_1);

    std::cout << "Simulating " << particles.size() << " particles for " << simSteps << " time-steps\n";
    //Calculate starting forces
    particleContainer.updateContainer();
    particleContainer.prepareComputation();
    particleContainer.iterate();
    particleContainer.iteratePairs();
    particleContainer.finalizeComputation();

    output.writeBegin(0, particleContainer.particleCount(true));
    particleContainer.output(outputW, includeHaloInOutput);
    output.writeFinalize();


    for (unsigned int step = 1; step <= simSteps; ++step) {
        std::cout << "Step: " << step << "\n";

        particleContainer.updateContainer();
        particleContainer.prepareComputation();

        particleContainer.preStep();

        particleContainer.iterate();
        particleContainer.iteratePairs();

        particleContainer.postStep();
        particleContainer.finalizeComputation();


#ifdef DOREVERSE
        if ( step == simSteps / 2) {
            std::cout << "Inverse now" << std::endl;
            integrator.reverse();
        }
#endif

        if (step % writeFrequency == 0 || step == simSteps) {
            output.writeBegin(step, particleContainer.particleCount(true));
            particleContainer.output(outputW, includeHaloInOutput);
            output.writeFinalize();
        }
    }

    std::cout << "Finished simulation" << std::endl;


}
