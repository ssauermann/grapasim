#include <vector>
#include <iostream>
#include <cmath>
#include <Output/VTKWriter.h>
#include <Generators/SphereGenerator.h>
#include <Generators/MaxwellBoltzmann.h>
#include "Simulation.h"
#include "Integration/Leapfrog.h"
#include "Particles/Particle.h"
#include "Particles/LinkedCells.h"
#include "Forces/SpringForce.h"
#include "Output/XYZWriter.h"

void Simulation::run() {

    Leapfrog::stepSize = 1.25e-06;

    unsigned int simSteps = 300000;
    unsigned int writeFrequency = 1000;

    bool includeHaloInOutput = true;

    auto force = SpringForce();

    Domain domain = {.x = std::make_pair(-0.1, 0.1), .y = std::make_pair(-0.1, 3), .z = std::make_pair(-1, 1)};
    Vector cellSizeTarget = {0.01, 0.01, 1};
    std::vector<Particle> particles;

    auto generator = SphereGenerator(3);
    generator.mass = 0.1;
    generator.mesh = 0.0101;
    generator.dimensions = 2;
    generator.size = 0.005; //0.0005;

    generator.generate(particles);
    auto mwb = MaxwellBoltzmannDistribution(0.001, 2);
    for (auto &p: particles) {
        // p.v.y = -5;
        mwb.apply(p);
    }

    std::string filename = "sim";
    auto output = VTKWriter(filename, 0);

    auto particleContainer = LinkedCells(domain, cellSizeTarget, particles);

    auto outputW = std::bind(&OutputWriter::plotParticle, std::ref(output), std::placeholders::_1);

    std::cout << "Simulating " << particles.size() << " particles\n";
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
