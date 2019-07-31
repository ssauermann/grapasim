#include <vector>
#include <iostream>
#include <Output/VTKWriter.h>
#include <Generators/SphereGenerator.h>
#include <Generators/MaxwellBoltzmann.h>
#include "Simulation.h"
#include "Particles/LinkedCellsImpl.h"
#include <chrono>

using Clock=std::chrono::high_resolution_clock;
void Simulation::run() {

    unsigned int simSteps = 200;
    unsigned int writeFrequency = 100;
    unsigned int timeFrequency = 10;
    unsigned int decompFrequency = 100;

    bool includeHaloInOutput = true;

    Domain domain = {.x = std::make_pair(0, 0.256), .y = std::make_pair(0, 0.256), .z = std::make_pair(0, 0.256)};
    Vector cellSizeTarget = {0.001, 0.001, 0.001};
    std::vector<Particle> particles;

    SphereGenerator generator(100);
    generator.mass = 0.001;
    generator.mesh = 0.0011;
    generator.dimensions = 3;
    generator.size = 0.0005; //0.0005;
    generator.center.x = 0.128;
    generator.center.y = 0.128;
    generator.center.z = 0.128;

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
    particleContainer.updateDecomp();
    std::cout << "Init Update fin\n";
    particleContainer.prepareComputation();
    std::cout << "Init Prep fin\n";
    particleContainer.iterate();
    std::cout << "Init Iter fin\n";
    particleContainer.iteratePairs();
    std::cout << "Init IterPair fin\n";
    particleContainer.finalizeComputation();
    std::cout << "Init Fin fin\n";

#ifdef VTK
    std::cout << "Writing initial output\n";
    output.writeBegin(0, particleContainer.particleCount(true));
    particleContainer.output(outputW, includeHaloInOutput);
    output.writeFinalize();
#endif


    auto t1 = Clock::now();
    for (unsigned int step = 1; step <= simSteps; ++step) {
        std::cout << "Step: " << step << "\n";

        particleContainer.updateContainer();
        if(step % decompFrequency == 0){
            particleContainer.updateDecomp();
        }
        particleContainer.prepareComputation();

        particleContainer.preStep();

        particleContainer.iterate();
        particleContainer.iteratePairs();

        particleContainer.postStep();
        particleContainer.finalizeComputation();

        if(step % timeFrequency == 0 || step == simSteps){
            auto t2 = Clock::now();
            std::cout << "Time for " << step << " steps: "
                      << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
                      << " nanoseconds" << std::endl;
        }

#ifdef VTK
        if (step % writeFrequency == 0 || step == simSteps) {
            std::cout << "Writing output...\n";
            output.writeBegin(step, particleContainer.particleCount(true));
            particleContainer.output(outputW, includeHaloInOutput);
            output.writeFinalize();
        }
#endif
    }

    std::cout << "Finished simulation" << std::endl;


}
