#include <vector>
#include <iostream>
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
    PRECISION stepSize = 0.01;
    auto integrator = Leapfrog(stepSize);
    auto force = VelocityFieldOnly();

    auto generator = SphereGenerator(20);
    generator.center[1] = 30;

    std::vector<Particle> particles;

    generator.generate(particles);

    std::string filename = "sim";
    auto output = VTKWriter(particles, filename, 0);
    auto particleContainer = LinkedCells(particles);

    auto preStep = std::bind(&Integrator::doStepPreForce, integrator, std::placeholders::_1);
    auto postStep = std::bind(&Integrator::doStepPostForce, integrator, std::placeholders::_1);
    auto forces = std::bind(&Forces::calculate, force, std::placeholders::_1);

    auto print = [](Particle &p) {
        std::string velocity = "[";
        std::string position= "[";

        for (int i = 0; i < DIMENSIONS; ++i) {
            velocity += std::to_string(p.v[i]);
            position += std::to_string(p.x[i]);
            if(i < DIMENSIONS - 1){
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

        output.write(step);

    }

    std::cout << "Finished simulation" << std::endl;


}
