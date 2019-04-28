#include "XYZWriter.h"
#include "../Particles/Particle.h"

#include <fstream>
#include <sstream>
#include <iomanip>


void XYZWriter::write(unsigned long iteration) {

    std::stringstream ss;
    ss << this->filename << "_";
    ss << std::setw(3) << std::setfill('0') << iteration;
    ss << ".xyz";

    auto filenamePadded = ss.str();

    std::ofstream file(filenamePadded);

    file << this->particles.size() << std::endl;
    file << "Generated by GraPaSim. See http://openbabel.org/wiki/XYZ_(format) for file format documentation."
         << std::endl;

    // Always print numbers with decimal point
    file.setf(std::ios_base::showpoint);

    for (auto &p: this->particles) {
        file << "P ";
        for (int i = 0; i < DIMENSIONS; ++i) {
            file << p.x[i] << " ";
        }

        file << std::endl;
    }
    file.close();
}
