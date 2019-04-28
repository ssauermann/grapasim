#pragma once

#include "../Constants.h"
#include "OutputWriter.h"

class XYZWriter : public OutputWriter {


public:

    XYZWriter(const std::vector<Particle> &particles, const std::string &filename) : OutputWriter(particles,
                                                                                                 filename) {}

    void write(unsigned long iteration) override;

};

