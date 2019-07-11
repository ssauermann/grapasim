__global__ void calculate(Particle *particle, int numParticles) {
    int idx = blockIdx.x;
    // Apply gravity
    if(idx < numParticles) {
        particle[idx].F.y -= 9.80665 * particle[idx].mass;
    }
}

__global__ void interact(Particle *particleL1, Particle *particleL2, int numParticles1, int numParticles2) {

    int idx = blockIdx.x;
    int idy = blockIdx.y;

    auto particle1 = particleL1[idx];
    auto particle2 = particleL2[idy];
    if (particle1 == particle2) {
        return;
    }
    if(idx >= numParticles1 || idy >= numParticles2){
        return;
    }


    auto l2norm = (particle2.x - particle1.x).l2norm();

    assert(l2norm > 0);

    // penetration depth
    auto xi = particle1.radius + particle2.radius - l2norm;

    if (xi <= 0) {
        return;
    }

    // Normal vector
    auto N = (particle2.x - particle1.x) / l2norm;
    // Relative velocity
    auto V = particle1.v - particle2.v;

    // Relative velocity in normal direction
    auto xiDot = V * N;

    auto fn = -k * xi - gamma * xiDot;
    auto Fn = fn * N;

    particle1.F += Fn;
}