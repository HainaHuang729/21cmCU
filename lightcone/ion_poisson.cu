// Reproduce the eight static OpenMP GSL streams used by the ionization solver.
__device__ unsigned int mt_word(unsigned int *state, unsigned int &index) {
    if (index == 624) {
        for (int i = 0; i < 624; ++i) {
            const unsigned int pair = (state[i] & 0x80000000U) |
                                      (state[(i+1)%624] & 0x7fffffffU);
            state[i] = state[(i+397)%624] ^ (pair >> 1) ^
                       ((pair & 1U) ? 0x9908b0dfU : 0U);
        }
        index = 0;
    }
    unsigned int value = state[index++];
    value ^= value >> 11;
    value ^= (value << 7) & 0x9d2c5680U;
    value ^= (value << 15) & 0xefc60000U;
    return value ^ (value >> 18);
}

extern "C" __global__ void ion_poisson(int *draws, unsigned int *workspace,
                                       long n, int streams, double limit) {
    const int lane = blockIdx.x*blockDim.x + threadIdx.x;
    if (lane >= streams) return;
    unsigned int *state = workspace + 624*lane;
    state[0] = lane == 0 ? 4357U : (unsigned int)lane;
    for (unsigned int i=1; i<624; ++i) {
        const unsigned int last = state[i-1];
        state[i] = 1812433253U*(last ^ (last >> 30)) + i;
    }
    unsigned int index = 624;
    const long block = n/streams, remainder = n%streams;
    const long first = lane*block + (lane < remainder ? lane : remainder);
    const long last = first + block + (lane < remainder);
    for (long i=first; i<last; ++i) {
        double prod = 1.;
        int count = 0;
        do {
            prod *= (double)mt_word(state,index) * (1.0/4294967296.0);
            ++count;
        } while (prod > limit);
        draws[i] = count-1;
    }
}
