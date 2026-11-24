// One condition consumes a contiguous segment of its native GSL stream.
// These functions preserve the order of the stopping sum and mass correction.
struct MassSampleWords {
    const unsigned *words;
    int count;
    int consumed = 0;

    __device__ bool take(unsigned &word) {
        if (consumed == count) return false;
        word = words[consumed++];
        return true;
    }

    __device__ bool uniform_index(int size, int &index) {
        const unsigned scale = 0xffffffffu / (unsigned)size;
        unsigned word;
        do {
            if (!take(word)) return false;
            index = word / scale;
        } while (index >= size);
        return true;
    }
};

struct MassSampleTable {
    const double *ratios;
    int nx, ny;
    double x_min, y_min, x_width, y_width, min_log_probability;

    __device__ bool mass(double condition, double condition_mass, unsigned word,
                         double &result) const {
        double log_probability =
            word ? reference_log(word * 2.3283064365386963e-10) : min_log_probability;
        log_probability = fmax(min_log_probability, log_probability);
        int x = (int)floor((condition - x_min) / x_width);
        int y = (int)floor((log_probability - y_min) / y_width);
        if (x < 0 || x + 1 >= nx || y < 0 || y + 1 >= ny) return false;

        // The optimized native release subtracts the axis origin before the bin offset.
        double dx = ((condition - x_min) - x_width * x) / x_width;
        double dy = ((log_probability - y_min) - y_width * y) / y_width;
        const double *low = ratios + x * ny + y;
        const double *high = low + ny;
        double left = low[0] * (1. - dy) + low[1] * dy;
        double right = high[0] * (1. - dy) + high[1] * dy;
        result = fmin(1., fmax(0., left * (1. - dx) + right * dx)) * condition_mass;
        return true;
    }
};

__device__ bool correct_catalog_mass(MassSampleWords &stream, double target, double total,
                                     int &size, float *masses) {
    int choice;
    if (!stream.uniform_index(2, choice)) return false;
    if (choice) {
        if (fabs((total - masses[size - 1]) - target) < fabs(total - target)) --size;
        return true;
    }
    int removed;
    double before_removal;
    float mass;
    do {
        do {
            if (!stream.uniform_index(size, removed)) return false;
        } while (masses[removed] == 0.f);
        mass = masses[removed];
        before_removal = total;
        total -= mass;
        masses[removed] = 0.f;
    } while (total > target);
    // The native release keeps the pre-subtraction double in a register.
    if (fabs(before_removal - target) < fabs(total - target)) masses[removed] = mass;
    return true;
}

__device__ bool draw_limited_catalog_mass(const MassSampleTable &table, MassSampleWords &stream,
                                          double condition, double condition_mass, double target,
                                          bool rare, int capacity, float *masses, int &size) {
    if (rare && capacity) {
        masses[0] = (float)target;
        size = 1;
        return true;
    }
    size = 0;
    double total = 0.;
    while (total < target) {
        unsigned word;
        double mass;
        if (size == capacity || !stream.take(word) ||
            !table.mass(condition, condition_mass, word, mass))
            return false;
        total += mass;
        masses[size++] = (float)mass;
    }
    return size && correct_catalog_mass(stream, target, total, size, masses);
}
