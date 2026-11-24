// Native nion_fraction_mcg in hmf.c applies both cutoffs in log space;
// unlike ACGs its stellar and escape power laws pivot at 10^7 solar masses.
__device__ double molecular_weight(double x, double mass, double stellar, double escape,
                                   double turnover, double atomic_mass) {
    return exp(stellar + escape - mass / atomic_mass - turnover / mass + x);
}
