"""Check 4.3.2 perturbation coefficients in alternating native cosmology scopes."""

import ctypes

import numpy as np
import py21cmfast as p21
from py21cmfast.drivers._global_initialization import c_state

from growth import perturb_coefficients


def main():
    native = ctypes.CDLL(p21.c_21cmfast.__file__)
    for function in (native.ddickedt, native.dicke):
        function.argtypes = [ctypes.c_double]
        function.restype = ctypes.c_double
    baseline = p21.InputParameters(random_seed=424242)
    cases = (
        baseline.evolve_input_structs(OMm=0.308, hlittle=0.678),
        baseline,
        baseline.evolve_input_structs(
            OMm=0.29, hlittle=0.7, INITIAL_REDSHIFT=277.123456
        ),
    )
    for inputs in cases:
        for redshift in (6.123456789, 9.375, 18.5, 34.999999):
            z = float(np.float32(redshift))
            with c_state(inputs, broadcast_inputs=True):
                expected = native.ddickedt(z) / native.dicke(z)
                growth = native.dicke(z)
                native_start = np.float32(inputs.simulation_options.INITIAL_REDSHIFT)
                initial = native.dicke(float(native_start))
            with c_state(baseline, broadcast_inputs=True):
                before = native.dicke(z)
                actual = perturb_coefficients(inputs, redshift)
                assert native.dicke(z) == before
            np.testing.assert_array_equal(actual, [growth, initial, expected])
    print("12 matched native triples (36 growth coefficients)")


if __name__ == "__main__":
    main()
