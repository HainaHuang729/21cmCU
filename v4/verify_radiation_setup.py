"""Check 4.3.2 radiation lookup tables on one saved atomic-cooling node."""

import argparse
from pathlib import Path

import h5py
import numpy as np
from py21cmfast import LightCone
from py21cmfast.drivers import single_field
from py21cmfast.drivers._global_initialization import c_state
from py21cmfast.wrapper.outputs import EmissivityFields, TsBox

from evolution_backend import evolution_backend


def verify(cpu_cone, reference_fields, output, node):
    inputs = LightCone.from_file(cpu_cone).inputs
    epochs = inputs.node_redshifts
    if inputs.astro_options.USE_MCGS or not 1 <= node < len(epochs):
        raise ValueError("Select an evolved node from an atomic-cooling run")
    current, previous = epochs[node], epochs[node - 1]
    spin = TsBox.new(inputs=inputs, redshift=previous)
    history = []
    with h5py.File(reference_fields) as saved:
        spin.set(
            "xray_ionised_fraction",
            saved[f"compute_spin_temperature/{previous:.12g}/xray_ionised_fraction"][:],
        )
        # The atomic-cooling setup uses the history for its redshift range
        # and the presence of stars. Shell emissivities are used in a later step.
        for z in (epochs[0], current):
            box = EmissivityFields.new(inputs=inputs, redshift=z)
            box.set(
                "sfrd_acg", saved[f"compute_emissivity_fields/{z:.12g}/sfrd_acg"][:]
            )
            history.append(box)
    if not np.any(history[-1].get("sfrd_acg")):
        raise ValueError("Choose a node with nonzero star formation")

    def tables():
        with c_state(inputs):
            return single_field.setup_radiation_fields(
                redshift=current,
                emissivity_fields_list=history,
                previous_spin_temp=spin,
            )

    cpu = tables()
    output.mkdir(exist_ok=True, parents=True)
    with evolution_backend(output / "gpu-stages.json"):
        gpu = tables()
    for name in ("inverse_diff", "m_xHII_low_box", "inverse_val_box"):
        np.testing.assert_array_equal(gpu.get(name), cpu.get(name), err_msg=name)
    for name in (
        "freq_int_heat_tbl",
        "freq_int_ion_tbl",
        "freq_int_lya_tbl",
        "lya_flux_continuum_prefactor_acg",
        "lya_flux_injected_prefactor_acg",
    ):
        ref = cpu.get(name)
        assert np.all(np.isfinite(ref)) and np.any(ref)
        np.testing.assert_allclose(gpu.get(name), ref, atol=0, rtol=1e-12, err_msg=name)
    print(f"Radiation setup matched at z={current:.8f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu_cone", type=Path)
    parser.add_argument("reference_fields", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--node", type=int, default=1)
    args = parser.parse_args()
    verify(args.cpu_cone, args.reference_fields, args.output, args.node)
