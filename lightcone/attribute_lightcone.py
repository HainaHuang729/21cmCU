"""Attribute matched lightcone residuals by field and neutral-fraction support."""
import h5py
import json
import numpy as np
from pathlib import Path
import sys

root = Path(sys.argv[1])
fields = ("brightness_temp", "xH_box", "x_e_box", "Tk_box", "Ts_box", "dNrec_box")
atol, rtol = 1e-8, 1e-4
totals = {}
extrema = {}
first = {}
by_slice_counts = {f: [] for f in fields}
any_field_by_slice = []
with h5py.File(root / "cpu/lightcone.h5") as cpu, h5py.File(root / sys.argv[2] / "lightcone.h5") as gpu:
    for key in ("point", "seed", "dim", "hii_dim"):
        assert cpu.attrs[key] == gpu.attrs[key]
    if "box_len" in cpu.attrs and "box_len" in gpu.attrs:
        assert float(cpu.attrs["box_len"]) == float(gpu.attrs["box_len"])
    for axis in ("redshift", "distance_Mpc", "node_redshift"):
        np.testing.assert_array_equal(cpu[axis][:], gpu[axis][:])
    zs = cpu["redshift"][:]
    side = int(cpu.attrs["hii_dim"])
    for f in fields:
        assert cpu[f].shape == gpu[f].shape == (side, side, len(zs)), f
    nodes = np.sort(cpu["node_redshift"][:])

    def site(ij, start, reference, test):
        i, j, k = map(int, ij)
        k += start
        right = int(np.searchsorted(nodes, zs[k]))
        return dict(pixel=[i, j], slice=k, z=float(zs[k]),
                    node_bounds=nodes[max(0,right-1):right+1].tolist(),
                    cpu={f:float(reference[f][ij]) for f in fields},
                    gpu={f:float(test[f][ij]) for f in fields})

    for start in range(0, len(zs), 16):
        reference = {f:cpu[f][:,:,start:start+16].astype(np.float64) for f in fields}
        test = {f:gpu[f][:,:,start:start+16].astype(np.float64) for f in fields}
        assert all(np.isfinite(box[f]).all() for box in (reference,test) for f in fields)
        delta = {f:test[f]-reference[f] for f in fields}
        errors = {f:np.abs(delta[f]) > atol + rtol*np.abs(reference[f]) for f in fields}
        any_field_by_slice.extend(np.count_nonzero(np.logical_or.reduce(tuple(errors.values())), axis=(0,1)).tolist())
        hi = "xH_box"
        bt = "brightness_temp"
        rec = "dNrec_box"
        thermal = errors["x_e_box"] | errors["Tk_box"] | errors["Ts_box"]
        zero_c = reference[hi] == 0
        zero_g = test[hi] == 0
        changed_zero = zero_c ^ zero_g
        bright_fail = errors[bt]
        neutral_fail = errors[hi]
        positive = (~zero_c) & (~zero_g)
        # On the interpolated output, this is a diagnostic rescaling,
        # since products and interpolation do not in general commute.
        swapped = test[bt][positive] * (reference[hi][positive] / test[hi][positive])
        rescaled_pass = np.count_nonzero(bright_fail[positive] & (
            np.abs(swapped-reference[bt][positive]) <= atol + rtol*np.abs(reference[bt][positive])))
        counts = {
            **{f:np.count_nonzero(errors[f]) for f in fields},
            "brightness_and_xHI":np.count_nonzero(bright_fail & neutral_fail),
            "brightness_and_thermal":np.count_nonzero(bright_fail & thermal),
            "brightness_xHI_zero_transition":np.count_nonzero(bright_fail & changed_zero),
            "brightness_both_xHI_positive":np.count_nonzero(bright_fail & positive),
            "brightness_pass_after_xHI_rescale":rescaled_pass,
            "xHI_zero_transition":np.count_nonzero(neutral_fail & changed_zero),
            "xHI_jump_ge_0.1":np.count_nonzero(np.abs(delta[hi])>=0.1),
            "xHI_jump_ge_0.5":np.count_nonzero(np.abs(delta[hi])>=0.5),
            "rec_and_xHI":np.count_nonzero(errors[rec] & neutral_fail),
            "rec_and_thermal":np.count_nonzero(errors[rec] & thermal),
            "rec_both_xHI_zero":np.count_nonzero(errors[rec] & zero_c & zero_g),
            "rec_jump_ge_0.01":np.count_nonzero(np.abs(delta[rec])>=0.01),
        }
        for key, count in counts.items():
            totals[key] = totals.get(key, 0) + int(count)
        for f in fields:
            absolute = np.abs(delta[f])
            ij = np.unravel_index(np.argmax(absolute), absolute.shape)
            value = float(absolute[ij])
            if not f in extrema or value > extrema[f]["abs"]:
                extrema[f] = dict(abs=value, **site(ij, start, reference, test))
            by_slice = np.count_nonzero(errors[f], axis=(0,1))
            by_slice_counts[f].extend(by_slice.tolist())
            bad = np.flatnonzero(by_slice)
            if bad.size:
                k = int(bad[-1])
                val = absolute[:,:,k]
                ij = (*np.unravel_index(np.argmax(np.where(errors[f][:,:,k],val,-1)),val.shape), k)
                first[f] = dict(failed=int(by_slice[k]), **site(ij, start, reference, test))
for f in fields:
    assert len(by_slice_counts[f]) == len(zs) and sum(by_slice_counts[f]) == totals[f]
print(json.dumps(dict(counts=totals, first_high_z=first, largest_absolute=extrema, 
                      atol=atol, rtol=rtol, samples_per_field=side*side*len(zs),
                      any_field_out_of_tolerance=sum(any_field_by_slice),
                      redshift_by_slice=zs.tolist(), failed_by_slice=by_slice_counts,
                      any_field_failed_by_slice=any_field_by_slice),
                 indent=2, allow_nan=False))
