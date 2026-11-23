"""Check v4 coeval brightness and optical depth on matched upstream inputs."""

import json
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import brightness
import lightcone_pair
from py21cmfast.drivers import single_field
from tests import produce_integration_test_data as official

out = Path(sys.argv[1])
out.mkdir()
cpu = single_field.brightness_temperature
gpu = brightness.brightness_temperature_gpu
settings = official.get_all_options_struct
rows = []
for spin in (True, False):

    def config(*a, **kw):
        kw.update(USE_TS_FLUCT=spin)
        return settings(*a, **kw)

    def compare(**kwargs):
        a = cpu(**kwargs, write=False, regenerate=True)
        b = gpu(**kwargs)
        row = {"z": float(kwargs["ionized_box"].redshift), "spin": spin}
        for k in b.arrays:
            if not b.arrays[k].state.computed_in_mem:
                continue
            x, y = (obj.get(k).astype(float) for obj in (a, b))
            assert np.isfinite(x).all() and np.isfinite(y).all()
            diff = abs(x - y)
            row[k] = {
                "max_abs": float(diff.max()),
                "failed": int((diff > 1.0e-8 + 1.0e-4 * abs(x)).sum()),
            }
        if any(v["failed"] for v in row.values() if isinstance(v, dict)):
            raise AssertionError(row)
        rows.append(row)
        (out / "brightness-check.json").write_text(json.dumps(rows))
        return b

    run_dir = out / str(spin)
    run_dir.mkdir()
    if not spin:
        # The official integration lightconer requests spin fields. For the
        # no-spin branch only compare the emitted brightness cube.
        fields = ("brightness_temp", "neutral_fraction")
    else:
        fields = lightcone_pair.FIELDS
    with patch.object(official, "get_all_options_struct", config), patch.object(
        brightness, "brightness_temperature_gpu", compare
    ), patch.object(lightcone_pair, "FIELDS", fields):
        lightcone_pair.run(run_dir, 192, "gpu")
