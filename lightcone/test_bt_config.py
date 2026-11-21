"""Unit tests for BT / IC2 parameter parsing and KP_h_Mpc -> KP conversion."""
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bt_config import resolve_bt_cosmo, split_bt_10d_params  # noqa: E402


class BTConfigTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parent.parent
        self.base_cfg = json.loads((root / "point_config.json").read_text())

    def test_point_config_kp_conversion_matches_bt_theta(self):
        theta_kp_h_mpc = float(self.base_cfg["theta"][8])
        cosmo, eff_kph = resolve_bt_cosmo(self.base_cfg["cosmo"], kp_h_mpc=theta_kp_h_mpc)
        self.assertAlmostEqual(cosmo["KP"], float(self.base_cfg["cosmo"]["KP"]), places=12)
        self.assertTrue(math.isclose(eff_kph, theta_kp_h_mpc, rel_tol=1e-12))

    def test_kp_h_mpc_scaling_with_default_and_shifted_hlittle(self):
        cosmo_def, eff_def = resolve_bt_cosmo(self.base_cfg["cosmo"], kp_h_mpc=10.0, ms=2.5)
        self.assertAlmostEqual(cosmo_def["KP"], 6.78, places=12)
        self.assertEqual(cosmo_def["MS"], 2.5)
        self.assertAlmostEqual(eff_def, 10.0, places=12)

        cosmo_shift, eff_shift = resolve_bt_cosmo(
            self.base_cfg["cosmo"],
            overrides={"hlittle": 0.665, "KP_h_Mpc": 2.0, "MS": 2.1},
        )
        self.assertAlmostEqual(cosmo_shift["KP"], 1.33, places=12)
        self.assertEqual(cosmo_shift["MS"], 2.1)
        self.assertAlmostEqual(eff_shift, 2.0, places=12)
        self.assertNotIn("KP_h_Mpc", cosmo_shift)

    def test_conflicting_kp_and_kp_h_mpc_raises(self):
        with self.assertRaises(ValueError):
            resolve_bt_cosmo(
                self.base_cfg["cosmo"],
                overrides={"KP": 5.0, "KP_h_Mpc": 10.0},
            )
        with self.assertRaises(ValueError):
            resolve_bt_cosmo(
                self.base_cfg["cosmo"],
                overrides={"KP": 5.0},
                kp_h_mpc=10.0,
            )

    def test_flat_10d_bt_split_enforces_power_spectrum_6(self):
        flat = {
            "F_STAR10": -1.3,
            "ALPHA_STAR": 0.5,
            "F_ESC10": -1.0,
            "ALPHA_ESC": -0.5,
            "M_TURN": 9.0,
            "t_STAR": 0.5,
            "L_X": 40.5,
            "NU_X_THRESH": 500.0,
            "KP_h_Mpc": 10.0,
            "MS": 2.5,
            "seed": 725213656658,
        }
        base = json.loads(json.dumps(self.base_cfg))
        base["user"]["POWER_SPECTRUM"] = 0
        out = split_bt_10d_params(flat, base)
        self.assertEqual(out["user"]["POWER_SPECTRUM"], 6)
        self.assertAlmostEqual(out["cosmo"]["KP"], 6.78, places=12)
        self.assertEqual(out["cosmo"]["MS"], 2.5)
        self.assertEqual(out["seed"], 725213656658)
        self.assertNotIn("KP_h_Mpc", out["cosmo"])
        self.assertAlmostEqual(out["bt_meta"]["KP_h_Mpc"], 10.0, places=12)
        self.assertEqual(out["astro"]["L_X"], 40.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
