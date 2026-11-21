"""Parameter and unit resolver for the BT / IC2 broken-power-law 21cmFAST branch."""
from copy import deepcopy

BT_ASTRO_KEYS = (
    "F_STAR10",
    "ALPHA_STAR",
    "F_ESC10",
    "ALPHA_ESC",
    "M_TURN",
    "t_STAR",
    "L_X",
    "NU_X_THRESH",
)

BT_COSMO_KEYS = (
    "OMm",
    "OMb",
    "hlittle",
    "SIGMA_8",
    "POWER_INDEX",
    "KP",
    "MS",
)

BT_DEFAULT_COSMO = {
    "OMm": 0.308,
    "OMb": 0.0484,
    "hlittle": 0.678,
    "SIGMA_8": 0.815,
    "POWER_INDEX": 0.968,
}


def resolve_bt_cosmo(base_cosmo, overrides=None, *, kp_h_mpc=None, ms=None):
    """Resolve cosmology parameters, converting BT ``KP_h_Mpc`` [h/Mpc] to native ``KP`` [1/Mpc]."""
    resolved = dict(BT_DEFAULT_COSMO)
    if base_cosmo:
        for k, v in base_cosmo.items():
            if k != "KP_h_Mpc":
                resolved[k] = float(v)

    ov = dict(overrides) if overrides else {}
    has_ov_kp = "KP" in ov and ov["KP"] is not None
    ov_kp = float(ov.pop("KP")) if has_ov_kp else None
    ov_kph = ov.pop("KP_h_Mpc", None)

    for k, v in ov.items():
        resolved[k] = float(v)

    hlittle = float(resolved["hlittle"])
    if hlittle <= 0.0:
        raise ValueError(f"hlittle must be positive, got {hlittle}")

    eff_kph = kp_h_mpc if kp_h_mpc is not None else ov_kph
    if eff_kph is not None:
        eff_kph = float(eff_kph)
        if kp_h_mpc is not None and ov_kph is not None:
            ov_kph_f = float(ov_kph)
            if abs(eff_kph - ov_kph_f) > 1e-6 * max(1.0, abs(eff_kph)):
                raise ValueError(
                    f"Conflicting KP_h_Mpc values: override={ov_kph_f}, explicit={eff_kph}"
                )
        converted_kp = eff_kph * hlittle
        if has_ov_kp and abs(ov_kp - converted_kp) > 1e-6 * max(1.0, abs(ov_kp)):
            raise ValueError(
                f"Conflicting KP ({ov_kp} 1/Mpc) and KP_h_Mpc ({eff_kph} h/Mpc -> {converted_kp} 1/Mpc)"
            )
        resolved["KP"] = converted_kp
    elif has_ov_kp:
        resolved["KP"] = ov_kp

    if ms is not None:
        resolved["MS"] = float(ms)

    if "KP" not in resolved or float(resolved["KP"]) <= 0.0:
        raise ValueError(f"KP must be positive, got {resolved.get('KP')}")
    if "MS" not in resolved:
        raise ValueError("MS must be specified for BT POWER_SPECTRUM=6")

    resolved["KP"] = float(resolved["KP"])
    resolved["MS"] = float(resolved["MS"])
    kp_h_mpc_effective = resolved["KP"] / hlittle
    return resolved, kp_h_mpc_effective


def split_bt_10d_params(params_dict, base_config):
    """Split a flat 10D or nested BT parameter dict into validated py21cmfast config sections."""
    cfg = deepcopy(base_config)
    cfg.setdefault("user", {})
    cfg.setdefault("astro", {})
    cfg.setdefault("cosmo", {})
    cfg["user"]["POWER_SPECTRUM"] = 6

    p = dict(params_dict)
    cosmo_ov = {}
    if isinstance(p.get("cosmo"), dict):
        cosmo_ov.update(p["cosmo"])
    if isinstance(p.get("astro"), dict):
        cfg["astro"].update({k: float(v) for k, v in p["astro"].items()})
    if isinstance(p.get("user"), dict):
        cfg["user"].update(p["user"])
        cfg["user"]["POWER_SPECTRUM"] = 6

    for k in BT_ASTRO_KEYS:
        if k in p and p[k] is not None:
            cfg["astro"][k] = float(p[k])

    for k in BT_COSMO_KEYS:
        if k in p and p[k] is not None:
            cosmo_ov[k] = float(p[k])

    if "KP_h_Mpc" in p and p["KP_h_Mpc"] is not None:
        cosmo_ov["KP_h_Mpc"] = float(p["KP_h_Mpc"])

    if "seed" in p and p["seed"] is not None:
        cfg["seed"] = int(p["seed"])
    if "index" in p and p["index"] is not None:
        cfg["index"] = int(p["index"])

    cosmo, kp_h_mpc_eff = resolve_bt_cosmo(cfg["cosmo"], cosmo_ov)
    cfg["cosmo"] = cosmo
    cfg["bt_meta"] = {
        "KP_h_Mpc": kp_h_mpc_eff,
        "KP_Mpc_inv": cosmo["KP"],
        "MS": cosmo["MS"],
        "POWER_SPECTRUM": 6,
    }
    return cfg
