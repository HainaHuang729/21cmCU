"""One-line quick_lightcone + SKA1-Low mock observation + interactive 3D WebGL HTML export."""

from __future__ import annotations

from pathlib import Path

import py21cmcu


def main() -> None:
    """Generate a lightcone, apply SKA1-Low mock observation, and export 3D interactive HTML."""
    out_dir = Path("out_quick_viz")
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "lightcone_3d.html"

    lc = py21cmcu.quick_lightcone(
        z_min=6.5,
        z_max=10.5,
        box_len=192.0,
        dim=96,
        hii_dim=48,
        seed=42,
        instrument="SKA1-Low",
        html_3d_path=html_path,
    )
    print(
        f"Lightcone shape: {lc.shape} | z=[{lc.lightcone_redshifts.min():.2f}, "
        f"{lc.lightcone_redshifts.max():.2f}] | HTML viewer: {html_path}"
    )

    if py21cmcu.cuda_available():
        py21cmcu.cleanup_cuda()


if __name__ == "__main__":
    main()
