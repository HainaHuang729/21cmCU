"""Extract non-FMA codelet bodies from an unpacked FFTW 3.3.10 release."""

import gzip
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
catalog = {}
harmonics = {}
for folder in (
    "dft/scalar/codelets",
    "dft/simd/common",
    "rdft/simd/common",
    "rdft/scalar/r2cf",
    "rdft/scalar/r2cb",
    "rdft/scalar/r2r",
):
    for source in sorted((root / folder).glob("*.c")):
        name = source.stem
        if not re.fullmatch(
            r"(?:n1|n[12][fb]v|t[123][fb]u?v|q1[fb]v|hc2c[fb]dftv|r2c[fb](?:II|III)?|h[fb]2?)_\d+",
            name,
        ):
            continue
        original = source.read_text()
        notice = original[: original.index("*/") + 2]
        text = original.split("#else")[-1]
        start = text.index(f"static void {name}(")
        body = text[start : text.index("\n}", start) + 2].replace(
            "static void", "__device__ void", 1
        )
        desc = text[text.find("static const tw_instr twinstr[]") :]
        if desc:
            numbers = re.findall(
                r"VTW\(0, (\d+)\)|TW_CEXP, 1, (\d+)", desc.split("TW_NEXT")[0]
            )
            if numbers:
                harmonics[name] = [int(a or b) for a, b in numbers]
        catalog[name] = (
            notice
            + "\n// FFTW 3.3.10: "
            + str(source.relative_to(root))
            + " (GPL-2.0-or-later)\n"
            + body
        )
Path(__file__).with_name("codelets.json.gz").write_bytes(
    gzip.compress(json.dumps([catalog, harmonics]).encode(), mtime=0)
)
