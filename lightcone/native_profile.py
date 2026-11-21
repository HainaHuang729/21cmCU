"""Measure each top-level native phase without validation callbacks."""
from pathlib import Path
import re

PHASES = (
    ("GenerateICs.c", "ComputeInitialConditions", "ic"),
    ("PerturbField.c", "ComputePerturbField", "perturbation"),
    ("SpinTemperatureBox.c", "ComputeTsBox", "thermal"),
    ("IonisationBox.c", "ComputeIonizedBox", "ionization"),
    ("BrightnessTemperatureBox.c", "ComputeBrightnessTemp", "brightness"),
)


def patch_profile(source_dir):
    for i, (filename, function, _) in enumerate(PHASES):
        path = Path(source_dir)/filename
        text = path.read_text()
        signature = re.compile(r"(?m)^int "+function+r"\s*\(([^{};]+)\)\s*\{")
        signatures = list(signature.finditer(text))
        assert len(signatures)==1, function
        args = signatures[0].group(1)
        names = ", ".join(re.findall(r"\b\w+\s*$", arg)[0].strip() for arg in args.split(","))
        text = signature.sub(lambda m: f"static int Profile_{function}({m.group(1)}) {{",text)
        if i==0:
            text = """
static double ps_phase_seconds[5];
static unsigned long long ps_phase_calls[5];
void GetPSNativeProfile(double *seconds, unsigned long long *calls) {
    for (int i=0; i<5; ++i) {
        seconds[i] = ps_phase_seconds[i]; calls[i] = ps_phase_calls[i];
    }
}
""" + text
        text += f"""
int {function}({args}) {{
    const double begin = omp_get_wtime();
    const int status = Profile_{function}({names});
    ps_phase_seconds[{i}] += omp_get_wtime()-begin;
    ps_phase_calls[{i}]++;
    return status;
}}
"""
        path.write_text(text)
