"""CUDA auxiliary-table quadrature shared by the v3 lightcone backends."""

import json
import os
from pathlib import Path

import cupy as cp


def table_source(filename):
    directory = Path(__file__).parent
    return (directory / "table_qag.cuh").read_text() + (
        directory / filename
    ).read_text()


def table_rule():
    return cp.asarray(
        json.loads(Path(os.environ["PS_GK61_RULE"]).read_text()), dtype=cp.float64
    )
