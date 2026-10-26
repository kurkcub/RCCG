"""Exact checks of all nine PWL reference sequences, including end jumps."""

from __future__ import annotations

import json
from fractions import Fraction as F
from pathlib import Path

import numpy as np

from localization import KNOT_SPACING, REFERENCE_AMPLITUDE, REFERENCE_SLOPE, REFERENCE_SLOPE_JUMP
from verification import require

ROOT = Path(__file__).resolve().parents[1]


def main():
    d = np.load(ROOT / "data/reference_knots.npz", allow_pickle=False)
    h = KNOT_SPACING
    R, nu, J = REFERENCE_AMPLITUDE, REFERENCE_SLOPE, REFERENCE_SLOPE_JUMP
    report = {}
    for key in d.files:
        values = [F(float(x)) for x in d[key]]
        require(values[0] == 0, f"{key}: the reference must start at zero")
        slopes = [(right - left) / h for left, right in zip(values, values[1:])]
        jumps = (
            [slopes[0]] + [right - left for left, right in zip(slopes, slopes[1:])] + [-slopes[-1]]
        )
        a = max(map(abs, values))
        b = max(map(abs, slopes))
        c = max(map(abs, jumps))
        require(a <= R and b <= nu and c <= J, f"{key}: outside the common reference class")
        report[key] = {
            "amplitude_exact": str(a),
            "slope_exact": str(b),
            "slope_jump_exact": str(c),
            "passes_common_class": True,
            "preview_end_time_exact": str((len(values) - 1) * h),
        }
    require(len(report) == 9, "Expected the nine reference cases of the comparison")
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
