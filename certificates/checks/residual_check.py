"""Outward interval validation of the full and local shaped residual sets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from verification import verify_auxiliary, verify_shape

ROOT = Path(__file__).resolve().parents[1]


def main():
    d = dict(np.load(ROOT / "data/common_model.npz", allow_pickle=False))
    plant = {key[6:]: value for key, value in d.items() if key.startswith("plant_")}
    auxiliary = {key[10:]: value for key, value in d.items() if key.startswith("auxiliary_")}
    report = {
        "full_region_residual": verify_shape(plant, d["shape_Draw"]),
        "local_cubic_residual": verify_auxiliary(auxiliary),
        "full_region": plant["bounds"].tolist(),
        "local_region": auxiliary["bounds"].tolist(),
    }
    (ROOT / "witnesses/residual_interval_results.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
