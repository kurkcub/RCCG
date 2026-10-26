"""Synthesize the ordinary origin-Jacobian output-feedback controller.

Uses the same two-Riccati synthesis and weights as the Galerkin controller
(python/design.py), applied to the four-state Jacobian of the plant at the origin.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))
from design import design_controller  # noqa: E402


def main():
    data = dict(np.load(HERE.parent / "python/model_data.npz", allow_pickle=False))
    weights = {key[7:]: value for key, value in data.items() if key.startswith("weight_")}
    A = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-10000.0, -200.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-0.5 / 0.55, 0.0, -9.81 / 0.55, -1 / (0.9 * 0.55)],
        ]
    )
    B = np.array([[0.0], [10000.0], [0.0], [0.0]])
    C = np.eye(4)[2:4]
    E = np.vstack((np.zeros((2, 2)), np.eye(2)))
    controller, report = design_controller(dict(A=A, B=B, C=C, E=E), np.zeros((2, 1)), weights)
    out = HERE / "generated"
    out.mkdir(exist_ok=True)
    np.savez_compressed(
        out / "jacobian_controller.npz",
        **controller,
        plant_A=A,
        plant_B=B,
        plant_C=C,
    )
    (out / "jacobian_design.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
