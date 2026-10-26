"""Galerkin construction, synthesis, regional certificates and the sweep for Figures 1 and 2.

Steps 1-4 start from the fixed inputs in paper_data.npz and write model_data.npz
and results.json; step 5 runs the degree study and step 6 the finite-horizon
weighted-gain sweep. Figures are drawn by plot_figures.py.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache/mpl"))

import numpy as np

from degree_comparison import run as degree_study
from design import ORDERS, design_controller, galerkin_model, verify_galerkin
from localization import certify_localization
from sweep import run_sweep
from verification import (
    generate_storage,
    verify_auxiliary,
    verify_initial,
    verify_shape,
    verify_storage,
)

ROOT = Path(__file__).resolve().parent


def _group(data, prefix):
    return {
        key[len(prefix) + 1 :]: value for key, value in data.items() if key.startswith(prefix + "_")
    }


def main():
    started = time.perf_counter()
    data = dict(np.load(ROOT / "paper_data.npz", allow_pickle=False))
    model, weight, auxiliary = (_group(data, name) for name in ("plant", "weight", "auxiliary"))
    RD, U = data["shape_Draw"], data["shape_U"]

    print("1/6 Galerkin construction", flush=True)
    stored = model["A"].copy()
    mdi = galerkin_model(model["bounds"], degree=3, orders=ORDERS)
    for key in ("A", "B", "C", "E"):
        model[key] = np.asarray(mdi[key]).ravel() if key == "B" else np.asarray(mdi[key])
        data["plant_" + key] = model[key]
    A = model["A"]
    consistency = verify_galerkin(model)
    blend = float(auxiliary["blend"])
    auxiliary["A"] = A.copy()
    base_row = np.r_[auxiliary["acceleration_base"], np.zeros(4)]
    auxiliary["A"][3] = (1 - blend) * base_row + blend * A[3]
    data["auxiliary_A"] = auxiliary["A"]

    print("2/6 H-infinity synthesis", flush=True)
    controller, synthesis = design_controller(model, U @ RD, weight)

    print("3/6 Residual and storage certificates", flush=True)
    shape = verify_shape(model, RD)
    auxiliary_check = verify_auxiliary(auxiliary)
    full, perf, storage = generate_storage(model, RD, controller)
    matrix_checks = verify_storage(model, RD, controller, full, perf)
    initial = verify_initial(full)

    print("4/6 Regional containment", flush=True)
    localization = certify_localization(
        auxiliary, controller, progress=lambda s: print("  " + s, flush=True)
    )
    data.update({"controller_" + k: v for k, v in controller.items()})
    data.update({"certificate_" + k: np.asarray(v) for k, v in full.items()})
    data.update(
        P_performance=perf["P"],
        gamma_performance=np.asarray(perf["gamma"]),
        epsilon_performance=np.asarray(perf["epsilon"]),
        coordinate_bounds=np.asarray(localization["physical_bounds"]),
    )
    np.savez_compressed(ROOT / "model_data.npz", **data)
    report = dict(
        integration=dict(
            method="MDI-TP",
            metadata=mdi["metadata"],
            tensor_consistency_error=consistency,
            stored_matrix_max_change=float(np.max(abs(A - stored))),
        ),
        synthesis=synthesis,
        shaped_residual=shape,
        auxiliary_residual=auxiliary_check,
        storage=storage,
        matrix_checks=matrix_checks,
        coherent_initial=initial,
        localization=localization,
    )
    # Written now as well, so that the certificate results survive a failure below.
    (ROOT / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    print("5/6 Modeling and degree study", flush=True)
    degree_study(data)

    print("6/6 Finite-horizon weighted-gain sweep", flush=True)
    run_sweep(data, ROOT)

    report["seconds"] = time.perf_counter() - started
    (ROOT / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Containment bounds:", localization["physical_bounds"], flush=True)
    print("Elapsed %.1f seconds" % report["seconds"], flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures-only", action="store_true", help="Redraw from a completed run.")
    args = parser.parse_args()
    if not args.figures_only:
        main()
    from plot_figures import run as draw

    draw(ROOT)
