"""Run the nine reference cases with the four fixed controllers."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np

import simulations as sim
from design_data import load_data

HERE = Path(__file__).resolve().parent
DT = 0.005  # output record spacing (s)
T = 80.0  # test duration (s)
PREVIEW = 1.0  # reference preview available to MPC (s)


def metrics(case, name, b, omega, t, x, r, u, completed, status, regional):
    e = r - x[:, 2]
    row = dict(
        case=case,
        controller=name,
        mean=float(b),
        omega=float(omega),
        completed=bool(completed),
        status=status,
        inside_Omega=regional["inside_Omega"],
        inside_local_box=regional["inside_local_box"],
        first_Omega_exit=regional["first_Omega_exit"],
        first_local_box_exit=regional["first_local_box_exit"],
        regional_monitoring=regional["regional_monitoring"],
        stop_time=float(t[-1]),
    )
    if completed:
        row.update(
            total_RMSE=float(np.sqrt(np.trapezoid(e * e, t) / T)),
            total_peak_error=float(np.max(abs(e))),
        )
    return row


def native_region(first_O, first_L):
    return dict(
        inside_Omega=bool(first_O < 0),
        inside_local_box=bool(first_L < 0),
        first_Omega_exit=None if first_O < 0 else float(first_O),
        first_local_box_exit=None if first_L < 0 else float(first_L),
        regional_monitoring="native integration substeps",
    )


def saved_region(t, x):
    outside_box = np.flatnonzero(np.any(abs(x) > sim.BOX, axis=1))
    outside_local = np.flatnonzero(np.any(abs(x) > sim.LOCAL, axis=1))
    return dict(
        inside_Omega=len(outside_box) == 0,
        inside_local_box=len(outside_local) == 0,
        first_Omega_exit=None if len(outside_box) == 0 else float(t[outside_box[0]]),
        first_local_box_exit=None if len(outside_local) == 0 else float(t[outside_local[0]]),
        regional_monitoring="saved 0.005 second grid",
    )


def simulate_hinf(k, matrix, reference):
    r = reference[: int(round(T / DT)) + 1]
    result = sim.hinf_kernel(r, DT, np.zeros(4 + len(k["A"])), k["C"], k["D"], *matrix, 0.0005)
    x, r, u, _, _, _, _, _, stop, first_O, first_L, code = result
    t = np.arange(len(r)) * DT
    t[-1] = stop
    return (
        t,
        x,
        r,
        u,
        code == 0,
        ["completed", "escape", "nonfinite"][code],
        native_region(first_O, first_L),
        {},
    )


def simulate_mpc(data, reference):
    x, u, stop, code, planned = sim.mpc_kernel(
        reference,
        int(round(T / DT)),
        np.zeros(4),
        0.0,
        data["mpc_modes"],
        data["mpc_Kx"],
        data["mpc_Kr"],
        float(data["mpc_Kprev"]),
        0.0005,
    )
    t = np.arange(stop + 1) * DT
    r = reference[: stop + 1]
    u = np.r_[u, u[-1]]
    return (
        t,
        x,
        r,
        u,
        code == 0,
        ["completed", "nonfinite", "angle_guard", "actuator_or_command_guard"][code],
        saved_region(t, x),
        {},
    )


def simulate_bs(data, knots):
    result = sim.bs_kernel(
        knots[:4001],
        data["bs_coefficients"],
        1e-5,
        DT,
        float(data["bs_filter_w"]),
        data["bs_feedback_coefficients"],
    )
    x, filters, jet, r, u, maximum, maxu, maxe, energy, stop, first_O, first_L, code = result
    t = np.arange(len(r)) * DT
    t[-1] = stop
    extra = dict(reference_filter_state=filters, reference_jet=jet)
    return (
        t,
        x,
        r,
        u,
        code == 0,
        ["completed", "nonfinite", "guard_stop"][code],
        native_region(first_O, first_L),
        extra,
    )


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    traces = output / "traces"
    traces.mkdir(exist_ok=True)
    data = load_data()
    np.savez_compressed(output / "controllers_used.npz", **data)
    controllers = {
        name: {letter: data[name + "_" + letter] for letter in "ABCD"}
        for name in ["jacobian", "galerkin3"]
    }
    matrices = {name: sim.matrices(k, 0.0005) for name, k in controllers.items()}
    time = np.arange(int(round((T + PREVIEW) / DT)) + 1) * DT
    rows = []
    for i, (case, b, omega) in enumerate(
        zip(data["case_ids"], data["offsets"], data["frequencies"])
    ):
        case = str(case)
        knots = data["references"][i]
        reference = np.interp(time, data["reference_knots"], knots)
        for name in ["jacobian", "galerkin3", "backstepping", "hybrid_mpc"]:
            if name in controllers:
                result = simulate_hinf(controllers[name], matrices[name], reference)
            elif name == "backstepping":
                result = simulate_bs(data, knots)
            else:
                result = simulate_mpc(data, reference)
            t, x, r, u, completed, status, region, extra = result
            row = metrics(case, name, b, omega, t, x, r, u, completed, status, region)
            relative = Path("traces") / f"{case}_{name}.npz"
            np.savez_compressed(
                output / relative,
                time=t,
                state=x,
                reference=r,
                error=r - x[:, 2],
                control=u,
                **extra,
            )
            row["trace_path"] = relative.as_posix()
            rows.append(row)
        (output / "comparison_results.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(f"{i + 1}/{len(data['case_ids'])} {case}", flush=True)
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with (output / "comparison_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "results")
    run(parser.parse_args().output)
