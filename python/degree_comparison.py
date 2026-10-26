"""Degree study of the Galerkin lift along one open-loop trajectory (Figures 1a and 1c).

Compares the nominal LTI realizations of degrees 1, 3 and 5 with the nonlinear
plant, replays the degree-3 model with the exact closure term, and collects
the residual family for Figure 1b.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache/mpl"))

import numpy as np
from numba import njit
from numpy.polynomial.hermite import hermder, hermval
from scipy.integrate import solve_ivp

from design import ORDERS, galerkin_model
from plant import acceleration, residual_family

ROOT = Path(__file__).resolve().parent


def herm(n, x, derivative=False):
    c = np.zeros(n + 1)
    c[n] = 1
    return hermval(x, hermder(c) if derivative else c)


def lift(x, modes):
    x = np.atleast_2d(x)
    q, v = x[:, 2:].T
    return np.column_stack([x[:, :2]] + [0.5 * herm(a, q) * herm(b, v) for a, b in modes])


def drift(x, modes):
    x = np.atleast_2d(x)
    q, v = x[:, 2:].T
    a = acceleration(x)
    return np.column_stack(
        [
            0.5 * (herm(i, q, True) * herm(j, v) * v + herm(i, q) * herm(j, v, True) * a)
            for i, j in modes
        ]
    )


def projection(box, degree):
    model = galerkin_model(box, degree, ORDERS)
    diagnostics = dict(model["diagnostics"])
    diagnostics["quadrature"] = model["metadata"]
    return model["modes"], model["A"], model["B"][:, 0], diagnostics


@njit
def corrected_rhs(t, z, A, B, pairs):
    q, v = z[2], z[3]
    n = pairs.max()
    hq, hv = np.ones(n + 1), np.ones(n + 1)
    if n > 0:
        hq[1], hv[1] = 2 * q, 2 * v
    for k in range(1, n):
        hq[k + 1] = 2 * q * hq[k] - 2 * k * hq[k - 1]
        hv[k + 1] = 2 * v * hv[k] - 2 * k * hv[k - 1]
    sn, cs = np.sin(q), np.cos(q)
    acc = (-9.81 * sn - v / 0.9 - 0.5 * z[0] * cs - 0.45 * cs * sn * v * v) / (1 - 0.45 * cs * cs)
    phi = np.empty(len(z))
    phi[:2] = z[:2]
    exact = np.empty(len(pairs))
    for j in range(len(pairs)):
        a, b = pairs[j]
        phi[j + 2] = 0.5 * hq[a] * hv[b]
        exact[j] = 0.0
        if a > 0:
            exact[j] += a * hq[a - 1] * hv[b] * v
        if b > 0:
            exact[j] += b * hq[a] * hv[b - 1] * acc
    rhs = A @ z + B * (12 * np.sin(0.2 * t) + 2 * np.sin(0.47 * t))
    rhs[2:] += exact - A[2:] @ phi
    return rhs


def command(t):
    return 12 * np.sin(0.2 * t) + 2 * np.sin(0.47 * t)


def physical(t, x):
    return [x[1], -10000 * x[0] - 200 * x[1] + 10000 * command(t), x[3], acceleration(x)[0]]


def run(data):
    """data holds plant_A, plant_bounds, shape_Draw and shape_U of the realization."""
    box = data["plant_bounds"]
    t = np.linspace(0, 40, 8001)
    x0 = np.array([0.0, 0.0, 0.0, 0.1])
    opts = dict(t_eval=t, rtol=2e-12, atol=2e-14, method="DOP853")
    physical_solution = solve_ivp(physical, (0, 40), x0, **opts)
    if not physical_solution.success:
        raise RuntimeError("The nonlinear reference simulation failed.")
    x = physical_solution.y.T
    if not np.all(np.abs(x) < box):
        raise RuntimeError("The open-loop trajectory leaves the quadrature box.")
    stored = {"t": t, "xi": x}
    report = {
        "box": box.tolist(),
        "command": "12 sin(0.2 t) + 2 sin(0.47 t)",
        "initial_state": x0.tolist(),
        "time_horizon": 40.0,
        "quadrature_orders": list(ORDERS),
        "integration_algorithm": "MDI-TP with staged polynomial contractions",
        "physical_state_max": np.max(np.abs(x), axis=0).tolist(),
        "degree": {},
    }
    for d in (1, 3, 5):
        modes, A, B, summary = projection(box, d)
        z0 = lift(x0, modes)[0]
        nominal = solve_ivp(lambda t, z: A @ z + B * command(t), (0, 40), z0, **opts)
        if not nominal.success:
            raise RuntimeError(f"The degree-{d} nominal simulation failed.")
        error = nominal.y[2:4].T - x[:, 2:4]
        res = drift(x, modes) - lift(x, modes) @ A[2:].T
        summary.update(
            states=len(A),
            modes=modes,
            nominal_max_error=np.max(np.abs(error), axis=0).tolist(),
            nominal_rms_error=np.sqrt(np.trapezoid(error**2, t, axis=0) / t[-1]).tolist(),
            common_acceleration_residual_path_max=float(np.abs(res[:, 1]).max()),
            common_acceleration_residual_path_rms=float(
                np.sqrt(np.trapezoid(res[:, 1] ** 2, t) / t[-1])
            ),
        )
        if d == 3:
            pairs = np.asarray(modes, dtype=np.int64)
            replay = solve_ivp(lambda t, z: corrected_rhs(t, z, A, B, pairs), (0, 40), z0, **opts)
            if not replay.success:
                raise RuntimeError("The corrected degree-3 simulation failed.")
            stored["corrected_3"] = replay.y.T
            summary["corrected_max_error"] = np.max(
                np.abs(replay.y[2:4].T - x[:, 2:4]), axis=0
            ).tolist()
            summary["existing_model_max_coefficient_difference"] = float(
                np.max(np.abs(A - data["plant_A"]))
            )
        stored.update({f"A_{d}": A, f"B_{d}": B, f"nominal_{d}": nominal.y.T, f"residual_{d}": res})
        report["degree"][str(d)] = summary
        print("  degree", d, "maximum errors", summary["nominal_max_error"], flush=True)
    np.savez_compressed(ROOT / "degree_results.npz", **stored)
    (ROOT / "degree_results.json").write_text(json.dumps(report, indent=2) + "\n")
    family = residual_family(data)
    y = np.where(family["active"], family["normalized"], np.nan)
    np.savez_compressed(
        ROOT / "degree_figure_data.npz",
        t=t,
        xi=x,
        nominal_1=stored["nominal_1"],
        nominal_3=stored["nominal_3"],
        nominal_5=stored["nominal_5"],
        corrected_3=stored["corrected_3"],
        family_t=family["t"],
        family_lo=np.nanmin(y, axis=1),
        family_hi=np.nanmax(y, axis=1),
    )


if __name__ == "__main__":
    source = ROOT / "model_data.npz"
    if not source.exists():
        source = ROOT / "paper_data.npz"
    run(dict(np.load(source, allow_pickle=False)))
