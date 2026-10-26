"""Cart-pendulum plant, cubic Hermite lift and the fixed closed-loop vector fields.

Model-independent functions shared by run.py, degree_comparison.py and sweep.py.
The physical state is (p1, p2, q, v): the two actuator states, the pendulum angle
and its angular velocity.
"""

from __future__ import annotations

import numpy as np
from numba import njit


def acceleration(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    p, _, theta, v = x.T
    sn, cs = np.sin(theta), np.cos(theta)
    return (-9.81 * sn - v / 0.9 - 0.5 * p * cs - 0.45 * cs * sn * v * v) / (1 - 0.45 * cs * cs)


def lift(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    q, v = x[:, 2], x[:, 3]
    return np.column_stack(
        (
            x[:, :2],
            q,
            v,
            4 * v**3 - 6 * v,
            4 * q * v * v - 2 * q,
            4 * q * q * v - 2 * v,
            4 * q**3 - 6 * q,
        )
    )


def drift(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    q, v = x[:, 2], x[:, 3]
    acc = acceleration(x)
    return np.column_stack(
        (
            v,
            acc,
            (12 * v * v - 6) * acc,
            4 * v**3 - 2 * v + 8 * q * v * acc,
            8 * q * v * v + (4 * q * q - 2) * acc,
            (12 * q * q - 6) * v,
        )
    )


def residual(x: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Closure residual gamma(x) - A_lift z(x) of the lifted realization A."""
    return drift(x) - lift(x) @ A[2:].T


def normalized_residual(x: np.ndarray, A: np.ndarray, RD: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Residual in the shaped coordinates RD^-1 U^T r(x)."""
    return np.linalg.solve(RD, (residual(x, A) @ U).T).T


@njit
def _family(x0, h, duration, save_every, box):
    steps = int(round(duration / h))
    n = len(x0)
    x = x0.copy()
    active = np.ones(n, dtype=np.bool_)
    count = steps // save_every + 1
    history = np.empty((count, n, 4))
    valid = np.empty((count, n), dtype=np.bool_)
    exits = np.full(n, np.nan)
    history[0], valid[0] = x, active
    k = np.empty((4, 4))
    v = np.empty(4)
    rec = 1
    for step in range(steps):
        for i in range(n):
            for stage in range(4):
                v[:] = x[i]
                if stage in (1, 2):
                    v += 0.5 * h * k[stage - 1]
                elif stage == 3:
                    v += h * k[2]
                t = step * h + (0 if stage == 0 else h if stage == 3 else h / 2)
                u = x0[i, 0] + 2 * np.sin(0.2 * t)
                sn, cs = np.sin(v[2]), np.cos(v[2])
                k[stage, 0] = v[1]
                k[stage, 1] = -10000 * v[0] - 200 * v[1] + 10000 * u
                k[stage, 2] = v[3]
                k[stage, 3] = (
                    -9.81 * sn - v[3] / 0.9 - 0.5 * v[0] * cs - 0.45 * cs * sn * v[3] ** 2
                ) / (1 - 0.45 * cs**2)
            x[i] += h * (k[0] + 2 * k[1] + 2 * k[2] + k[3]) / 6
            if active[i] and np.any(np.abs(x[i]) >= box):
                active[i] = False
                exits[i] = (step + 1) * h
        if (step + 1) % save_every == 0:
            history[rec], valid[rec] = x, active
            rec += 1
    return history, valid, exits


def residual_family(data, step=0.00025) -> dict[str, np.ndarray]:
    """Shaped residual along a family of open-loop trajectories (Figure 1b).

    data holds plant_A, plant_bounds, shape_Draw and shape_U of the realization.
    """
    sample_stride, _ = output_marks(step, np.array([0.005, 10.0]))
    q, v = np.meshgrid(np.linspace(-0.7, 0.7, 17), np.linspace(-0.45, 0.45, 17), indexing="ij")
    x0 = np.column_stack((-19.62 * np.tan(q.ravel()), np.zeros(q.size), q.ravel(), v.ravel()))
    history, active, exits = _family(x0, step, 10.0, int(sample_stride), data["plant_bounds"])
    flat = history.reshape(-1, 4)
    norm = np.linalg.norm(
        normalized_residual(flat, data["plant_A"], data["shape_Draw"], data["shape_U"]), axis=1
    ).reshape(history.shape[:2])
    return dict(
        t=np.linspace(0, 10, len(history)), xi=history, active=active, exits=exits, normalized=norm
    )


@njit(inline="always")
def _feedback_rhs(x, r, ak, bk, ck, dk, aw, bw, au, bu, out):
    nk, nw = len(ak), len(aw)
    e = r - x[2]
    u = dk * e
    for j in range(nk):
        u += ck[j] * x[4 + j]
    sn, cs, v = np.sin(x[2]), np.cos(x[2]), x[3]
    out[0] = x[1]
    out[1] = -10000 * x[0] - 200 * x[1] + 10000 * u
    out[2] = v
    out[3] = (-9.81 * sn - v / 0.9 - 0.5 * cs * x[0] - 0.45 * cs * sn * v * v) / (
        1 - 0.45 * cs * cs
    )
    for i in range(nk):
        value = bk[i] * e
        for j in range(nk):
            value += ak[i, j] * x[4 + j]
        out[4 + i] = value
    for i in range(nw):
        value = bw[i] * e
        for j in range(nw):
            value += aw[i, j] * x[4 + nk + j]
        out[4 + nk + i] = value
    for i in range(len(au)):
        value = bu[i] * u
        for j in range(len(au)):
            value += au[i, j] * x[4 + nk + nw + j]
        out[4 + nk + nw + i] = value


def output_marks(step, offsets):
    """Require positive, knot-aligned steps before entering the RK4 loops."""
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Integration step must be finite and positive.")
    marks = np.rint(np.asarray(offsets) / step).astype(np.int64)
    if np.any(marks <= 0) or not np.allclose(marks * step, offsets, rtol=0, atol=1e-12):
        raise ValueError("Integration step must divide all output offsets.")
    return marks
