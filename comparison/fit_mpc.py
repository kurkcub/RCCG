"""Physics predictor and residual fit for the Koopman-based hybrid MPC [49].

The physics predictor is the Galerkin projection of the sampled nominal
dynamics onto the Hermite dictionary with Gaussian weight; the data model is a
least-squares fit on the supplied one-step pairs, and the two are mixed with a
scalar weight chosen on held-out pairs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from numba import njit
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import expm

from simulations import actuator_rhs, features, mpc_gains, propagate_actuator

HERE = Path(__file__).resolve().parent
TS = 0.005
FIT_FRACTION = 0.9  # share of the pairs used for the data model; the rest sets the mixing weight


def dictionary():
    qv = []
    for degree in range(1, 7):
        qv.extend(
            sorted(
                (p for p in itertools.product(range(degree + 1), repeat=2) if sum(p) == degree),
                reverse=True,
            )
        )
    return np.vstack(
        (
            np.eye(4, dtype=np.int64)[:2],
            np.column_stack((np.zeros((len(qv), 2), dtype=np.int64), qv)),
        )
    )


@njit
def features_dot(x, xdot, modes):
    ns, n = x.shape
    degree = np.max(modes)
    h = np.ones((ns, n, degree + 1))
    h[:, :, 1] = 2 * x
    for d in range(1, degree):
        h[:, :, d + 1] = 2 * x * h[:, :, d] - 2 * d * h[:, :, d - 1]
    dp = np.zeros((ns, len(modes)))
    for j in range(len(modes)):
        for a in range(n):
            power = modes[j, a]
            if power:
                for k in range(ns):
                    value = power * xdot[k, a]
                    for b in range(n):
                        value *= h[k, b, modes[j, b] - (1 if a == b else 0)]
                    dp[k, j] += value
    return dp


@njit
def sampled_rhs(x, u):
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = actuator_rhs(x[i], u[i])
    return out


def physics_generator(modes):
    orders = [7, 7, 96, 8]
    nodes = []
    weights = []
    for order in orders:
        nd, wt = hermgauss(order)
        nodes.append(nd)
        weights.append(wt / np.sqrt(np.pi))
    grid = np.array(np.meshgrid(*nodes, indexing="ij")).reshape(4, -1).T
    wgrid = np.prod(np.array(np.meshgrid(*weights, indexing="ij")), axis=0).ravel()
    N = len(modes)
    norms = np.zeros(N + 1)
    cross = np.zeros((N + 1, N + 1))
    for command in [-10 / np.sqrt(3), 10 / np.sqrt(3)]:
        for first in range(0, len(grid), 6000):
            x = grid[first : first + 6000]
            u = np.full(len(x), command)
            wt = 0.5 * wgrid[first : first + 6000]
            phi = np.column_stack((features(x, modes), u))
            dphi = np.column_stack((features_dot(x, sampled_rhs(x, u), modes), np.zeros(len(x))))
            norms += np.sum(phi * phi * wt[:, None], axis=0)
            cross += (dphi.T * wt) @ phi
    return cross / norms[None, :]


@njit
def coverage_trajectory(u, x0):
    x = np.empty((len(u), 4))
    x[0] = x0
    for k in range(len(u) - 1):
        x[k + 1] = propagate_actuator(x[k], u[k], steps=10)
    return x


@njit
def one_step_pairs(X, U):
    Y = np.empty_like(X)
    for i in range(len(X)):
        Y[i] = propagate_actuator(X[i], U[i], steps=10)
    return Y


def training_pairs():
    rng = np.random.default_rng(481910)
    time = np.arange(8001) * TS
    knots = np.arange(21) * 2.0
    states = []
    for _ in range(10):
        x0 = np.array([0.0, 0.0, rng.uniform(-0.1, 0.1), 0.0])
        command = rng.uniform(-16, 16, 21)
        command[0] = 0.0
        u = np.interp(time, knots, command)
        states.append(coverage_trajectory(u, x0)[:-1])
    X = np.asarray(states).reshape(-1, 4)
    U = np.random.default_rng(481911).uniform(-16, 16, len(X))
    return X, one_step_pairs(X, U), U


def hybrid_fit(X, Y, U, modes, Kp):
    alpha = np.column_stack((features(X, modes), U))
    beta = np.column_stack((features(Y, modes), U))
    split = int(FIT_FRACTION * len(alpha))
    scale = np.max(abs(alpha), axis=0)
    left, singular, right = np.linalg.svd(alpha[:split] / scale, full_matrices=False)
    tolerance = max(alpha[:split].shape) * np.spacing(singular[0])
    inverse = np.zeros_like(singular)
    inverse[singular > tolerance] = 1 / singular[singular > tolerance]
    normalized = (right.T * inverse) @ (left.T @ (beta[:split] / scale))
    Ke = scale[:, None] * normalized.T / scale[None, :]
    correction = Ke - Kp
    predicted = alpha[split:] @ correction.T
    target = beta[split:] - alpha[split:] @ Kp.T
    gamma = float(np.clip(np.sum(predicted * target) / np.sum(predicted * predicted), 0, 1))
    return Ke, Kp + gamma * correction, gamma


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regenerate-data",
        action="store_true",
        help="Generate the supplied nominal one-step pairs from their seeds.",
    )
    args = parser.parse_args()
    modes = dictionary()
    out = HERE / "generated"
    out.mkdir(exist_ok=True)
    if args.regenerate_data:
        X, Y, U = training_pairs()
        np.savez_compressed(out / "mpc_training.npz", X=X, Y=Y, U=U, Ts=TS, seed=481911)
    else:
        pairs = np.load(HERE / "mpc_training.npz", allow_pickle=False)
        X, Y, U = [pairs[key] for key in ["X", "Y", "U"]]
    G = physics_generator(modes)
    Kp = expm(TS * G)
    Ke, Kc, gamma = hybrid_fit(X, Y, U, modes, Kp)
    Kx, Kr, Kprev = mpc_gains(Kc, H=200, R=1e-6)
    np.savez_compressed(
        out / "mpc_model.npz",
        G=G,
        Kp=Kp,
        Ke=Ke,
        Kc=Kc,
        modes=modes,
        mixing_weight=gamma,
        Kx=Kx,
        Kr=Kr,
        Kprev=Kprev,
    )
    report = dict(
        state_observables=len(modes),
        Ts=TS,
        fit_pairs=int(FIT_FRACTION * len(X)),
        mixing_pairs=len(X) - int(FIT_FRACTION * len(X)),
        mixing_weight=gamma,
    )
    if args.regenerate_data:
        saved = np.load(HERE / "mpc_training.npz", allow_pickle=False)
        report["training_pair_max_difference"] = {
            key: float(np.max(abs(value - saved[key])))
            for key, value in [("X", X), ("Y", Y), ("U", U)]
        }
    (out / "mpc_fit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
