"""Interval lower bound for the sampled MPC closure convolution."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction as F
from math import factorial
from pathlib import Path

import numpy as np
from mpmath import iv

from verification import require

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "witnesses"
iv.dps = 100


def point(q):
    q = F(q)
    return iv.mpf(int(q.numerator)) / int(q.denominator)


def intervals(a):
    return np.array([[point(q) for q in row] for row in a], dtype=object)


def interval_exp(a):
    n = len(a)
    norm = max(sum(abs(q) for q in row) for row in a)
    s = 0
    while norm > F(1, 8):
        norm /= 2
        s += 1
    small = intervals(a / F(2) ** s)
    eye = intervals(np.eye(n, dtype=int))
    term = eye.copy()
    total = eye.copy()
    order = 40
    for k in range(1, order + 1):
        term = (term @ small) / k
        total += term
    # Taylor remainder of the scaled exponential: ||R||_inf <= exp(norm) norm^(N+1)/(N+1)!
    # with exp(norm) <= exp(1/8) < 2; every entry error is bounded by this exact rational.
    error = 2 * norm ** (order + 1) / factorial(order + 1)
    interval_error = iv.mpf([-1, 1]) * point(error)
    total += interval_error
    for _ in range(s):
        total = total @ total
    return total, dict(
        scaling_squarings=s,
        taylor_order=order,
        scaled_infinity_norm_exact=str(norm),
        exact_entrywise_Taylor_remainder_bound=str(error),
        decimal_precision=iv.dps,
    )


def main():
    source = ROOT / "data/common_model.npz"
    data = np.load(source, allow_pickle=False)
    A = np.array([[F(float(v)) for v in row] for row in data["auxiliary_A"]], dtype=object)
    B = np.array([F(float(v)) for v in data["auxiliary_B"]], dtype=object)
    E = np.array([[F(float(v)) for v in row] for row in data["auxiliary_E"]], dtype=object)
    D = np.array([[F(float(v)) for v in row] for row in data["auxiliary_D_closure"]], dtype=object)
    G = E @ D
    splitfile = ROOT / "witnesses/mpc_exact_feedback_split.json"
    split = json.loads(splitfile.read_text())
    L = [F(q) for q in split["linear_cubic_coefficients_exact"]]
    matrices = np.load(ROOT / "data/mpc_realization.npz", allow_pickle=False)
    kprev = F(float(matrices["Kprev"]))
    generator = np.full((13, 13), F(0), dtype=object)
    generator[:8, :8] = A
    generator[:8, 8] = B
    generator[:8, 9:] = G
    exponential, metadata = interval_exp(generator * F(1, 200))
    discrete = np.empty((9, 9), dtype=object)
    discrete[:8, :] = exponential[:8, :9]
    discrete[8, :] = [point(q) for q in L + [kprev]]
    closure = np.full((9, 4), point(0), dtype=object)
    closure[:8, :] = exponential[:8, 9:]
    row = np.array([[point(int(i == 3)) for i in range(9)]], dtype=object)
    prefix = point(0)
    terms = []
    for j in range(40):
        value = (row @ closure)[0]
        norm = iv.sqrt(sum((v**2 for v in value), point(0)))
        prefix += norm
        terms.append(str(norm))
        row = row @ discrete
    threshold = F(507, 1000)
    require(
        prefix.a > point(threshold), "The closure prefix does not exceed the velocity threshold"
    )
    report = {
        "claim": (
            "The fixed independent-closure-channel convolution bound has velocity component "
            "greater than 0.507, already from 40 samples (0.2 s)."
        ),
        "scope": "Independent-channel containment test on the common cubic outer realization.",
        "exact_formula": (
            "sum(j=0..39)||e_v^T F9^j Gamma_D||2 <= integral-sum of the corresponding "
            "continuous closure kernels."
        ),
        "mathematical_reason": (
            "For each interval, norm(integral kernel) <= integral norm(kernel). Finite "
            "nonnegative prefix is a lower bound on the infinite independent-channel "
            "component gain."
        ),
        "sampling_period_exact": "1/200",
        "state_order": [
            "p1",
            "p2",
            "q",
            "v",
            "H3(v)/2",
            "H1(q)H2(v)/2",
            "H2(q)H1(v)/2",
            "H3(q)/2",
            "held u",
        ],
        "prefix_samples": 40,
        "prefix_duration": 0.2,
        "prefix_interval": str(prefix),
        "strict_rational_lower_bound": str(threshold),
        "strictly_exceeds_local_velocity_cap": bool(prefix.a > point(F(47, 100))),
        "enclosure_region_scope": "Local cubic enclosure with velocity cap 0.47 rad/s.",
        "matrix_exponential_enclosure": metadata,
        "term_intervals": terms,
        "included_input": "Dloc closure channel.",
        "initial_state_scope": "Zero initial state.",
        "paper_data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "feedback_split_sha256": hashlib.sha256(splitfile.read_bytes()).hexdigest(),
    }
    (OUT / "mpc_finite_prefix_closure_lower.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
