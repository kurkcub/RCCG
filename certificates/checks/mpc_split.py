"""Exact polynomial split of the Koopman-based hybrid MPC feedback."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

import numpy as np

from verification import require

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "witnesses"
MODEL = ROOT / "data/mpc_realization.npz"


def hermite(n):
    previous = [0]
    current = [1]
    for degree in range(n):
        nxt = [0] + [2 * c for c in current]
        for j, c in enumerate(previous):
            nxt[j] -= 2 * degree * c
        previous, current = current, nxt
    return current


def hvalue(n, x):
    return sum((F(c) * x**j for j, c in enumerate(hermite(n))), F(0))


def main():
    model = np.load(MODEL, allow_pickle=False)
    gain = list(map(lambda x: F(float(x)), model["Kx"]))
    modes = model["modes"]
    poly = {}
    for coefficient, mode in zip(gain[2:], modes[2:]):
        require(mode[0] == 0 and mode[1] == 0, "Lift modes must not depend on the actuator states")
        for i, ci in enumerate(hermite(int(mode[2]))):
            for j, cj in enumerate(hermite(int(mode[3]))):
                poly[i, j] = poly.get((i, j), F(0)) + coefficient * F(ci * cj, 2)
    poly = {power: coefficient for power, coefficient in poly.items() if coefficient}
    constant = poly.get((0, 0), F(0))
    L = [
        gain[0],
        gain[1],
        poly.get((1, 0), F(0)),
        poly.get((0, 1), F(0)),
        poly.get((0, 3), F(0)) / 4,
        poly.get((1, 2), F(0)) / 4,
        poly.get((2, 1), F(0)) / 4,
        poly.get((3, 0), F(0)) / 4,
    ]
    # H3(t)/2=4t^3−6t, while H1(q)H2(v)/2=4qv²−2q.
    L[2] += 6 * L[7] + 2 * L[5]
    L[3] += 6 * L[4] + 2 * L[6]
    remainder = {
        power: coefficient for power, coefficient in poly.items() if sum(power) not in [0, 1, 3]
    }
    reconstructed = dict(remainder)
    reconstructed.update(
        {
            (0, 0): constant,
            (1, 0): L[2] - 2 * L[5] - 6 * L[7],
            (0, 1): L[3] - 6 * L[4] - 2 * L[6],
            (0, 3): 4 * L[4],
            (1, 2): 4 * L[5],
            (2, 1): 4 * L[6],
            (3, 0): 4 * L[7],
        }
    )
    require(
        {power: value for power, value in reconstructed.items() if value} == poly,
        "The cubic split does not reproduce the full feedback polynomial",
    )
    require(
        list(modes[0]) == [1, 0, 0, 0] and list(modes[1]) == [0, 1, 0, 0],
        "The first two observables must be the actuator coordinates",
    )
    rng = np.random.default_rng(202610212)
    for _ in range(40):
        p, p2, q, v = [F(int(value), 1024) for value in rng.integers(-700, 701, 4)]
        state = [p, p2, q, v]
        original = sum(
            (
                coefficient
                * F(1, 2)
                * np.prod(
                    [hvalue(int(degree), coordinate) for degree, coordinate in zip(mode, state)]
                )
                for coefficient, mode in zip(gain, modes)
            ),
            F(0),
        )
        z = [
            p,
            p2,
            q,
            v,
            F(1, 2) * hvalue(3, v),
            F(1, 2) * hvalue(1, q) * hvalue(2, v),
            F(1, 2) * hvalue(2, q) * hvalue(1, v),
            F(1, 2) * hvalue(3, q),
        ]
        eta = sum((coefficient * q**i * v**j for (i, j), coefficient in remainder.items()), F(0))
        split = constant + sum((a * b for a, b in zip(L, z)), F(0)) + eta
        require(original == split, "The exact split identity failed at a random rational point")
    result = dict(
        realization_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        exact_identity="Kx*Psi29(x)=constant+L8*Z8(x)+eta(q,v)",
        raw_cubic_order=[
            "p1",
            "p2",
            "q",
            "v",
            "H3(v)/2",
            "H1(q)H2(v)/2",
            "H2(q)H1(v)/2",
            "H3(q)/2",
        ],
        H3_convention="Physicists H3(t)/2=4t^3−6t.",
        constant_exact=str(constant),
        constant=float(constant),
        linear_cubic_coefficients_exact=list(map(str, L)),
        linear_cubic_coefficients=list(map(float, L)),
        remainder=[
            dict(q_degree=i, v_degree=j, coefficient_exact=str(value), coefficient=float(value))
            for (i, j), value in sorted(remainder.items())
        ],
        exact_identity_test_count=40,
        all_exact_identities_passed=True,
        full_polynomial_coefficient_identity_passed=True,
        Kprev=float(model["Kprev"]),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mpc_exact_feedback_split.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
