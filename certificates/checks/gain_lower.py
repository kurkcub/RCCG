"""Interval lower bound on the induced gain of the Jacobian closed loop at one frequency."""

from __future__ import annotations

import json
from fractions import Fraction as F
from pathlib import Path

import numpy as np
from mpmath import iv
from scipy.linalg import lu_factor

from verification import canonical_fraction, require

ROOT = Path(__file__).resolve().parents[1]


def as_float(q):
    return np.array([[float(x) for x in row] for row in q])


def interval_lower(abcd, omega=F(41, 10), fixed_input=None):
    """Certify ||G(j omega) v|| / ||v|| > 45.3 for a fixed rational input direction v.

    The floating-point SVD only proposes the direction; the frequency response is then
    solved with complex interval Gaussian elimination at 80 digits.
    """
    A, B, C, D = abcd
    an, bn, cn, dn = map(as_float, abcd)
    G = cn @ np.linalg.solve(1j * float(omega) * np.eye(len(A)) - an, bn) + dn
    _, _, Vh = np.linalg.svd(G)
    v = Vh.conj().T[:, 0]
    den = 10**8
    vre = [F(int(round(q.real * den)), den) for q in v]
    vim = [F(int(round(q.imag * den)), den) for q in v]
    if fixed_input is not None:
        vre = list(map(F, fixed_input["rounded_complex_input_real"]))
        vim = list(map(F, fixed_input["rounded_complex_input_imag"]))
    # Outward complex intervals validate the numerical pivot ordering.
    _, piv = lu_factor(1j * float(omega) * np.eye(len(A)) - an)
    iv.dps = 80

    def ivf(q):
        return iv.mpf(q.numerator) / q.denominator

    def ivc(re, im=F(0)):
        return iv.mpc(ivf(re), ivf(im))

    Mq = [[ivc(-A[i][j], omega if i == j else F(0)) for j in range(len(A))] for i in range(len(A))]
    vv = [ivc(re, im) for re, im in zip(vre, vim)]
    rhs = [sum((ivf(B[i][j]) * vv[j] for j in range(5)), ivc(F(0))) for i in range(len(A))]
    for k, pk in enumerate(piv):
        if pk != k:
            Mq[k], Mq[pk] = Mq[pk], Mq[k]
            rhs[k], rhs[pk] = rhs[pk], rhs[k]
        pivot = Mq[k][k]
        if 0 in pivot.real and 0 in pivot.imag:
            raise ArithmeticError("Interval pivot contains complex zero")
        for i in range(k + 1, len(A)):
            fac = Mq[i][k] / pivot
            for j in range(k + 1, len(A)):
                Mq[i][j] -= fac * Mq[k][j]
            rhs[i] -= fac * rhs[k]
            Mq[i][k] = ivc(F(0))
    solution = [ivc(F(0)) for _ in A]
    for i in reversed(range(len(A))):
        solution[i] = (
            rhs[i] - sum((Mq[i][j] * solution[j] for j in range(i + 1, len(A))), ivc(F(0)))
        ) / Mq[i][i]
    out = [
        sum((ivf(C[i][j]) * solution[j] for j in range(len(A))), ivc(F(0)))
        + sum((ivf(D[i][j]) * vv[j] for j in range(5)), ivc(F(0)))
        for i in range(2)
    ]
    norm2 = sum((re * re + im * im for re, im in zip(vre, vim)), F(0))

    def lower_abs2(z):
        value = iv.mpf(0)
        for q in [z.real, z.imag]:
            if 0 not in q:
                value += q * q
        return value

    square = sum((lower_abs2(z) for z in out), iv.mpf(0)) / ivf(norm2)
    threshold = F(453, 10)
    gain = iv.sqrt(square)
    require(
        square.a > ivf(threshold * threshold), "The interval gain does not exceed the threshold"
    )
    report = dict(
        frequency_exact=str(omega),
        rounded_complex_input_real=[str(q) for q in vre],
        rounded_complex_input_imag=[str(q) for q in vim],
        input_norm_squared_exact=str(norm2),
        output_intervals=[str(z) for z in out],
        gain_squared_interval=str(square),
        gain_interval=str(gain),
        rigorous_gain_lower_bound=str(threshold),
        strict_lower_bound_passed=True,
        interval_decimal_precision=80,
        proof=(
            "The fixed rational input vector has certified output energy/input energy "
            ">45.3^2 at omega=4.1. This is a lower bound on the common-outer induced norm, "
            "independent of storage choice."
        ),
    )
    return report


def main():
    d = dict(np.load(ROOT / "data/common_model.npz", allow_pickle=False))
    m = {key[6:]: value for key, value in d.items() if key.startswith("plant_")}
    k = dict(np.load(ROOT / "data/jacobian_nominal_controller.npz", allow_pickle=False))
    target = ROOT / "witnesses/jacobian_frequency_gain_lower.json"
    fixed = json.loads(target.read_text()) if target.exists() else None
    result = interval_lower(canonical_fraction(m, d["shape_Draw"], k), fixed_input=fixed)
    (ROOT / "witnesses/jacobian_frequency_gain_lower.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
