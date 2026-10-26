"""Exact bounded-real checks for the two designed continuous controllers."""

from __future__ import annotations

import gzip
import json
from fractions import Fraction as F
from pathlib import Path

import numpy as np

from verification import (
    add,
    blockput,
    canonical_fraction,
    exact_ldl_pd,
    eye,
    mul,
    rational,
    require,
    scale,
    tr,
    zero,
)

ROOT = Path(__file__).resolve().parents[1]


def serialized(matrix):
    return [[str(q) for q in row] for row in matrix]


def main(write=True):
    source = dict(np.load(ROOT / "data/common_model.npz", allow_pickle=False))
    model = {key[6:]: value for key, value in source.items() if key.startswith("plant_")}
    report = {
        "input_order": ["wc1", "wc2", "wc3", "wc4", "r/0.682"],
        "output_order": ["WP(r-q)", "WU(u)"],
        "cases": {},
    }
    weight_keys = [f"{letter}_{weight}" for weight in ["WP", "WU"] for letter in "ABCD"]
    controllers = {
        name: dict(np.load(ROOT / "data" / f"{name}_controller.npz", allow_pickle=False))
        for name in ["galerkin3", "jacobian_nominal"]
    }
    require(
        all(
            np.array_equal(controllers["galerkin3"][key], controllers["jacobian_nominal"][key])
            for key in weight_keys
        ),
        "The two controllers must carry identical weights",
    )
    require(
        all(
            np.array_equal(controllers["galerkin3"][key], source["weight_" + key])
            for key in weight_keys
        ),
        "The controller weights differ from the common model weights",
    )
    for name, k in controllers.items():
        w = dict(np.load(ROOT / "data" / f"{name}_brl_witness.npz", allow_pickle=False))
        a, b, c, d = canonical_fraction(model, source["shape_Draw"], k)
        n = len(a)
        t = rational(w["performance_analysis_transform"])
        ttt = mul(tr(t), t)
        case = {
            "state_order": str(w["full_state_order"]),
            "transform_gram": exact_ldl_pd(ttt, name + ":T.T*T"),
            "witnesses": {},
        }
        witnesses = [
            (
                "performance",
                F(str(float(w["performance_gamma"]))) ** 2,
                F(float(w["performance_epsilon"])),
                rational(w["performance_P"]),
            )
        ]
        if name == "galerkin3":
            require(
                np.array_equal(w["performance_analysis_transform"], w["full_analysis_transform"]),
                "The storage and performance witnesses must share the analysis coordinates",
            )
            witnesses.append(
                (
                    "coercive_storage",
                    F(float(w["full_gamma2"])),
                    F(float(w["full_epsilon"])),
                    rational(w["full_P"]),
                )
            )
        for tag, g2, epsilon, P in witnesses:
            require(
                g2 > 0 and epsilon > 0, f"{name}:{tag}: gamma squared and epsilon must be positive"
            )
            pc = mul(mul(tr(t), P), t)
            upper = add(mul(tr(a), pc), mul(pc, a), mul(tr(c), c), scale(ttt, epsilon))
            cross = add(mul(pc, b), mul(tr(c), d))
            lower = add(mul(tr(d), d), scale(eye(5), -g2))
            lmi = zero(n + 5, n + 5)
            blockput(lmi, upper, 0, 0)
            blockput(lmi, cross, 0, n)
            blockput(lmi, tr(cross), n, 0)
            blockput(lmi, lower, n, n)
            negative = scale(lmi, -1)
            checks = {
                "P_analysis": exact_ldl_pd(P, name + ":" + tag + ":P"),
                "negative_BRL": exact_ldl_pd(negative, name + ":" + tag + ":-L"),
                "epsilon_exact": str(epsilon),
                "gamma_squared_exact": str(g2),
                "canonical_coercivity_metric": "epsilon*T.T*T",
                "canonical_storage": "P_canonical=T.T*P_analysis*T",
            }
            target = ROOT / "witnesses" / f"{name}_{tag}_exact.json.gz"
            artifact = {
                "A_exact": serialized(a),
                "B_exact": serialized(b),
                "C_exact": serialized(c),
                "D_exact": serialized(d),
                "P_analysis_exact": serialized(P),
                "analysis_transform_exact": serialized(t),
                "P_canonical_exact": serialized(pc),
                "negative_BRL_residual_exact": serialized(negative),
                "epsilon_exact": str(epsilon),
                "gamma_squared_exact": str(g2),
                "checks": checks,
            }
            if target.exists():
                with gzip.open(target, "rt") as f:
                    saved = json.load(f)
                for key in artifact:
                    if key != "checks":
                        require(artifact[key] == saved[key], name + ": saved exact matrix differs")
            if write:
                with gzip.open(target, "wt") as f:
                    json.dump(artifact, f, separators=(",", ":"))
            case["witnesses"][tag] = checks
        case["performance_gamma"] = float(w["performance_gamma"])
        case["performance_epsilon"] = float(w["performance_epsilon"])
        report["cases"][name] = case
    if write:
        (ROOT / "witnesses/brl_check_results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
