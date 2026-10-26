"""Validate exact modal intervals and the common reference-class bounds."""

from __future__ import annotations

import gzip
import json
from fractions import Fraction as F
from pathlib import Path

import numpy as np

import localization as loc
from verification import require

ROOT = Path(__file__).resolve().parents[1]


def decode(value):
    if isinstance(value, dict):
        if set(value) == {"fraction"}:
            return F(value["fraction"])
        return {key: decode(x) for key, x in value.items()}
    if isinstance(value, list):
        return [decode(x) for x in value]
    return value


def load_modal(name):
    with gzip.open(ROOT / "witnesses" / f"{name}_modal_intervals.json.gz", "rt") as f:
        return decode(json.load(f))


def validate_modal(model, k, data):
    a, b = loc.closed_loop_matrices(model, k)
    characteristic, numerators = loc.exact_transfer(a, b)
    require(
        list(map(str, characteristic)) == data["exact_transfers"]["characteristic"],
        "The stored characteristic polynomial differs from the exact one",
    )
    centers = data["centers"]
    radius = data["radius"]
    n = len(a)
    require(len(centers) == n, "The pole disks do not cover the characteristic degree")
    derivative = [(j + 1) * characteristic[j + 1] for j in range(n)]
    for j, center in enumerate(centers):
        sh = loc.shifted(characteristic, center)
        linear = loc.modulus(sh[1], False) * radius
        other = loc.modulus(sh[0]) + sum(
            (loc.modulus(sh[v]) * radius**v for v in range(2, n + 1)), F(0)
        )
        require(linear > other and center[0] + radius < 0, "A stable Rouche disk check failed")
        require(
            all(
                (center[0] - old[0]) ** 2 + (center[1] - old[1]) ** 2 > (2 * radius) ** 2
                for old in centers[:j]
            ),
            "Candidate pole disks overlap",
        )
        ds = loc.shifted(derivative, center)
        variation = sum((loc.modulus(ds[v]) * radius**v for v in range(1, len(ds))), F(0))
        denominator = loc.modulus(ds[0], False) - variation
        require(denominator > 0, "Denominator derivative enclosure reaches zero")
        for i in range(4):
            for column in range(11):
                ns = loc.shifted(numerators[i][column], center)
                residue = loc.cdiv(ns[0], ds[0])
                nominal = data["residues"][j][i][column]
                nv = sum((loc.modulus(ns[v]) * radius**v for v in range(1, len(ns))), F(0))
                error = (nv + loc.modulus(residue) * variation) / denominator + loc.modulus(
                    loc.csub(nominal, residue)
                )
                require(
                    data["errors"][j][i][column] >= error, "A stored residue error is too small"
                )
    dc = [numerators[i][4][0] / characteristic[0] for i in range(4)]
    moment = [
        (numerators[i][4][0] * characteristic[1] - numerators[i][4][1] * characteristic[0])
        / characteristic[0] ** 2
        for i in range(4)
    ]
    require(
        list(map(str, dc)) == data["exact_transfers"]["reference_DC_exact"],
        "The stored reference DC values differ from the exact ones",
    )
    require(
        list(map(str, moment)) == data["exact_transfers"]["reference_first_moment_exact"],
        "The stored reference first moments differ from the exact ones",
    )
    # Form the exact error transfer (D-N)/D directly from the angle transfer N/D.
    error_numerator = characteristic.copy()
    for j, value in enumerate(numerators[2][4]):
        error_numerator[j] -= value
    error_dc = error_numerator[0] / characteristic[0]
    error_moment = (
        error_numerator[0] * characteristic[1]
        - error_numerator[1] * characteristic[0]
    ) / characteristic[0] ** 2
    require(
        error_numerator[-1] == characteristic[-1]
        and error_dc == 1 - dc[2] and error_moment == -moment[2],
        "The exact tracking-error readout differs from r-x1",
    )
    # G-G(0)+m1*s has a double zero at the origin. Dividing by s^2
    # leaves only the stable characteristic denominator used by the lattice bound.
    primitive_numerator = [F(0)] * (len(characteristic) + 1)
    for j, value in enumerate(numerators[2][4]):
        primitive_numerator[j] += value
    for j, value in enumerate(characteristic):
        primitive_numerator[j] -= dc[2] * value
        primitive_numerator[j + 1] += moment[2] * value
    require(
        primitive_numerator[0] == primitive_numerator[1] == 0,
        "The reference second primitive has an uncancelled origin pole",
    )
    return {
        "exact_transfer_identity": True,
        "stable_disjoint_Rouche_disks": n,
        "residue_intervals_validated": n * 4 * 11,
        "exact_reference_DC_and_first_moments": True,
        "tracking_error_transfer": dict(
            direct_feedthrough_exact="1", error_DC_exact=str(error_dc),
            error_first_moment_exact=str(error_moment),
            second_primitive_origin_cancellation=True,
        ),
    }


def recompute_envelope(data):
    loc.iv.dps = 45
    closure = [loc.bound_vector_l1(data, i, [0, 1, 2, 3], rtol=1e-3) for i in range(4)]
    coefficients = list(map(F, data["metadata"]["initial_coefficient_bounds_exact"]))
    initial = [
        loc.bound_initial_sup(data, i, list(range(5, 11)), coefficients, atol=tolerance)
        for i, tolerance in enumerate([5e-4, 0.05, 1e-5, 1e-5])
    ]
    reference = loc.certify_reference(data)
    error = loc.tracking_error_bound(
        data, F(closure[2]["upper"]), F(initial[2]["upper"]),
        F(reference["coordinate_details"][2]["true_lattice_envelope_upper"]),
    )
    return {"closure": closure, "initial": initial, "reference": reference,
            "tracking_error": error}


def main(full=False):
    d = dict(np.load(ROOT / "data/common_model.npz", allow_pickle=False))
    model = {key[10:]: value for key, value in d.items() if key.startswith("auxiliary_")}
    bounds = json.loads((ROOT / "witnesses/localization_source_bounds.json").read_text())["cases"]
    poly = json.loads((ROOT / "witnesses/reference_polytope.json").read_text())
    witness = json.loads((ROOT / "witnesses/common_reference_class.json").read_text())
    R, nu, J = [
        F(witness["reference_class"][key])
        for key in ["amplitude_exact", "slope_exact", "slope_jump_exact"]
    ]
    require(
        (R, nu, J) == (loc.REFERENCE_AMPLITUDE, loc.REFERENCE_SLOPE, loc.REFERENCE_SLOPE_JUMP)
        and F(witness["reference_class"]["knot_spacing_exact"]) == loc.KNOT_SPACING,
        "The witness reference class differs from the class in localization.py",
    )
    box = list(map(lambda v: F(float(v)), model["bounds"]))
    report = {
        "reference_class": witness["reference_class"],
        "cases": {},
        "full_envelope_recomputed": full,
    }
    for name in ["galerkin3", "jacobian_nominal"]:
        k = dict(np.load(ROOT / "data" / f"{name}_controller.npz", allow_pickle=False))
        modal = load_modal(name)
        result = validate_modal(model, k, modal)
        source = bounds[name]
        require(np.array_equal(source["auxiliary_box"], model["bounds"]), "Auxiliary box mismatch")
        if full:
            print("Recomputing continuous and lattice enclosures:", name, flush=True)
            fresh = recompute_envelope(modal)
            (ROOT / "witnesses" / f"{name}_recomputed_envelopes.json").write_text(
                json.dumps(fresh, indent=2) + "\n"
            )
            # The exact pole proposals can differ below the disk radius. The
            # saved proof uses a conservative directed maximum if needed.
            for i in range(4):
                require(
                    fresh["closure"][i]["upper"] <= source["closure_bounds"][i],
                    f"{name}: recomputed closure bound {i} exceeds the stored one",
                )
                require(
                    fresh["initial"][i]["upper"] <= source["initial_bounds"][i],
                    f"{name}: recomputed initial bound {i} exceeds the stored one",
                )
                require(
                    fresh["reference"]["coordinate_details"][i]["true_lattice_envelope_upper"]
                    <= source["reference_certificate"]["coordinate_details"][i][
                        "true_lattice_envelope_upper"
                    ],
                    f"{name}: recomputed lattice envelope {i} exceeds the stored one",
                )
        physical = []
        margins = []
        for i, row in enumerate(poly["cases"][name]["polytope"]):
            closure = F(float(source["closure_bounds"][i]))
            initial = F(float(source["initial_bounds"][i]))
            detail = source["reference_certificate"]["coordinate_details"][i]
            dc = abs(F(modal["exact_transfers"]["reference_DC_exact"][i]))
            moment = abs(F(modal["exact_transfers"]["reference_first_moment_exact"][i]))
            lattice = F(float(detail["true_lattice_envelope_upper"]))
            require(
                F(row["amplitude_coefficient"]["exact"]) == dc
                and F(row["slope_coefficient"]["exact"]) == moment
                and F(row["slope_jump_coefficient_upper"]["exact"]) == lattice,
                f"{name}: the reference polytope coefficients differ from the exact values",
            )
            value = closure + initial + dc * R + moment * nu + lattice * J
            margin = box[i] - value
            require(
                str(value) == witness["cases"][name]["physical_bounds_exact"][i]
                and str(margin) == witness["cases"][name]["strict_margins_exact"][i],
                f"{name}: the stored exact bound or margin {i} does not match",
            )
            physical.append(str(value))
            margins.append(str(margin))
        error = loc.tracking_error_bound(
            modal, F(source["closure_bounds"][2]), F(source["initial_bounds"][2]),
            F(source["reference_certificate"]["coordinate_details"][2][
                "true_lattice_envelope_upper"]),
        )
        for key in ["error_DC_exact", "error_first_moment_exact"]:
            require(
                error[key] == result["tracking_error_transfer"][key],
                f"{name}: the error-envelope coefficient differs from its exact transfer",
            )
        error["all_time"] = all(F(q) > 0 for q in margins)
        require(
            error == witness["cases"][name]["tracking_error"],
            f"{name}: the stored tracking-error bound differs from its exact sum",
        )
        if full:
            require(
                F(fresh["tracking_error"]["total_exact"]) <= F(error["total_exact"]),
                f"{name}: the recomputed tracking-error envelope exceeds the stored bound",
            )
        result.update(
            physical_bounds_exact=physical,
            physical_bounds=list(map(lambda q: float(F(q)), physical)),
            margins_exact=margins,
            strict_containment=all(F(q) > 0 for q in margins),
            tracking_error=error,
        )
        report["cases"][name] = result
    return report


if __name__ == "__main__":
    import sys

    print(json.dumps(main("--full" in sys.argv), indent=2))
