"""Build certificate witnesses for the controllers used by the simulations."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
from fractions import Fraction as F
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "python"))
sys.path.insert(1, str(ROOT.parent / "comparison"))
sys.path.insert(2, str(ROOT / "checks"))

import numpy as np
import localization as loc
from design_data import fingerprint, load_data
from localization_check import recompute_envelope, validate_modal
from verification import generate_storage, require

NAMES = {"galerkin3": "galerkin3", "jacobian_nominal": "jacobian"}


def group(data, prefix):
    return {k[len(prefix) + 1:]: v for k, v in data.items() if k.startswith(prefix + "_")}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def fraction_json(value):
    if isinstance(value, F):
        return {"fraction": str(value)}
    raise TypeError(type(value).__name__)


def check_sources():
    data = load_data()
    current = dict(np.load(ROOT.parent / "python/model_data.npz", allow_pickle=False))
    common = dict(np.load(ROOT / "data/common_model.npz", allow_pickle=False))
    for key, value in common.items():
        require(np.array_equal(value, current[key]), "Common model differs from the design: " + key)
    for name, prefix in NAMES.items():
        k = dict(np.load(ROOT / "data" / f"{name}_controller.npz", allow_pickle=False))
        for q in "ABCD":
            require(np.array_equal(k[q], data[prefix + "_" + q]), name + ": controller source mismatch")
        for key, value in group(data, "weight").items():
            require(np.array_equal(k[key], value), name + ": weight source mismatch")
    mpc = dict(np.load(ROOT / "data/mpc_realization.npz", allow_pickle=False))
    for dest, src in {"Kc": "model", "modes": "modes", "Kx": "Kx", "Kr": "Kr", "Kprev": "Kprev"}.items():
        require(np.array_equal(mpc[dest], data["mpc_" + src]), "MPC source mismatch: " + src)
    refs = np.load(ROOT / "data/reference_knots.npz", allow_pickle=False)
    for name, values in zip(data["case_ids"], data["references"]):
        require(np.array_equal(refs[str(name)], values), "Reference source mismatch: " + str(name))


def write_plot_summary(report):
    localization = report["localization"]
    rows = []
    for name, title in [("galerkin3", "Galerkin d=3 + H-infinity"), ("jacobian_nominal", "Jacobian + H-infinity")]:
        case = localization["cases"][name]
        bounds = [loc.directed_float(F(q)) for q in case["physical_bounds_exact"]]
        rows.append(dict(controller=title, physical_bounds_upper=bounds,
                         velocity_closure_component_strict_lower=None,
                         fixed_containment_test_passed=case["strict_containment"]))
    rows.append(dict(controller="Koopman-based hybrid MPC [49]", physical_bounds_upper=None,
                     velocity_closure_component_strict_lower=float(F(report["mpc_closure_lower"]["strict_rational_lower_bound"])),
                     fixed_containment_test_passed=False))
    write_json(ROOT.parent / "comparison/certificate_summary.json",
               dict(reference_class=localization["reference_class"], rows=rows, design_fingerprint=fingerprint(load_data())))


def main():
    data = load_data()
    generated = dict(np.load(ROOT.parent / "python/model_data.npz", allow_pickle=False))
    common = {k: v for k, v in generated.items() if k.startswith(("plant_", "shape_", "weight_", "auxiliary_"))}
    model, auxiliary, weights = (group(common, name) for name in ("plant", "auxiliary", "weight"))
    folder = ROOT / "data"
    witnesses = ROOT / "witnesses"
    folder.mkdir(exist_ok=True)
    witnesses.mkdir(exist_ok=True)
    for path in witnesses.glob("*_exact.json.gz"):
        path.unlink()
    for path in witnesses.glob("*_recomputed_envelopes.json"):
        path.unlink()
    (witnesses / "jacobian_frequency_gain_lower.json").unlink(missing_ok=True)
    np.savez_compressed(folder / "common_model.npz", **common)
    np.savez_compressed(folder / "reference_knots.npz", **dict(zip(map(str, data["case_ids"]), data["references"])))
    np.savez_compressed(folder / "mpc_realization.npz", Kc=data["mpc_model"], modes=data["mpc_modes"],
                        Kx=data["mpc_Kx"], Kr=data["mpc_Kr"], Kprev=data["mpc_Kprev"], H=200, R=1e-6)
    reference_class = dict(amplitude_exact=str(loc.REFERENCE_AMPLITUDE), slope_exact=str(loc.REFERENCE_SLOPE),
                           slope_jump_exact=str(loc.REFERENCE_SLOPE_JUMP), knot_spacing_exact=str(loc.KNOT_SPACING))
    source, polytope = {"cases": {}}, {"cases": {}}
    box = list(map(lambda v: F(float(v)), auxiliary["bounds"]))
    regional = dict(reference_class=reference_class, auxiliary_box_exact=list(map(str, box)), cases={})
    for name, prefix in NAMES.items():
        print("Building certificates:", name, flush=True)
        controller = dict(weights, **{q: data[prefix + "_" + q] for q in "ABCD"})
        filename = folder / f"{name}_controller.npz"
        np.savez_compressed(filename, **controller)
        if name == "galerkin3":
            full = group(generated, "certificate")
            perf = dict(P=generated["P_performance"], gamma=generated["gamma_performance"],
                        epsilon=generated["epsilon_performance"], analysis_transform=full["analysis_transform"])
        else:
            full, perf, report = generate_storage(model, common["shape_Draw"], controller)
            write_json(witnesses / f"{name}_storage_search.json", report)
        w = {"performance_" + key: perf[key] for key in ("P", "gamma", "epsilon", "analysis_transform")}
        w["full_state_order"] = full["state_order"]
        if name == "galerkin3":
            w.update({"full_" + key: full[key] for key in
                      ("P", "epsilon", "gamma2", "analysis_transform", "M", "a", "analysis_inverse",
                       "Acanonical", "Bcanonical", "Ccanonical", "Dcanonical", "Hcanonical", "input_order")})
        np.savez_compressed(folder / f"{name}_brl_witness.npz", **w)
        modal = loc.build_kernels(auxiliary, controller, progress=lambda s: print("  " + s, flush=True))
        modal_checks = validate_modal(auxiliary, controller, modal)
        with gzip.open(witnesses / f"{name}_modal_intervals.json.gz", "wt") as stream:
            json.dump(modal, stream, default=fraction_json, separators=(",", ":"))
        fresh = recompute_envelope(modal)
        write_json(witnesses / f"{name}_recomputed_envelopes.json", fresh)
        closure = [np.nextafter(q["upper"], np.inf).item() for q in fresh["closure"]]
        initial = [np.nextafter(q["upper"], np.inf).item() for q in fresh["initial"]]
        lattice = [np.nextafter(q["true_lattice_envelope_upper"], np.inf).item() for q in fresh["reference"]["coordinate_details"]]
        source["cases"][name] = dict(auxiliary_box=auxiliary["bounds"].tolist(), closure_bounds=closure,
                                     initial_bounds=initial, reference_certificate=dict(coordinate_details=[
                                         dict(true_lattice_envelope_upper=q) for q in lattice]))
        rows, values, margins = [], [], []
        for i in range(4):
            dc = abs(F(modal["exact_transfers"]["reference_DC_exact"][i]))
            moment = abs(F(modal["exact_transfers"]["reference_first_moment_exact"][i]))
            base = F(closure[i]) + F(initial[i])
            coeff = F(lattice[i])
            value = base + dc * loc.REFERENCE_AMPLITUDE + moment * loc.REFERENCE_SLOPE + coeff * loc.REFERENCE_SLOPE_JUMP
            margin = box[i] - value
            rows.append(dict(coordinate=i, amplitude_coefficient=dict(exact=str(dc), decimal=float(dc)),
                             slope_coefficient=dict(exact=str(moment), decimal=float(moment)),
                             slope_jump_coefficient_upper=dict(exact=str(coeff), decimal=float(coeff)),
                             remaining_budget=dict(exact=str(box[i] - base), decimal=float(box[i] - base))))
            values.append(value)
            margins.append(margin)
        polytope["cases"][name] = dict(polytope=rows)
        error = loc.tracking_error_bound(modal, F(closure[2]), F(initial[2]), F(lattice[2]))
        for key in ["error_DC_exact", "error_first_moment_exact"]:
            require(error[key] == modal_checks["tracking_error_transfer"][key],
                    name + ": tracking-error coefficient differs from its exact transfer")
        error["all_time"] = all(q > 0 for q in margins)
        regional["cases"][name] = dict(controller_source_sha256=hashlib.sha256(filename.read_bytes()).hexdigest(),
            physical_bounds_upper=list(map(loc.directed_float, values)), physical_bounds_exact=list(map(str, values)),
            strict_margins_exact=list(map(str, margins)), strict_containment=all(q > 0 for q in margins),
            common_outer_BRL_gamma=float(perf["gamma"]), tracking_error=error)
        if name == "galerkin3":
            require(all(q > 0 for q in margins), "Galerkin containment failed")
        print("  bounds:", regional["cases"][name]["physical_bounds_upper"], flush=True)
    write_json(witnesses / "localization_source_bounds.json", source)
    write_json(witnesses / "reference_polytope.json", polytope)
    write_json(witnesses / "common_reference_class.json", regional)
    check_sources()


if __name__ == "__main__":
    main()
