"""Controller and reference inputs shared by simulations and certificates."""

from pathlib import Path
import hashlib

import numpy as np

HERE = Path(__file__).resolve().parent


def load_data():
    data = dict(np.load(HERE / "inputs.npz", allow_pickle=False))
    gal = dict(np.load(HERE.parent / "python/model_data.npz", allow_pickle=False))
    jac = dict(np.load(HERE / "generated/jacobian_controller.npz", allow_pickle=False))
    mpc = dict(np.load(HERE / "generated/mpc_model.npz", allow_pickle=False))
    for q in "ABCD":
        data["galerkin3_" + q] = gal["controller_" + q]
        data["jacobian_" + q] = jac[q]
    data.update({k: v for k, v in gal.items() if k.startswith("weight_")})
    for dst, src in {
        "model": "Kc", "modes": "modes", "Kx": "Kx", "Kr": "Kr", "Kprev": "Kprev",
        "generator": "G", "physics_model": "Kp", "fitted_model": "Ke",
        "mixing_weight": "mixing_weight",
    }.items():
        data["mpc_" + dst] = mpc[src]
    gain = float(data["bs_common_gain"])
    data["bs_feedback_coefficients"] = np.array(
        [gain**4 + 3 * gain**2 + 1, 4 * gain**3 + 6 * gain, 6 * gain**2 + 3, 4 * gain, 1]
    )
    return data


def fingerprint(data):
    digest = hashlib.sha256()
    for key in sorted(data):
        value = np.ascontiguousarray(data[key])
        digest.update(key.encode())
        digest.update(str((value.dtype.str, value.shape)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()
