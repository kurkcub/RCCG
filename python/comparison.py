"""Six nonlinear tracking comparisons without MATLAB.

Running this file generates the comparison data and Figure 5.
run() returns the data for the full study.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from localization import verify_reference
import time

import numpy as np
from numba import njit
from scipy.io import loadmat

BS_FILTER_W = 500.0
OUTPUT_STEP = 0.001
STOP_TIME = 175.0
CASE_NAMES = ('Hinf_nominal', 'Hinf_P1', 'Hinf_P2',
              'BS_nominal', 'BS_P1', 'BS_P2')


def physical_coefficients():
    """Same point-mass and uniform-rod parameters as init_comparison.m."""
    parameters = ((1.1, .9, 1., 1.), (1.1, 1.8, 1., 1.),
                  (5.5, .45, 2., 4.))
    coefficients = []
    for i, (M, m, length, damping) in enumerate(parameters):
        ell = length if i < 2 else length / 2
        inertia = m * length**2 if i < 2 else m * length**2 / 3
        coefficients.append((m*9.81*ell/inertia, damping/inertia,
            inertia*(M+m)/(m*ell), m*ell,
            m*m*ell*ell/((M+m)*inertia)))
    return np.array(coefficients), parameters


@njit(inline='always')
def _acceleration(x, k):
    sn, cs, v = np.sin(x[2]), np.cos(x[2]), x[3]
    return (-k[0]*sn-k[1]*v-cs/k[2]*(x[0]+k[3]*sn*v*v))/(1-k[4]*cs*cs)


@njit(inline='always')
def _bs_command(x, rjet):
    # The nominal Lie derivatives are used for all three physical plants.
    p, p2, q, v = x
    sn, cs = np.sin(q), np.cos(q)
    denominator = 1-.45*cs*cs
    d1 = .9*cs*sn
    d2 = .9*(cs*cs-sn*sn)
    numerator = -9.81*sn-v/.9-.5*p*cs-.45*cs*sn*v*v
    n1 = -9.81*cs+.5*p*sn-.45*(cs*cs-sn*sn)*v*v
    n2 = 9.81*sn+.5*p*cs+1.8*cs*sn*v*v
    nv = -1/.9-.9*cs*sn*v
    nqv = -.9*(cs*cs-sn*sn)*v
    a = numerator/denominator
    aq = n1/denominator-numerator*d1/denominator**2
    av = nv/denominator
    aqq = (n2/denominator-2*n1*d1/denominator**2
           -numerator*d2/denominator**2+2*numerator*d1*d1/denominator**3)
    aqv = nqv/denominator-nv*d1/denominator**2
    avv = -.9*cs*sn/denominator
    b = -.5*cs/denominator
    bq = .5*sn/denominator+.5*cs*d1/denominator**2
    phi3 = b*p2+aq*v+av*a
    phi4 = (2*bq*p2*v+aqq*v*v+2*aqv*v*a+avv*a*a
            +b*(-10000*p-200*p2)+aq*a+av*phi3)
    return (rjet[4]-phi4+109*(rjet[0]-q)+126*(rjet[1]-v)
            +57*(rjet[2]-a)+12*(rjet[3]-phi3))/(10000*b)


@njit(inline='always')
def _reference_jet(tau, slope, delta, r, w, out):
    # Exact dirty derivatives for the current linear reference interval.
    # The four low-pass states are driven by the constant reference slope.
    z = w*tau
    e = np.exp(-z)
    u1 = e*delta[0]
    u2 = e*(delta[1]+z*delta[0])
    u3 = e*(delta[2]+z*delta[1]+.5*z*z*delta[0])
    u4 = e*(delta[3]+z*delta[2]+.5*z*z*delta[1]+z*z*z/6*delta[0])
    out[0] = r
    out[1] = slope+u1
    out[2] = w*(u1-u2)
    out[3] = w*w*(u1-2*u2+u3)
    out[4] = w*w*w*(u1-3*u2+3*u3-u4)


@njit(inline='always')
def _rhs(hx, bx, rjet, ak, bk, ck, dk, coefficients, hf, bf):
    nk = len(ak)
    for i in range(3):
        error = rjet[0]-hx[i, 2]
        command = dk*error
        for j in range(nk):
            command += ck[j]*hx[i, 4+j]
        hf[i, 0] = hx[i, 1]
        hf[i, 1] = -10000*hx[i, 0]-200*hx[i, 1]+10000*command
        hf[i, 2] = hx[i, 3]
        hf[i, 3] = _acceleration(hx[i], coefficients[i])
        for j in range(nk):
            value = bk[j]*error
            for k in range(nk):
                value += ak[j, k]*hx[i, 4+k]
            hf[i, 4+j] = value
        command = _bs_command(bx[i], rjet)
        bf[i, 0] = bx[i, 1]
        bf[i, 1] = -10000*bx[i, 0]-200*bx[i, 1]+10000*command
        bf[i, 2] = bx[i, 3]
        bf[i, 3] = _acceleration(bx[i], coefficients[i])


@njit
def _integrate(tk, rk, ak, bk, ck, dk, coefficients, step, output_step, w):
    nout = int(round((tk[-1]-tk[0])/output_step))+1
    y = np.zeros((nout, 7))
    hx, bx = np.zeros((3, 4+len(ak))), np.zeros((3, 4))
    hk, bkk = np.empty((4, 3, hx.shape[1])), np.empty((4, 3, 4))
    ht, bt = np.empty_like(hx), np.empty_like(bx)
    jet0, jetm, jete = np.empty(5), np.empty(5), np.empty(5)
    filters = np.zeros(4)
    coordinate_maxima = np.zeros((6, 4))
    command_maxima, error_maxima = np.zeros(6), np.zeros(6)
    rec = 0
    y[0, 6] = rk[0]
    for block in range(len(tk)-1):
        width = .02
        count = int(round(width/step))
        h = width/count
        save_every = int(round(output_step/h))
        slope = (rk[block+1]-rk[block])/width
        delta = filters-slope
        for j in range(count):
            tau = j*h
            r0 = rk[block]+(rk[block+1]-rk[block])*j/count
            rm = rk[block]+(rk[block+1]-rk[block])*(j+.5)/count
            re = rk[block]+(rk[block+1]-rk[block])*(j+1)/count
            _reference_jet(tau, slope, delta, r0, w, jet0)
            _reference_jet(tau+.5*h, slope, delta, rm, w, jetm)
            _reference_jet(tau+h, slope, delta, re, w, jete)
            _rhs(hx, bx, jet0, ak, bk, ck, dk, coefficients, hk[0], bkk[0])
            ht[:] = hx+.5*h*hk[0]
            bt[:] = bx+.5*h*bkk[0]
            _rhs(ht, bt, jetm, ak, bk, ck, dk, coefficients, hk[1], bkk[1])
            ht[:] = hx+.5*h*hk[1]
            bt[:] = bx+.5*h*bkk[1]
            _rhs(ht, bt, jetm, ak, bk, ck, dk, coefficients, hk[2], bkk[2])
            ht[:] = hx+h*hk[2]
            bt[:] = bx+h*bkk[2]
            _rhs(ht, bt, jete, ak, bk, ck, dk, coefficients, hk[3], bkk[3])
            hx += h/6*(hk[0]+2*hk[1]+2*hk[2]+hk[3])
            bx += h/6*(bkk[0]+2*bkk[1]+2*bkk[2]+bkk[3])
            for i in range(3):
                command = dk*(re-hx[i, 2])
                for k in range(len(ak)):
                    command += ck[k]*hx[i, 4+k]
                command_maxima[i] = max(command_maxima[i], abs(command))
                command_maxima[i+3] = max(command_maxima[i+3], abs(_bs_command(bx[i], jete)))
                error_maxima[i] = max(error_maxima[i], abs(re-hx[i, 2]))
                error_maxima[i+3] = max(error_maxima[i+3], abs(re-bx[i, 2]))
                for k in range(4):
                    coordinate_maxima[i, k] = max(coordinate_maxima[i, k], abs(hx[i, k]))
                    coordinate_maxima[i+3, k] = max(coordinate_maxima[i+3, k], abs(bx[i, k]))
            if (j+1) % save_every == 0:
                rec += 1
                for i in range(3):
                    y[rec, i] = hx[i, 2]
                    y[rec, i+3] = bx[i, 2]
                y[rec, 6] = re
        z = w*width
        e = np.exp(-z)
        filters[0] = slope+e*delta[0]
        filters[1] = slope+e*(delta[1]+z*delta[0])
        filters[2] = slope+e*(delta[2]+z*delta[1]+.5*z*z*delta[0])
        filters[3] = slope+e*(delta[3]+z*delta[2]+.5*z*z*delta[1]+z*z*z/6*delta[0])
    return y[:rec+1], coordinate_maxima, command_maxima, error_maxima


def _load_inputs(root):
    candidates = (root/'comparison_data.mat', root.parent/'matlab'/'comparison_data.mat')
    reference_path = next((p for p in candidates if p.is_file()), None)
    if reference_path is None:
        raise FileNotFoundError('Expected comparison_data.mat here or in the sibling matlab folder.')
    snapshot = loadmat(reference_path, simplify_cells=True)
    source = root/'Galerkin_Control.mat'
    if source.is_file():
        loaded = loadmat(source, simplify_cells=True)
        values = [loaded[q+'K'] for q in 'ABCD']
    elif (root/'results.npz').is_file():
        source = root/'results.npz'
        with np.load(source, allow_pickle=False) as loaded:
            values = [loaded['controller_'+q] for q in 'ABCD']
    else:
        source = reference_path
        values = [snapshot[q] for q in 'ABCD']
    ak, bk, ck, dk = values
    ak = np.asarray(ak, dtype=float)
    bk = np.asarray(bk, dtype=float).reshape(-1)
    ck = np.asarray(ck, dtype=float).reshape(-1)
    dk = float(np.asarray(dk).item())
    if ak.shape != (14, 14) or bk.shape != (14,) or ck.shape != (14,):
        raise ValueError('Expected the 14-state SISO controller.')
    if not all(np.all(np.isfinite(x)) for x in (ak, bk, ck, dk)):
        raise ValueError('Controller matrices must be finite.')
    reference = np.asarray(snapshot['r_test'], dtype=float)
    return reference, (ak, bk, ck, dk), reference_path, source


def run(root=None, step=1e-5, stop_time=STOP_TIME):
    root = Path(__file__).resolve().parent if root is None else Path(root).resolve()
    if not np.isfinite(stop_time) or stop_time <= 0:
        raise ValueError('Stop time must be finite and positive.')
    reference, controller, reference_path, source = _load_inputs(root)
    if (not np.isfinite(step) or step <= 0 or step > 1e-5
            or not np.isclose(round(OUTPUT_STEP/step)*step, OUTPUT_STEP, rtol=0, atol=1e-13)):
        raise ValueError('Use a positive step no larger than 1e-5 that divides 0.001 seconds.')
    if (reference.ndim != 2 or reference.shape[1] != 2
            or len(reference) < 2 or not np.all(np.isfinite(reference))
            or not np.allclose(np.diff(reference[:, 0]), .02, rtol=0, atol=1e-11)
            or reference[0, 0] != 0 or reference[0, 1] != 0):
        raise ValueError('Expected the continuous 0.02-second reference record starting at zero.')
    if not np.isclose(round(stop_time/.02)*.02, stop_time, rtol=0, atol=1e-12):
        raise ValueError('Stop time must coincide with a reference knot.')
    reference[:,0] = verify_reference(reference[:,0],reference[:,1])
    reference = reference[reference[:, 0] <= stop_time+1e-10]
    if len(reference) < 2 or not np.isclose(reference[-1, 0], stop_time, rtol=0, atol=1e-10):
        raise ValueError('The reference record does not reach the requested stop time.')
    coefficients, parameters = physical_coefficients()
    start = time.monotonic()
    y, coordinates, commands, error_peaks = _integrate(reference[:, 0], reference[:, 1],
        *controller, coefficients, step, OUTPUT_STEP, BS_FILTER_W)
    elapsed = time.monotonic()-start
    t = np.arange(len(y))*OUTPUT_STEP
    if not all(np.all(np.isfinite(x)) for x in (y, coordinates, commands, error_peaks)):
        raise FloatingPointError('A nonlinear comparison trajectory became nonfinite.')
    error = y[:, 6:7]-y[:, :6]
    rmse = np.sqrt(np.trapezoid(error*error, t, axis=0)/stop_time)
    sample_rmse = np.sqrt(np.mean(error*error, axis=0))
    peak = np.max(abs(error), axis=0)
    data = dict(t=t, y=y, time_rmse=rmse, sample_rmse=sample_rmse,
        max_error=peak, max_error_internal=error_peaks,
        coordinate_maxima=coordinates, command_maxima=commands)
    np.savez_compressed(root/'comparison_results.npz', **data)
    digest = hashlib.sha256()
    for value in controller:
        digest.update(np.asarray(value).tobytes())
    report = dict(integration='RK4 physical nonlinear plants; exact PWL reference-filter propagation',
        step=step, output_step=OUTPUT_STEP, stop_time=stop_time, seconds=elapsed,
        initial_state='zero for all plant/controller/reference-filter states',
        BS_filter_w=BS_FILTER_W, BS_slow_coefficients=[109, 126, 57, 12, 1],
        controller_source=str(source.relative_to(root)) if source.is_relative_to(root) else '../matlab/comparison_data.mat',
        controller_sha256=digest.hexdigest(), reference_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
        plant_parameters_M_m_L_damping=parameters, plant_coefficients=coefficients.tolist(),
        column_order=list(CASE_NAMES)+['reference'], cases={})
    for i, name in enumerate(CASE_NAMES):
        report['cases'][name] = dict(time_rmse=float(rmse[i]), sample_rmse=float(sample_rmse[i]),
            max_error=float(peak[i]), max_error_internal=float(error_peaks[i]),
            command_maximum=float(commands[i]), coordinate_maxima=coordinates[i].tolist())
    (root/'comparison_results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    fields = ['case', 'time_rmse', 'sample_rmse', 'max_error', 'max_error_internal', 'command_maximum']
    with (root/'comparison_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, values in report['cases'].items():
            writer.writerow({'case': name, **{key: values[key] for key in fields[1:]}})
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=float, default=1e-5)
    args = parser.parse_args()
    output_root = Path(__file__).resolve().parent
    run(output_root, step=args.step)
    from plot_comparison import run as plot_comparison
    plot_comparison(output_root)
