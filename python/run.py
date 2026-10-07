from __future__ import annotations

import argparse
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "galerkin-mpl"))
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from design import verify_galerkin, design_controller, galerkin_model
from verification import (verify_shape, verify_auxiliary, generate_storage,
                          verify_storage, verify_initial)
from localization import verify_reference, certify_localization

ROOT = Path(__file__).resolve().parent
DATA = dict(np.load(ROOT / "paper_data.npz", allow_pickle=False))
A, B, C = (DATA["plant_" + k] for k in ("A", "B", "C"))
BOX = DATA["plant_bounds"]
NONNULL = DATA["plant_nonnull"]
RD = DATA["shape_Draw"]
U = DATA["shape_U"]


def acceleration(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    p, _, theta, v = x.T
    sn, cs = np.sin(theta), np.cos(theta)
    return (-9.81*sn-v/0.9-0.5*p*cs-0.45*cs*sn*v*v)/(1-0.45*cs*cs)


def lift(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    q, v = x[:, 2], x[:, 3]
    return np.column_stack((x[:, :2], q, v, 4*v**3-6*v,
                            4*q*v*v-2*q, 4*q*q*v-2*v, 4*q**3-6*q))


def drift(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    q, v = x[:, 2], x[:, 3]
    acc = acceleration(x)
    return np.column_stack((v, acc, (12*v*v-6)*acc,
        4*v**3-2*v+8*q*v*acc, 8*q*v*v+(4*q*q-2)*acc, (12*q*q-6)*v))


def residual(x: np.ndarray) -> np.ndarray:
    return drift(x)-lift(x) @ A[2:].T


def normalized_residual(x: np.ndarray) -> np.ndarray:
    return np.linalg.solve(RD, (residual(x) @ U).T).T


def realization_test() -> dict[str, np.ndarray]:
    t = np.linspace(0, 40, 8001)
    x0 = np.array([0., 0., 0., 0.1])
    def command(t):
        return 12*np.sin(0.2*t)+2*np.sin(0.47*t)
    def physical(t, x):
        return np.array([x[1], -10000*x[0]-200*x[1]+10000*command(t),
                         x[3], acceleration(x)[0]])
    def corrected(t, z):
        value = A @ z+B*command(t)
        value[2:] += residual(z[:4])[0]
        return value
    opts = dict(t_eval=t, rtol=2e-12, atol=2e-14, method="DOP853")
    exact = solve_ivp(corrected, (0, 40), lift(x0)[0], **opts)
    nominal = solve_ivp(lambda t,z: A @ z+B*command(t), (0, 40), lift(x0)[0], **opts)
    true = solve_ivp(physical, (0, 40), x0, **opts)
    if not all(s.success for s in (exact, nominal, true)):
        raise RuntimeError("The realization test did not integrate successfully.")
    x = true.y.T
    if np.any(np.abs(x) >= BOX):
        raise RuntimeError("The realization trajectory left the operating region.")
    return dict(t=t, xi=x, nonlinear=x[:, 2:], nominal=nominal.y.T @ C.T,
                corrected=exact.y.T @ C.T, residual=residual(x), wc=normalized_residual(x))


@njit
def _family(x0, h, duration, save_every):
    steps = int(round(duration/h))
    n = len(x0)
    x = x0.copy()
    active = np.ones(n, dtype=np.bool_)
    count = steps//save_every+1
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
                    v += 0.5*h*k[stage-1]
                elif stage == 3:
                    v += h*k[2]
                t = step*h+(0 if stage == 0 else h if stage == 3 else h/2)
                u = x0[i, 0]+2*np.sin(0.2*t)
                sn, cs = np.sin(v[2]), np.cos(v[2])
                k[stage, 0] = v[1]
                k[stage, 1] = -10000*v[0]-200*v[1]+10000*u
                k[stage, 2] = v[3]
                k[stage, 3] = (-9.81*sn-v[3]/.9-.5*v[0]*cs-.45*cs*sn*v[3]**2)/(1-.45*cs**2)
            x[i] += h*(k[0]+2*k[1]+2*k[2]+k[3])/6
            if active[i] and np.any(np.abs(x[i]) >= BOX):
                active[i] = False
                exits[i] = (step+1)*h
        if (step+1) % save_every == 0:
            history[rec], valid[rec] = x, active
            rec += 1
    return history, valid, exits


def residual_family(step=0.00025) -> dict[str, np.ndarray]:
    sample_stride, _ = output_marks(step, np.array([.005, 10.]))
    q, v = np.meshgrid(np.linspace(-.7, .7, 17), np.linspace(-.45, .45, 17), indexing="ij")
    x0 = np.column_stack((-19.62*np.tan(q.ravel()), np.zeros(q.size), q.ravel(), v.ravel()))
    history, active, exits = _family(x0, step, 10., int(sample_stride))
    flat = history.reshape(-1, 4)
    norm = np.linalg.norm(normalized_residual(flat), axis=1).reshape(history.shape[:2])
    return dict(t=np.linspace(0, 10, len(history)), xi=history, active=active,
                exits=exits, normalized=norm)


@njit(inline="always")
def _feedback_rhs(x, r, ak, bk, ck, dk, aw, bw, au, bu, out):
    nk, nw = len(ak), len(aw)
    e = r-x[2]
    u = dk*e
    for j in range(nk):
        u += ck[j]*x[4+j]
    sn, cs, v = np.sin(x[2]), np.cos(x[2]), x[3]
    out[0] = x[1]
    out[1] = -10000*x[0]-200*x[1]+10000*u
    out[2] = v
    out[3] = (-9.81*sn-v/.9-.5*cs*x[0]-.45*cs*sn*v*v)/(1-.45*cs*cs)
    for i in range(nk):
        value = bk[i]*e
        for j in range(nk):
            value += ak[i,j]*x[4+j]
        out[4+i] = value
    for i in range(nw):
        value = bw[i]*e
        for j in range(nw):
            value += aw[i,j]*x[4+nk+j]
        out[4+nk+i] = value
    for i in range(len(au)):
        value = bu[i]*u
        for j in range(len(au)):
            value += au[i,j]*x[4+nk+nw+j]
        out[4+nk+nw+i] = value


@njit
def _closed_loop(tk, rk, ak, bk, ck, dk, aw, bw, au, bu, step, marks, initial):
    blocks = len(tk)-1
    capacity = blocks*len(marks)+1+int(round(.02/step))
    tout = np.empty(capacity)
    rout = np.empty(capacity)
    states = np.empty((capacity, len(initial)))
    x = initial.copy()
    tout[0], rout[0], states[0] = tk[0], rk[0], x
    rec = 1
    k1, k2, k3, k4, tmp = [np.empty(len(x)) for _ in range(5)]
    coordinate_maxima = np.abs(initial[:4]).copy()
    error_peak = 0.
    control_peak = abs(dk*(rk[0]-initial[2])+np.dot(ck, initial[4:4+len(ak)]))
    for block in range(blocks):
        width = .02
        steps = int(round(width/step))
        h = width/steps
        r0, dr = rk[block], rk[block+1]-rk[block]
        mark = 0
        for j in range(steps):
            rr = r0+dr*j/steps
            rm = r0+dr*(j+.5)/steps
            re = r0+dr*(j+1.)/steps
            _feedback_rhs(x, rr, ak, bk, ck, dk, aw, bw, au, bu, k1)
            tmp[:] = x+.5*h*k1
            _feedback_rhs(tmp, rm, ak, bk, ck, dk, aw, bw, au, bu, k2)
            tmp[:] = x+.5*h*k2
            _feedback_rhs(tmp, rm, ak, bk, ck, dk, aw, bw, au, bu, k3)
            tmp[:] = x+h*k3
            _feedback_rhs(tmp, re, ak, bk, ck, dk, aw, bw, au, bu, k4)
            x += h*(k1+2*k2+2*k3+k4)/6.
            for k in range(4):
                coordinate_maxima[k] = max(coordinate_maxima[k], abs(x[k]))
            error_peak = max(error_peak, abs(re-x[2]))
            command = dk*(re-x[2])+np.dot(ck,x[4:4+len(ak)])
            control_peak = max(control_peak, abs(command))
            save = mark < len(marks) and j+1 == marks[mark]
            if save:
                mark += 1
            if save or block == 0:
                tout[rec] = tk[block]+(j+1)*h
                rout[rec], states[rec] = re, x
                rec += 1
    return tout[:rec], rout[:rec], states[:rec], coordinate_maxima, error_peak, control_peak


def output_marks(step, offsets):
    """Require positive, knot-aligned steps before entering the RK4 loops."""
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Integration step must be finite and positive.")
    marks = np.rint(np.asarray(offsets)/step).astype(np.int64)
    if np.any(marks <= 0) or not np.allclose(marks*step, offsets, rtol=0, atol=1e-12):
        raise ValueError("Integration step must divide all output offsets.")
    return marks


def feedback_test(step=1e-5, stop_time=None) -> dict[str, np.ndarray]:
    r0 = DATA['reference_reference']
    t0 = verify_reference(DATA['reference_t'],r0)
    if not np.allclose(np.diff(t0), .02, rtol=0, atol=1e-11):
        raise ValueError("The reference record must use 0.02-s knots.")
    stop = float(DATA['simulation_stop']) if stop_time is None else float(stop_time)
    index = int(np.searchsorted(t0, stop))
    if index >= len(t0) or abs(t0[index]-stop) > 1e-10:
        raise ValueError("The simulation stop must coincide with a supplied reference knot.")
    tk, rk = t0[:index+1], r0[:index+1]
    t0, r0 = tk, rk
    ak, bk, ck, dk = (DATA['controller_'+k] for k in ('A','B','C','D'))
    aw, bw = DATA['controller_A_WP'], DATA['controller_B_WP']
    au = DATA.get('controller_A_WU', np.zeros((0, 0)))
    bu = DATA.get('controller_B_WU', np.zeros((0, 1)))
    cu = DATA.get('controller_C_WU', np.zeros((1, 0)))
    du = DATA.get('controller_D_WU', np.asarray([[float(DATA.get('controller_WU', 0.0))]]))
    offsets = np.array([.0001,.0002,.0005,.001,.0015,.002,.003,.005,.0075,.01,.015,.02])
    marks = output_marks(step, offsets)
    initial = np.zeros(4+len(ak)+len(aw)+len(au))
    initial[3] = np.nextafter(.002, 0.)
    t, r, xx, coordinate_maxima, peak_error, peak_control = _closed_loop(
        tk, rk, ak, bk.ravel(), ck.ravel(), dk.item(), aw, bw.ravel(), au, bu.ravel(), step, marks, initial)
    if not np.isfinite(xx).all() or np.any(coordinate_maxima >= BOX):
        raise RuntimeError("The feedback trajectory is nonfinite or outside the region.")
    xi, xk = xx[:,:4], xx[:,4:4+len(ak)]
    xw = xx[:,4+len(ak):4+len(ak)+len(aw)]
    xu = xx[:,4+len(ak)+len(aw):]
    e = r-xi[:,2]
    ua = xk @ ck.ravel()+dk.item()*e
    zeta, ga = lift(xi), drift(xi)
    rr = ga-zeta @ A[2:].T
    wc = np.linalg.solve(RD, (rr @ U).T).T
    chi = np.column_stack((zeta,xw,xu,xk))
    derivative = np.column_stack((xi[:,1], -10000*xi[:,0]-200*xi[:,1]+10000*ua,
        ga, xw @ aw.T+e[:,None]*bw.ravel(), xu @ au.T+ua[:,None]*bu.ravel(),
        xk @ ak.T+e[:,None]*bk.ravel()))
    transform = DATA['certificate_analysis_transform']
    xt, dx = chi @ transform.T, derivative @ transform.T
    p = DATA['certificate_P']
    V = np.einsum('ij,ij->i', xt @ p, xt)
    dV = 2*np.einsum('ij,ij->i', xt @ p, dx)
    w = np.column_stack((wc, r/DATA['certificate_reference_cap']))
    z = xw @ DATA['controller_C_WP'].ravel()+DATA['controller_D_WP'].item()*e
    epsilon = DATA['certificate_epsilon'].item()
    gamma2 = DATA['certificate_gamma2'].item()
    a = DATA['certificate_a'].item()
    zu = xu @ cu.ravel()+float(du.item())*ua
    lhs = dV+z*z+zu*zu+epsilon*np.sum(xt*xt,axis=1)-gamma2*np.sum(w*w,axis=1)
    V_bound = V[0]*np.exp(-a*t)+2*gamma2/a*(-np.expm1(-a*t))
    kappa = np.linalg.norm(DATA['certificate_Ht'], 2)/np.sqrt(DATA['certificate_M'].item())
    state_norm = np.linalg.norm(xi, axis=1)
    state_bound = kappa*np.sqrt(V_bound)
    if np.any(state_norm > state_bound) or np.max(lhs)>1e-6:
        raise RuntimeError("A displayed inequality check failed.")
    idx = np.searchsorted(t,t0)
    idx = np.minimum(idx,len(t)-1)
    prev = np.maximum(idx-1,0)
    idx = np.where(abs(t[prev]-t0)<abs(t[idx]-t0),prev,idx)
    rmse = np.sqrt(np.mean(e[idx]**2))
    return dict(t=t, xi=xi, reference=r, control=ua, weighted_control=zu,
                error=e, V=V, dissipation=lhs,
                state_norm=state_norm, state_bound=state_bound,
                coordinate_maxima=coordinate_maxima,
                rmse=np.array(rmse), peak_error=np.array(peak_error), peak_control=np.array(peak_control))

def _group(prefix):
    return {key[len(prefix)+1:]: value for key,value in DATA.items()
            if key.startswith(prefix+'_')}

def _set_feedback_realization(controller, full):
    # Store the controller and its matching certificate together.
    for key in list(DATA):
        if key.startswith(('controller_', 'certificate_')):
            del DATA[key]
    DATA.update({'controller_'+key: value for key,value in controller.items()})
    DATA.update({'certificate_'+key: np.asarray(value) for key,value in full.items()})


def main():
    global A, B, C
    start = time.perf_counter()
    model, weight, auxiliary = _group('plant'), _group('weight'), _group('auxiliary')
    print('1/8  MDI-TP Galerkin construction and independent tensor-rule check', flush=True)
    source_A = model['A'].copy()
    mdi = galerkin_model(model['bounds'], degree=3, orders=(4,4,64,12))
    for key in ('A','B','C','E'):
        model[key] = np.asarray(mdi[key]).ravel() if key == 'B' else np.asarray(mdi[key])
        DATA['plant_'+key] = model[key]
    A, B, C = (model[key] for key in ('A','B','C'))
    gram_error = verify_galerkin(model)
    mdi_report = dict(method='MDI-TP', metadata=mdi['metadata'],
                      independent_tensor_max_abs_error=gram_error,
                      supplied_matrix_max_abs_change=float(np.max(np.abs(A-source_A))),
                      matrix_sha256=hashlib.sha256(A.tobytes()).hexdigest())
    DATA['reference_t'] = verify_reference(DATA['reference_t'], DATA['reference_reference'])
    auxiliary['A'] = model['A'].copy()
    blend = float(auxiliary['blend'])
    base = np.r_[auxiliary['acceleration_base'],np.zeros(4)]
    auxiliary['A'][3] = (1-blend)*base+blend*model['A'][3]
    DATA['auxiliary_A'] = auxiliary['A']
    print('2/8  H-infinity controller synthesis', flush=True)
    controller, synthesis = design_controller(model, U @ RD, weight)
    print(f'     Controller order: {len(controller["A"])}; norm estimate: {synthesis["norm_estimate"]:.4f}', flush=True)
    print('3/8  Whole-region residual enclosures', flush=True)
    shape_check = verify_shape(model, RD)
    auxiliary_check = verify_auxiliary(auxiliary)
    print('4/8  Performance and full-state certificates', flush=True)
    full, perf, storage_summary = generate_storage(model, RD, controller)
    matrix_checks = verify_storage(model, RD, controller, full, perf)
    initial = verify_initial(full)
    print('5/8  All-time convolution and localization bounds', flush=True)
    localization = certify_localization(auxiliary, controller,
        progress=lambda name: print('     '+name, flush=True))
    DATA['coordinate_bounds'] = np.asarray(localization['physical_bounds'])
    _set_feedback_realization(controller, full)
    DATA['gamma_performance'] = perf['gamma']
    DATA['P_performance'] = perf['P']
    DATA['epsilon_performance'] = perf['epsilon']
    # Save the regenerated model for the remaining studies.
    np.savez_compressed(ROOT/'paper_data.npz', **DATA)
    print('6/8  Realization and 10-second residual experiments', flush=True)
    single = realization_test()
    family = residual_family()
    print('7/8  Continuous-time nonlinear feedback simulation', flush=True)
    feedback = feedback_test()
    print('8/8  Saving feedback results', flush=True)
    maximum = float(np.max(family['normalized'][family['active']]))
    summary = dict(
        integration=mdi_report, galerkin_recalculation_error=gram_error, synthesis=synthesis,
        shaped_residual=shape_check, auxiliary_residual=auxiliary_check,
        storage=storage_summary, matrix_checks=matrix_checks,
        coherent_initial=initial, localization=localization,
        experiments=dict(normalized_residual_maximum=maximum,
            realization_nominal_max_error=np.max(np.abs(single['nominal']-single['nonlinear']),axis=0).tolist(),
            realization_corrected_max_error=np.max(np.abs(single['corrected']-single['nonlinear']),axis=0).tolist(),
            residual_paths_retained=int(family['active'][-1].sum()),
            residual_path_count=int(family['active'].shape[1]),
            tracking_RMSE=float(feedback['rmse']), peak_tracking_error=float(feedback['peak_error']),
            tracking_RMSE_definition='sqrt(mean(e(t_k)^2)) on the uniform 0.02 s reference knots over [0,175] s',
            peak_physical_states=feedback['coordinate_maxima'].tolist(),
            peak_recorded_control=float(np.max(np.abs(feedback['control']))),
            peak_control=float(feedback['peak_control']),
            dissipation_maximum=float(np.max(feedback['dissipation']))),
        seconds=time.perf_counter()-start)
    arrays = {'controller_'+key: value for key,value in controller.items()}
    arrays.update({'plant_'+key: model[key] for key in ('A','B','C','E','bounds')})
    arrays.update({key:value for key,value in DATA.items() if key.startswith('certificate_')})
    arrays.update({key:value for key,value in DATA.items() if key.startswith('auxiliary_')})
    arrays.update(shape=U @ RD, coordinate_bounds=DATA['coordinate_bounds'],
                  P_performance=perf['P'], gamma_performance=np.asarray(perf['gamma']),
                  epsilon_performance=np.asarray(perf['epsilon']),
                  t=feedback['t'], xi=feedback['xi'], Vcl=feedback['V'],
                  reference=feedback['reference'], control=feedback['control'], error=feedback['error'],
                  weighted_control=feedback['weighted_control'],
                  dissipation=feedback['dissipation'], state_bound=feedback['state_bound'])
    np.savez_compressed(ROOT/'results.npz', **arrays)
    (ROOT/'results.json').write_text(json.dumps(summary,indent=2)+'\n', encoding='utf-8')
    print(f'Normalized residual: {maximum:.4f}; retained paths: {summary["experiments"]["residual_paths_retained"]}/289.', flush=True)
    print(f'Tracking RMSE: {feedback["rmse"]:.3g} rad; localization ratio: {localization["beta"]:.4f}.', flush=True)
    print(f'Feedback study completed in {time.perf_counter()-start:.0f} s.', flush=True)
    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--figures-only', action='store_true', help='Redraw from saved numerical results.')
    args = parser.parse_args()
    if not args.figures_only:
        main()
        from degree_comparison import run as compare_degree
        print('Nested-degree experiment', flush=True)
        compare_degree()
        print('Zero-initial sweep experiment', flush=True)
        subprocess.run([sys.executable, str(ROOT/'sweep_test.py'), '--code', str(ROOT),
                        '--output', str(ROOT)], check=True, cwd=ROOT)
    from plot_figures import run as draw
    draw(ROOT)
    from comparison import run as compare_feedback
    from plot_comparison import run as draw_comparison
    if not args.figures_only:
        print('H-infinity and backstepping comparison', flush=True)
        compare_feedback(ROOT)
    draw_comparison(ROOT)
    print('Study complete.', flush=True)
