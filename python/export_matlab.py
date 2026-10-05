from pathlib import Path
import argparse
import json
import numpy as np
from scipy.io import loadmat, savemat
from scipy.linalg import eigvals, solve_continuous_are


def cascade(left, right):
    a, b, c, d = (left[q] for q in 'ABCD')
    e, f, g, h = (right[q] for q in 'ABCD')
    return dict(A=np.block([[e, np.zeros((len(e), len(a)))], [b @ g, a]]),
                B=np.vstack((f, b @ h)), C=np.hstack((d @ g, c)), D=d @ h)


def response(system, omega):
    a, b, c, d = (system[q] for q in 'ABCD')
    return c @ np.linalg.solve(1j * omega * np.eye(len(a)) - a, b) + d


def export(paper_path, output, rho=0.682, controller_path=None):
    rho = float(rho)
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError('The reference cap must be finite and positive.')
    paper = np.load(paper_path, allow_pickle=False)
    a = paper['plant_A']
    b = paper['plant_B'].reshape(-1, 1)
    c = paper['plant_C'][:1]
    shape = paper['shape_U'] @ paper['shape_Draw']
    selected = None
    if controller_path is not None:
        selected = np.load(controller_path, allow_pickle=False)
        prefix = 'controller_' if 'controller_A' in selected.files else ''
        # Keep the model and certificate from the same realization.
        for key in ('plant_A', 'plant_B', 'plant_C', 'plant_E', 'plant_bounds'):
            if key in selected.files and not np.array_equal(selected[key], paper[key]):
                raise ValueError(f'The controller results use a different {key}; regenerate them together.')
        if 'shape' in selected.files and not np.array_equal(selected['shape'], shape):
            raise ValueError('The controller results use a different closure enclosure.')
        for key in selected.files:
            if key.startswith('auxiliary_') and key in paper.files:
                if not np.array_equal(selected[key], paper[key]):
                    raise ValueError('The cached results use a different auxiliary localization model.')
        if 'coordinate_bounds' in selected.files and 'coordinate_bounds' in paper.files:
            if not np.array_equal(selected['coordinate_bounds'],paper['coordinate_bounds']):
                raise ValueError('The cached localization bounds differ from the current model.')
        for weight in ('WP', 'WU'):
            for q in 'ABCD':
                key = 'controller_' + q + '_' + weight
                paper_key = 'weight_' + q + '_' + weight
                if key in selected.files and paper_key in paper.files:
                    if not np.array_equal(selected[key], paper[paper_key]):
                        raise ValueError(f'The controller results use a different {weight} realization.')
        if 'certificate_reference_cap' in selected.files:
            if rho != float(selected['certificate_reference_cap']):
                raise ValueError('The reference cap differs from the cached certificate; regenerate the certificate.')
        ak, bk, ck, dk = (selected[prefix + q] for q in 'ABCD')
        nk = len(ak)
        if (ak.shape != (nk, nk) or bk.shape != (nk, 1)
                or ck.shape != (1, nk) or dk.shape != (1, 1)
                or not all(np.all(np.isfinite(value)) for value in (ak, bk, ck, dk))):
            raise ValueError('The controller must have a finite SISO state-space realization.')
    bd = paper['plant_E'] @ shape
    g = dict(A=a, B=b, C=c, D=np.zeros((1, 1)))
    wp = {q: paper[f'weight_{q}_WP'] for q in 'ABCD'}
    if 'weight_A_WU' in paper.files:
        wu = {q: paper[f'weight_{q}_WU'] for q in 'ABCD'}
    else:
        wu = dict(A=np.zeros((0, 0)), B=np.zeros((0, 1)), C=np.zeros((1, 0)),
                  D=np.asarray([[float(paper['weight_WU'])]]))
    pd = dict(A=a, B=bd, C=c, D=np.zeros((1, bd.shape[1])))
    if not (np.array_equal(a[:2, 2:], np.zeros_like(a[:2, 2:]))
            and np.array_equal(bd[:2], np.zeros_like(bd[:2]))):
        raise ValueError('The supplied model does not have the expected actuator structure.')
    af, bf, cf = a[2:, 2:], bd[2:], c[:, 2:]
    r = np.array([[rho ** 2]])
    x = solve_continuous_are(af.T, cf.T, bf @ bf.T, r)
    gain = x @ cf.T / rho ** 2
    factor = dict(A=af, B=gain * rho, C=cf, D=np.array([[rho]]))
    w1 = cascade(wp, factor)
    w2 = cascade(wu, factor)
    n, nf, nu, nd = len(a), len(wp['A']), len(wu['A']), bd.shape[1]
    ng = n+nf+nu
    ap = np.zeros((ng, ng))
    ap[:n, :n] = a
    ap[n:n+nf, :n] = -wp['B'] @ c
    ap[n:n+nf, n:n+nf] = wp['A']
    ap[n+nf:, n+nf:] = wu['A']
    bp = np.zeros((ng, nd + 2))
    bp[:n, :nd] = bd
    bp[n:n+nf, nd:nd + 1] = wp['B'] * rho
    bp[:n, -1:] = b
    bp[n+nf:, -1:] = wu['B']
    cp = np.zeros((3, ng))
    cp[:1, :n] = -wp['D'] @ c
    cp[:1, n:n+nf] = wp['C']
    cp[1:2, n+nf:] = wu['C']
    cp[2:, :n] = -c
    dp = np.zeros((3, nd + 2))
    dp[:1, nd:nd + 1] = wp['D'] * rho
    dp[1, -1] = wu['D'].item()
    dp[2, nd] = rho
    full = dict(A=ap, B=bp, C=cp, D=dp)
    payload = dict(G_data=g, WP_data=wp, WU_data=wu,
        Pd_data=pd, F_data=factor, W1eq_data=w1, W2eq_data=w2, P_data=full,
        reference_cap=rho, WU_DC=response(wu, 0).item().real,
        WU_high_frequency=wu['D'].item(), D_shape=shape,
        WP_num=paper['weight_WP_num'], WP_den=paper['weight_WP_den'],
        P_input_order=np.array([f'wc{k + 1}' for k in range(nd)] + ['r/reference_cap', 'ua'], dtype=object),
        P_output_order=np.array(['WP*(r-y)', 'WU*ua', 'r-y'], dtype=object),
        controller_input='e=r-y', controller_output='ua', sample_time=0.0)
    payload.update({key:paper[key] for key in paper.files if key.startswith('auxiliary_')})
    payload['stop_time'] = float(paper['simulation_stop'])
    payload['reference_t'] = paper['reference_t'][:, None]
    payload['reference_r'] = paper['reference_reference'][:, None]
    payload['xi0'] = np.array([[0.0], [0.0], [0.0], [np.nextafter(.002, 0.)]])
    payload['xW0'] = np.zeros((nf+nu, 1))
    payload['xWP0'] = np.zeros((nf, 1))
    payload['xWU0'] = np.zeros((nu, 1))
    for key in ('WU_num', 'WU_den'):
        if 'weight_'+key in paper.files:
            payload[key] = paper['weight_'+key]
    if controller_path is not None:
        payload['K_data'] = {q: selected[prefix + q] for q in 'ABCD'}
        payload.update({q + 'K': selected[prefix + q] for q in 'ABCD'})
        payload.update({key: selected[key] for key in selected.files if key.startswith('certificate_')})
        for key in ('coordinate_bounds', 'P_performance', 'gamma_performance', 'epsilon_performance'):
            if key in selected.files:
                payload[key] = selected[key]
        payload['xK0'] = np.zeros((len(payload['AK']), 1))
    sweep_file = Path(paper_path).with_name('sweep_results.npz')
    if sweep_file.exists() and 'xK0' in payload:
        sweep = np.load(sweep_file)
        payload['sweep_t'] = sweep['reference_t'][:, None]
        payload['sweep_r'] = sweep['reference_r'][:, None]
        payload['sweep_stop_time'] = float(sweep['reference_t'][-1])
        payload['sweep_xi0'] = np.zeros((4, 1))
        payload['sweep_xK0'] = np.zeros_like(payload['xK0'])
        payload['sweep_xW0'] = np.zeros_like(payload['xW0'])
        payload['sweep_measured_gain'] = float(sweep['gain'][-1])
    errors, objective_errors = [], []
    frequencies = np.r_[0, np.geomspace(1e-8, 1e8, 2000)]
    for omega in frequencies:
        pdw, fw = response(pd, omega), response(factor, omega).item()
        spectrum = rho ** 2 + np.sum(abs(pdw) ** 2)
        errors.append(abs(abs(fw) ** 2 - spectrum) / spectrum)
        if 'K_data' in payload:
            gw = response(g, omega).item()
            kw = response(payload['K_data'], omega).item()
            sw = 1 / (1 + gw * kw)
            column = np.array([[response(wp, omega).item() * sw], [response(wu, omega).item() * kw * sw]])
            full_transfer = column @ np.hstack((-pdw, [[rho]]))
            mixed_transfer = np.array([[response(w1, omega).item() * sw],
                                       [response(w2, omega).item() * kw * sw]])
            gain_full = np.linalg.svd(full_transfer, compute_uv=False)[0]
            objective_errors.append(abs(gain_full - np.linalg.norm(mixed_transfer)) / max(gain_full, 1e-300))
    riccati_terms = [af @ x + x @ af.T, bf @ bf.T, -gain @ cf @ x]
    riccati_residual = np.linalg.norm(sum(riccati_terms)) / sum(np.linalg.norm(q) for q in riccati_terms)
    report = dict(spectral_identity_max_relative_error=float(max(errors)),
                  riccati_relative_residual=float(riccati_residual),
                  factor_pole_abscissa=float(max(eigvals(af).real)),
                  factor_zero_abscissa=float(max(eigvals(af - gain @ cf).real)),
                  factor_order=len(af), equivalent_W1_order=len(w1['A']), equivalent_W2_order=len(w2['A']),
                  full_generalized_plant_order=len(ap), frequency_samples=len(frequencies),
                  matlab_executed=False, reference_cap=rho,
                  WU_DC=response(wu, 0).item().real, WU_high_frequency=wu['D'].item())
    if objective_errors:
        report['closed_loop_singular_value_max_relative_error'] = float(max(objective_errors))
        if max(objective_errors) > 1e-9:
            raise RuntimeError('The exported mixed-sensitivity objective failed its frequency check.')
    if max(errors) > 1e-10 or report['factor_zero_abscissa'] >= 0 or report['factor_pole_abscissa'] >= 0:
        raise RuntimeError('Spectral factor verification failed.')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    savemat(output, payload, do_compression=True, long_field_names=True)
    saved = loadmat(output, struct_as_record=False, squeeze_me=False)
    for name, data in payload.items():
        if isinstance(data, dict):
            for key, value in data.items():
                if not np.array_equal(value, getattr(saved[name][0, 0], key)):
                    raise RuntimeError(f'MAT round-trip mismatch: {name}.{key}')
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument('paper', type=Path, nargs='?', default=root/'paper_data.npz')
    parser.add_argument('output', type=Path, nargs='?', default=root/'Galerkin_Control.mat')
    parser.add_argument('--controller', type=Path, default=root/'results.npz')
    parser.add_argument('--rho', type=float, default=0.682)
    args = parser.parse_args()
    print(json.dumps(export(args.paper, args.output, args.rho, args.controller), indent=2))
