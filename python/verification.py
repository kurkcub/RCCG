from __future__ import annotations

from fractions import Fraction as F
from functools import lru_cache
from math import lcm
import time
import numpy as np
from scipy.linalg import eigvals, matrix_balance, solve_continuous_are, solve_continuous_lyapunov
from mpmath import iv
from design import norm_grid

def _require(condition, message):
    # Certificate checks must also run when Python is invoked with -O.
    if not condition:
        raise ValueError(message)

def _mp_import():
    import mpmath as mp
    mp.mp.dps = 65
    return mp

def _whiten(A, B, Qcoef, alpha, eta=1e-10):
    mp = _mp_import()
    mm = lambda x: mp.matrix(np.asarray(x, float).tolist())
    n = len(A)
    As, T = matrix_balance(A, permute=False)
    Bs = np.linalg.solve(T, B)
    am = mm(As) + mp.mpf(float(alpha)) / 2 * mp.eye(n)
    bm = mm(Bs)
    qm = bm * mm(Qcoef) * bm.T + mp.mpf(eta) * mp.eye(n)
    S = solve_continuous_lyapunov(As + alpha / 2 * np.eye(n), -np.array(qm.tolist(), float))
    sm = mm((S + S.T) / 2)
    for it in range(60):
        rr = am * sm + sm * am.T + qm
        nr = max(map(abs, rr))
        if nr < mp.mpf('1e-38'):
            break
        ds = solve_continuous_lyapunov(As + alpha / 2 * np.eye(n), -np.array(rr.tolist(), float))
        sm += mm((ds + ds.T) / 2)
        sm = (sm + sm.T) / 2
    lm = mp.cholesky(sm)
    tf = lm ** (-1) * mm(np.linalg.inv(T))
    return (np.asarray(tf.tolist(), float), float(nr))

def _strict_margin(mp, S):
    sf = np.asarray(S.tolist(), float)
    margin = -float(np.linalg.eigvalsh((sf + sf.T) / 2).max()) * 0.2
    if not np.isfinite(margin) or margin <= 0:
        margin = 1e-12
    for _ in range(20):
        try:
            mp.cholesky(-S - mp.mpf(margin) * mp.eye(S.rows))
            return float(margin)
        except ValueError:
            margin *= 0.1
    raise ValueError('Strict Schur complement check failed')

def _control_weight(k):
    if 'A_WU' in k:
        a, b, c, d = [np.asarray(k[q], float) for q in ('A_WU', 'B_WU', 'C_WU', 'D_WU')]
    else:
        a, b, c, d = np.zeros((0, 0)), np.zeros((0, 1)), np.zeros((1, 0)), np.array([[float(k.get('WU', 0.0))]])
    nu = len(a)
    if a.shape != (nu, nu) or b.shape != (nu, 1) or c.shape != (1, nu) or d.shape != (1, 1):
        raise ValueError('WU must have a SISO state-space realization')
    return a, b, c, d

def _state_order(k):
    return f"zeta(8),WP({len(k['A_WP'])}),WU({len(_control_weight(k)[0])}),K({len(k['A'])})"

def canonical_mpf(m, sh, k, mp):
    mm = lambda x: mp.matrix(np.asarray(x, float).tolist())
    fullshape = np.zeros((6, 4))
    fullshape[np.asarray(m['nonnull'], int)] = sh
    ap, bp, cy = (mm(m['A']), mm(m['B'][:, None]), mm(m['C'][:1]))
    e = mm(m['E'])
    ds = mm(fullshape)
    ak, bk, ck, dk = [mm(k[q]) for q in ('A', 'B', 'C', 'D')]
    aw, bw, cw, dw = [mm(k[q]) for q in ('A_WP', 'B_WP', 'C_WP', 'D_WP')]
    au, bu, cu, du = _control_weight(k)
    nz, nf, nuw, nk = (8, len(k['A_WP']), len(au), len(k['A']))
    iu = nz + nf
    ng = iu + nuw
    n = ng + nk
    nu = 5
    cap = mp.mpf(float(0.682))
    A = mp.zeros(n)
    B = mp.zeros(n, nu)
    weighted_u = nuw > 0 or np.any(du != 0)
    C = mp.zeros(2 if weighted_u else 1, n)
    D = mp.zeros(2 if weighted_u else 1, nu)
    H = mp.zeros(4, n)
    A[:nz, :nz] = ap - bp * dk * cy
    A[:nz, ng:n] = bp * ck
    A[ng:n, :nz] = -bk * cy
    A[ng:n, ng:n] = ak
    A[nz:iu, :nz] = -bw * cy
    A[nz:iu, nz:iu] = aw
    B[:nz, :4] = e * ds
    B[:nz, 4:5] = bp * dk * cap
    B[ng:n, 4:5] = bk * cap
    B[nz:iu, 4:5] = bw * cap
    C[0:1, :nz] = -dw * cy
    C[0:1, nz:iu] = cw
    D[0:1, 4:5] = dw * cap
    if weighted_u:
        dum = mm(du)
        C[1:2, :nz] = -dum * dk * cy
        C[1:2, ng:n] = dum * ck
        D[1:2, 4:5] = dum * dk * cap
    if nuw:
        aum, bum, cum = map(mm, (au, bu, cu))
        A[iu:ng, :nz] = -bum * dk * cy
        A[iu:ng, iu:ng] = aum
        A[iu:ng, ng:n] = bum * ck
        B[iu:ng, 4:5] = bum * dk * cap
        C[1:2, iu:ng] = cum
    H[0, 0] = H[1, 1] = 1
    H[2:4, :nz] = mm(m['C'])
    return (m, k, fullshape, A, B, C, D, H)

def arr(x):
    return np.asarray(x.tolist(), float)

def generate_storage(model, shape_matrix, controller):
    mp = _mp_import()
    mm = lambda x: mp.matrix(np.asarray(x, float).tolist())
    m, k, shape, A, B, C, D, H = canonical_mpf(model, shape_matrix, controller, mp)
    a, b, c, d, h = map(arr, (A, B, C, D, H))
    n = len(a)
    nu = b.shape[1]
    peak, ww, gg = norm_grid((a, b, c, d), nf=1800)
    absc = float(max(eigvals(a).real))
    alphaF = -2 * absc * 0.8
    decay = alphaF * 0.4
    tf, residual = _whiten(a, b, 2 / alphaF * np.eye(nu), alphaF)
    tfm = mm(tf)
    ti = tfm ** (-1)
    at = tfm * A * ti
    bt = tfm * B
    ct = C * ti
    ht = H * ti
    aa, bb, cc, hh = map(arr, (at, bt, ct, ht))
    chosen = None
    for M in np.geomspace(0.001, 100000000000000.0, 69):
        eps = decay * M
        g2 = alphaF * M
        R = g2 * np.eye(nu) - d.T @ d
        if np.linalg.eigvalsh(R)[0] <= 0:
            continue
        cross = M * bb + cc.T @ d
        Q = M * (aa + aa.T) + cc.T @ cc + eps * np.eye(n) + cross @ np.linalg.solve(R, cross.T)
        if max(np.linalg.eigvalsh((Q + Q.T) / 2)) < -max(1e-08, eps * 1e-06):
            chosen = (float(M), float(eps), float(g2))
            break
    if chosen is None:
        raise RuntimeError('Coercive full storage search failed')
    M, eps, g2 = chosen
    P = mp.mpf(M) * mp.eye(n)
    R = mp.mpf(g2) * mp.eye(nu) - D.T * D
    mp.cholesky(R)
    cross = P * bt + ct.T * D
    Q = at.T * P + P * at + ct.T * ct + mp.mpf(eps) * mp.eye(n) + cross * R ** (-1) * cross.T
    margin = _strict_margin(mp, Q)
    kapxi = float(np.linalg.norm(hh, 2) / np.sqrt(M))
    kape = float(np.linalg.norm(hh[2]) / np.sqrt(M))
    ultimate = 2 * g2 / decay
    perm = np.array([4, 0, 1, 2, 3])
    full = dict(Acanonical=a, Bcanonical=b, Ccanonical=c, Dcanonical=d, Hcanonical=h, At=aa, Bt=bb, Ct=cc, Dt=d, Ht=hh, P=M * np.eye(n), Pcanonical=M * tf.T @ tf, M=M, epsilon=eps, gamma2=g2, a=decay, analysis_transform=tf, analysis_inverse=arr(ti), closure_shape=shape, reference_cap=0.682, state_order=_state_order(k), input_order='wc(4),r/0.682', paper_input_permutation=perm, Bpaper=b[:, perm], Dpaper=d[:, perm], output='col(WP(r-y), WU(ua))', frequency=ww, frequency_gain=gg)
    storage = dict(dimension=n, Pmin=M, Pmax=M, epsilon=eps, gamma_analysis=float(np.sqrt(g2)), a=decay, strict_Schur_margin=margin, lyap_refinement_residual=residual, kappa_xi=kapxi, kappa_e=kape, mu_squared=2.0, De_norm=0.682, ultimate_V=ultimate, ultimate_state_bound=kapxi * np.sqrt(ultimate), ultimate_error_bound=kape * np.sqrt(ultimate) + 0.682 * np.sqrt(2.0), channelwise_tighter_ultimate_error_bound=kape * np.sqrt(ultimate) + 0.682)
    perf = None
    levels = [0.24, 0.25, 0.27, 0.3, 0.35, 0.5, 1.0, 1.5, 1.65, 2.5]
    levels += [float(np.ceil(10 * peak[0] * factor) / 10) for factor in (1.05, 1.1, 1.25, 1.5, 2)]
    for gamma in sorted(set(levels)):
        if gamma <= peak[0]:
            continue
        Rn = gamma ** 2 * np.eye(nu) - d.T @ d
        for epsilon in (1e-05, 1e-07, 1e-09, 1e-11):
            try:
                ash = aa + bb @ np.linalg.solve(Rn, d.T @ cc)
                q = cc.T @ cc + cc.T @ d @ np.linalg.solve(Rn, d.T @ cc) + epsilon * np.eye(n)
                pp = solve_continuous_are(ash, bb, q, -Rn, balanced=True)
                pp = (pp + pp.T) / 2
                PP = mm(pp)
                mp.cholesky(PP)
                RR = mp.mpf(gamma) ** 2 * mp.eye(nu) - D.T * D
                mp.cholesky(RR)
                nn = PP * bt + ct.T * D
                ss = at.T * PP + PP * at + ct.T * ct + mp.mpf(epsilon / 2) * mp.eye(n) + nn * RR ** (-1) * nn.T
                pmargin = _strict_margin(mp, ss)
                perf = dict(gamma=gamma, epsilon=epsilon / 2, strict_Schur_margin=pmargin, Pmin=float(min(np.linalg.eigvalsh(pp))), Pmax=float(max(np.linalg.eigvalsh(pp))), verified=True)
                perf_data = dict(P=pp, gamma=gamma, epsilon=epsilon / 2, analysis_transform=tf, strict_margin=pmargin)
                break
            except (ValueError, np.linalg.LinAlgError):
                pass
        if perf:
            break
    if perf is None:
        raise RuntimeError('No strict performance witness found')
    return full, perf_data, dict(norm_estimate=float(peak[0]),
        peak_frequency=float(peak[1]), closed_loop_abscissa=absc,
        storage=storage, performance=perf)

def M(a):
    a = np.atleast_2d(np.asarray(a, float))
    return [[F(float(x)) for x in row] for row in a]

def zero(n, m):
    return [[F(0) for _ in range(m)] for _ in range(n)]

def eye(n):
    return [[F(i == j) for j in range(n)] for i in range(n)]

def tr(a):
    return [list(v) for v in zip(*a)]

def add(*args):
    return [[sum((a[i][j] for a in args), F(0)) for j in range(len(args[0][0]))] for i in range(len(args[0]))]

def scale(a, s):
    return [[s * x for x in row] for row in a]

def mul(a, b):
    bt = tr(b)
    return [[sum((x * y for x, y in zip(row, col) if x and y), F(0)) for col in bt] for row in a]

def blockput(a, b, i, j):
    for ii, row in enumerate(b):
        for jj, v in enumerate(row):
            a[i + ii][j + jj] = v

def canonical_fraction(md, shape, k):
    ds = np.zeros((6, 4))
    ds[np.asarray(md['nonnull'], int)] = shape
    ap, bp, cy, e, dshape = (M(md['A']), M(md['B'][:, None]), M(md['C'][:1]), M(md['E']), M(ds))
    ak, bk, ck, dk = [M(k[q]) for q in ('A', 'B', 'C', 'D')]
    aw, bw, cw, dw = [M(k[q]) for q in ('A_WP', 'B_WP', 'C_WP', 'D_WP')]
    au, bu, cu, du = _control_weight(k)
    nz, nf, nuw = 8, len(k['A_WP']), len(au)
    iu = nz + nf
    ng = iu + nuw
    n = ng + len(k['A'])
    a = zero(n, n)
    b = zero(n, 5)
    weighted_u = nuw > 0 or np.any(du != 0)
    c = zero(2 if weighted_u else 1, n)
    d = zero(2 if weighted_u else 1, 5)
    cap = F(float(0.682))
    blockput(a, add(ap, scale(mul(mul(bp, dk), cy), -1)), 0, 0)
    blockput(a, mul(bp, ck), 0, ng)
    blockput(a, scale(mul(bk, cy), -1), ng, 0)
    blockput(a, ak, ng, ng)
    blockput(a, scale(mul(bw, cy), -1), 8, 0)
    blockput(a, aw, 8, 8)
    blockput(b, mul(e, dshape), 0, 0)
    blockput(b, scale(mul(bp, dk), cap), 0, 4)
    blockput(b, scale(bk, cap), ng, 4)
    blockput(b, scale(bw, cap), 8, 4)
    blockput(c, scale(mul(dw, cy), -1), 0, 0)
    blockput(c, cw, 0, 8)
    blockput(d, scale(dw, cap), 0, 4)
    if weighted_u:
        dum = M(du)
        blockput(c, scale(mul(mul(dum, dk), cy), -1), 1, 0)
        blockput(c, mul(dum, ck), 1, ng)
        blockput(d, scale(mul(dum, dk), cap), 1, 4)
    if nuw:
        aum, bum, cum = map(M, (au, bu, cu))
        blockput(a, scale(mul(mul(bum, dk), cy), -1), iu, 0)
        blockput(a, aum, iu, iu)
        blockput(a, mul(bum, ck), iu, ng)
        blockput(b, scale(mul(bum, dk), cap), iu, 4)
        blockput(c, cum, 1, iu)
    return (a, b, c, d)

def exact_ldl_pd(a, label):
    n = len(a)
    _require(all((a[i][j] == a[j][i] for i in range(n) for j in range(n))), label + ' asymmetric')
    den = lcm(*(x.denominator for row in a for x in row))
    z = [[x.numerator * (den // x.denominator) for x in row] for row in a]
    prev = 1
    pivots = []
    bits = []
    for k in range(n):
        pivot = z[k][k]
        if pivot <= 0:
            raise ArithmeticError(f'{label}: nonpositive leading minor {k}: {pivot}')
        pivots.append(F(pivot, prev * den))
        bits.append(pivot.bit_length())
        for i in range(k + 1, n):
            for j in range(i, n):
                numerator = pivot * z[i][j] - z[i][k] * z[k][j]
                quotient, remainder = divmod(numerator, prev)
                if remainder:
                    raise ArithmeticError(label + ': Bareiss remainder')
                z[i][j] = quotient
                z[j][i] = quotient
        for i in range(k + 1, n):
            z[i][k] = z[k][i] = 0
        prev = pivot
    lo = min(pivots)
    f = float(lo)
    lower = float(np.nextafter(f, -np.inf))
    _require(F(lower) < lo, 'verification.py: certificate check failed at F(lower) < lo')
    return dict(dimension=n, positive_definite=True, minimum_pivot_lower=lower)

def verify_storage(model, shape, controller, full, perf):
    """Check the saved floating-point realization as exact binary rationals.

    High-precision searches only propose witnesses. The Fraction/Bareiss
    checks below certify the resulting strict inequalities independently.
    """
    st = time.time()
    a, b, c, d = canonical_fraction(model, shape, controller)
    if not np.array_equal(full['analysis_transform'], perf['analysis_transform']):
        raise ValueError('Storage and performance witnesses use different analysis coordinates')
    t = M(full['analysis_transform'])
    ttt = mul(tr(t), t)
    n = len(a)
    report = dict(transform_gram=exact_ldl_pd(ttt, 'T.T*T'))
    for name, pt, eps, g2 in [('coercive', M(full['P']), F(float(full['epsilon'])), F(float(full['gamma2']))), ('performance', M(perf['P']), F(float(perf['epsilon'])), F(str(float(perf['gamma']))) ** 2)]:
        _require(eps > 0 and g2 > 0, name + ': epsilon and gamma squared must be positive')
        started = time.time()
        p = mul(mul(tr(t), pt), t)
        upper = add(mul(tr(a), p), mul(p, a), mul(tr(c), c), scale(ttt, eps))
        cross = add(mul(p, b), mul(tr(c), d))
        lower = add(mul(tr(d), d), scale(eye(5), -g2))
        L = zero(n + 5, n + 5)
        blockput(L, upper, 0, 0)
        blockput(L, cross, 0, n)
        blockput(L, tr(cross), n, 0)
        blockput(L, lower, n, n)
        out = dict(P_analysis=exact_ldl_pd(pt, name + ':Ptf'), negative_LMI=exact_ldl_pd(scale(L, -1), name + ':-L'), gamma_squared_exact=str(g2), epsilon_exact=str(eps), seconds=time.time() - started)
        report[name] = out
    report['readout_bounds'] = certified_readout_bounds(full)
    report.update(total_seconds=time.time() - st, all_checks_passed=True, full_closed_loop_dimension=n, external_input_dimension=5, state_order=_state_order(controller), input_order='wc4,r/.682', coercivity_metric_in_canonical_coordinates='epsilon*T.T*T')
    return report

def certified_readout_bounds(full):
    """Outward bounds for the scalar analysis-coordinate storage witness.

    The displayed spectral norms in generate_storage are numerical estimates.
    Here exact rational products and an inverse-defect bound certify a slightly
    larger Frobenius upper bound without assuming the stored inverse is exact.
    """
    t = np.asarray(full['analysis_transform'], float)
    ti = np.asarray(full['analysis_inverse'], float)
    p = np.asarray(full['P'], float)
    mass = F(float(p[0, 0]))
    _require(mass > 0 and np.array_equal(p, np.eye(len(p)) * p[0, 0]),
             'Readout bounds require the positive scalar-identity storage witness')
    # T^-1 = Ti (I - (I - T Ti))^-1, so its right inverse defect is relevant.
    eta = F(float(inverse_defect(ti, t)))
    _require(eta < 1, 'The analysis-coordinate inverse defect is not contractive')
    approximate_readout = mul(M(full['Hcanonical']), M(ti))
    xi_square = sum((x * x for row in approximate_readout for x in row), F(0))
    error_square = sum((x * x for x in approximate_readout[2]), F(0))
    xi_square /= (1 - eta) ** 2 * mass
    error_square /= (1 - eta) ** 2 * mass
    decay = F(float(full['epsilon'])) / mass
    _require(decay > 0, 'The certified decay rate must be positive')
    forced = 2 * F(float(full['gamma2'])) / decay

    def rational_interval(x):
        return iv.mpf(x.numerator) / x.denominator

    def upper_interval(x):
        return float(np.nextafter(float(x.b), np.inf))

    state_upper = upper_interval(iv.sqrt(rational_interval(xi_square * forced)))
    error_upper = upper_interval(iv.sqrt(rational_interval(error_square * forced))
        + rational_interval(F(float(full['reference_cap']))) * iv.sqrt(iv.mpf(2)))
    return dict(verified=True, analysis_inverse_defect_upper=float(eta),
        norm_method='exact rational Frobenius bound with inverse-defect correction',
        kappa_xi_upper=upper_interval(iv.sqrt(rational_interval(xi_square))),
        kappa_e_upper=upper_interval(iv.sqrt(rational_interval(error_square))),
        m_w_squared_exact='2', ultimate_state_bound_upper=state_upper,
        ultimate_error_bound_upper=error_upper)

def verify_initial(d):
    T = d['analysis_transform']
    P = d['P']
    n = len(P)
    _require(np.array_equal(P, np.eye(n) * P[0, 0]), 'verification.py: certificate check failed at np.array_equal(P, np.eye(n) * P[0, 0])')
    M = F(float(P[0, 0]))
    theta = F(1, 10000000)
    velocity = F(1, 500)
    raw = {2: {(1, 0): F(1)}, 3: {(0, 1): F(1)}, 4: {(0, 3): F(4), (0, 1): F(-6)}, 5: {(1, 2): F(4), (1, 0): F(-2)}, 6: {(2, 1): F(4), (0, 1): F(-2)}, 7: {(3, 0): F(4), (1, 0): F(-6)}}
    c0 = F(0)
    for i in range(n):
        coefficients = {}
        for j, terms in raw.items():
            for powers, value in terms.items():
                coefficients[powers] = coefficients.get(powers, F(0)) + F(float(T[i, j])) * value
        bound = sum((abs(value) * theta ** powers[0] * velocity ** powers[1] for powers, value in coefficients.items()), F(0))
        c0 += M * bound * bound

    def upper(x):
        result = float(np.nextafter(float(x), np.inf))
        _require(F(result) > x, 'verification.py: certificate check failed at F(result) > x')
        return result
    a = F(float(d['epsilon'])) / M
    a_lower = float(np.nextafter(float(a), -np.inf))
    _require(F(a_lower) < a, 'verification.py: certificate check failed at F(a_lower) < a')
    forced = 2 * F(float(d['gamma2'])) / a
    cN = max(c0, forced)
    return dict(Vcl_initial_upper=upper(c0), decay_rate_lower=a_lower,
                forced_storage_level_upper=upper(forced), c_N_upper=upper(cN))


iv.dps = 40
UROUND = np.finfo(float).eps

def set_region_model(md):
    global model, IDS, SCALES, A, T
    model = md
    IDS = [tuple(v) for v in md['indices']]
    SCALES = md['scales']
    A, T = md['A'], md['T']
    bounds = np.asarray(md['bounds'], float)
    _require(bounds.shape == (4,) and np.all(np.isfinite(bounds))
             and np.all(bounds > 0) and bounds[2] <= 1.5,
             'The interval trigonometric enclosure requires positive finite box widths and |theta| <= 1.5')

def down(x):
    return np.nextafter(np.asarray(x, dtype=float), -np.inf)

def up(x):
    return np.nextafter(np.asarray(x, dtype=float), np.inf)

class I:

    def __init__(self, l, h=None):
        self.l = np.asarray(l, dtype=float)
        self.h = np.asarray(l if h is None else h, dtype=float)

    def __add__(self, o):
        if not isinstance(o, I):
            o = I(o)
        return I(down(self.l + o.l), up(self.h + o.h))
    __radd__ = __add__

    def __neg__(self):
        return I(-self.h, -self.l)

    def __sub__(self, o):
        return self + (-o if isinstance(o, I) else -np.asarray(o))

    def __rsub__(self, o):
        return -self + o

    def __mul__(self, o):
        if not isinstance(o, I):
            o = I(o)
        vals = np.stack([self.l * o.l, self.l * o.h, self.h * o.l, self.h * o.h])
        return I(down(vals.min(axis=0)), up(vals.max(axis=0)))
    __rmul__ = __mul__

    def inv(self):
        if np.any((self.l <= 0) & (self.h >= 0)):
            raise ValueError('division by zero interval')
        return I(down(1 / self.h), up(1 / self.l))

    def __truediv__(self, o):
        return self * (o.inv() if isinstance(o, I) else I(o).inv())

    def sq(self):
        lower = np.where((self.l <= 0) & (self.h >= 0), 0, np.minimum(self.l ** 2, self.h ** 2))
        return I(np.maximum(0, down(lower)), up(np.maximum(self.l ** 2, self.h ** 2)))

    def absmax(self):
        return up(np.maximum(abs(self.l), abs(self.h)))

@lru_cache(None)
def trigpt(x, kind):
    a = iv.sin(iv.mpf(x)) if kind == 'sin' else iv.cos(iv.mpf(x))
    return (np.nextafter(float(a.a), -np.inf), np.nextafter(float(a.b), np.inf))

def sincos(x):
    slo = np.array([trigpt(v, 'sin')[0] for v in x.l])
    shi = np.array([trigpt(v, 'sin')[1] for v in x.h])
    absmax = np.maximum(abs(x.l), abs(x.h))
    absmin = np.where((x.l <= 0) & (x.h >= 0), 0, np.minimum(abs(x.l), abs(x.h)))
    clo = np.array([trigpt(v, 'cos')[0] for v in absmax])
    chi = np.array([trigpt(v, 'cos')[1] for v in absmin])
    return (I(slo, shi), I(clo, chi))

def herm(x):
    H = [I(np.ones_like(x.l)), 2 * x]
    for k in range(1, 5):
        H.append(2 * x * H[-1] - 2 * k * H[-2])
    D = [I(np.zeros_like(x.l))] + [2 * k * H[k - 1] for k in range(1, 6)]
    DD = [I(np.zeros_like(x.l)), I(np.zeros_like(x.l))] + [4 * k * (k - 1) * H[k - 2] for k in range(2, 6)]
    return (H, D, DD)

def herm_scaled(x, s):
    H, D, DD = herm(x / I(float(s)))
    return (H, [v / I(float(s)) for v in D], [v / I(float(s)).sq() for v in DD])

def idot(V, M):
    pos = np.maximum(M, 0)
    neg = np.minimum(M, 0)
    lo = V.l @ pos.T + V.h @ neg.T
    hi = V.h @ pos.T + V.l @ neg.T
    k = M.shape[1]
    gamma = 4 * (k + 2) * UROUND / (1 - 4 * (k + 2) * UROUND)
    mag = np.maximum(abs(V.l), abs(V.h)) @ abs(M).T
    err = up(gamma * mag + np.finfo(float).tiny)
    return I(down(lo - err), up(hi + err))

def stack(vs):
    return I(np.column_stack([v.l for v in vs]), np.column_stack([v.h for v in vs]))

def isumpositive(a, axis):
    n = a.shape[axis]
    return up(np.sum(a, axis=axis) * (1 + 8 * (n + 1) * UROUND) + np.finfo(float).tiny)

def dec(s):
    a = iv.mpf(str(s))
    return I(down(float(a.a)), up(float(a.b)))

def residual_interval(lo, hi, p2):
    p = I(lo[:, 0], hi[:, 0])
    x = I(lo[:, 1], hi[:, 1])
    v = I(lo[:, 2], hi[:, 2])
    s, c = sincos(x)
    den = 1 - D045 * c.sq()
    num = -D981 * s - D111 * v - 0.5 * c * p - D045 * c * s * v.sq()
    f = num / den
    H, Hp, _ = herm_scaled(p, SCALES[0])
    J, Jp, _ = herm_scaled(x, SCALES[1])
    K, Kp, _ = herm_scaled(v, SCALES[2])
    ph = []
    gm = []
    for au, a, b in IDS:
        ph.append(0.5 * H[au] * J[a] * K[b])
        gm.append(0.5 * (Hp[au] * J[a] * K[b] * p2 + H[au] * Jp[a] * K[b] * v + H[au] * J[a] * Kp[b] * f))
    return idot(stack(gm), T) - idot(idot(stack(ph), T), A[2:, 2:]) - I(p.l[:, None], p.h[:, None]) * A[2:, 0][None, :] - p2 * A[2:, 1][None, :]

def derivative_bounds(lo, hi, p2, W=None):
    p = I(lo[:, 0], hi[:, 0])
    x = I(lo[:, 1], hi[:, 1])
    v = I(lo[:, 2], hi[:, 2])
    s, c = sincos(x)
    den = 1 - D045 * c.sq()
    num = -D981 * s - D111 * v - 0.5 * c * p - D045 * c * s * v.sq()
    f = num / den
    fp = -0.5 * c / den
    fv = (-D111 - D09 * c * s * v) / den
    nx = -D981 * c + 0.5 * s * p - D045 * (c.sq() - s.sq()) * v.sq()
    dx = D09 * c * s
    fx = (nx * den - num * dx) / den.sq()
    H, Hp, Hpp = herm_scaled(p, SCALES[0])
    J, Jp, Jpp = herm_scaled(x, SCALES[1])
    K, Kp, Kpp = herm_scaled(v, SCALES[2])
    gd = [[], [], []]
    pd = [[], [], []]
    for au, a, b in IDS:
        phip = 0.5 * Hp[au] * J[a] * K[b]
        phix = 0.5 * H[au] * Jp[a] * K[b]
        phiv = 0.5 * H[au] * J[a] * Kp[b]
        px = 0.5 * Hp[au] * Jp[a] * K[b]
        pv = 0.5 * Hp[au] * J[a] * Kp[b]
        xv = 0.5 * H[au] * Jp[a] * Kp[b]
        xx = 0.5 * H[au] * Jpp[a] * K[b]
        vv = 0.5 * H[au] * J[a] * Kpp[b]
        gd[0].append(px * v + pv * f + phiv * fp)
        gd[1].append(px * p2 + xx * v + xv * f + phiv * fx)
        gd[2].append(pv * p2 + xv * v + phix + vv * f + phiv * fv)
        pd[0].append(phip)
        pd[1].append(phix)
        pd[2].append(phiv)
    outs = []
    for j in range(3):
        out = idot(stack(gd[j]), T) - idot(idot(stack(pd[j]), T), A[2:, 2:])
        if j == 0:
            out = out - I(A[2:, 0][None, :])
        if W is not None:
            out = idot(out, W.T)
        outs.append(out.absmax())
    return np.stack(outs, axis=2)

def upper_bounds(lo, hi, p2, W=None):
    cen = (lo + hi) / 2
    half = up(np.maximum(cen - lo, hi - cen))
    der = derivative_bounds(lo, hi, p2, W)
    rc0 = residual_interval(cen, cen, p2)
    rc = (idot(rc0, W.T) if W is not None else rc0).absmax()
    rad = isumpositive(up(der * half[:, None, :]), axis=2)
    comps = up(rc + rad)
    return up(np.sqrt(isumpositive(up(comps ** 2), axis=1)))


D045, D09, D981 = dec('.45'), dec('.9'), dec('9.81')
D111 = I(1)/D09

def inverse_defect(D, W):
    df = np.vectorize(lambda x: F(float(x)), otypes=[object])(D)
    wf = np.vectorize(lambda x: F(float(x)), otypes=[object])(W)
    E = np.eye(len(D), dtype=object) - wf @ df
    n1 = max((sum((abs(x) for x in E[:, j])) for j in range(len(E))))
    ni = max((sum((abs(x) for x in row)) for row in E))
    prod = n1 * ni
    v = iv.mpf(prod.numerator) / prod.denominator
    return float(np.nextafter(float(iv.sqrt(v).b), np.inf))

def check_structure():
    md = model
    A = md['A']
    n = len(A)
    _require(n == 8 and md['indices'].tolist() == [[0, 1, 0], [0, 0, 1], [0, 0, 3], [0, 1, 2], [0, 2, 1], [0, 3, 0]], "verification.py: certificate check failed at n == 8 and md['indices'].tolist() == [[0, 1, 0], [0, 0, 1], [0, 0, 3], [0, 1, 2], [0, 2, 1], [0, 3, 0]]")
    _require(np.array_equal(md['scales'], np.ones(3)) and np.array_equal(md['T'], np.eye(6)), "verification.py: certificate check failed at np.array_equal(md['scales'], np.ones(3)) and np.array_equal(md['T'], np.eye(6))")
    a = np.zeros(8)
    a[3] = 1
    _require(np.array_equal(A[2], a), 'verification.py: certificate check failed at np.array_equal(A[2], a)')
    a = np.zeros(8)
    a[6] = 3
    _require(np.array_equal(A[7], a), 'verification.py: certificate check failed at np.array_equal(A[7], a)')
    _require(np.all(A[2:, 1] == 0), 'verification.py: certificate check failed at np.all(A[2:, 1] == 0)')
    actuator_rows = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                              [-10000, -200, 0, 0, 0, 0, 0, 0]], float)
    _require(np.array_equal(A[:2], actuator_rows), 'The actuator drift rows are not exact')
    _require(np.array_equal(md['B'], np.array([0, 10000, 0, 0, 0, 0, 0, 0])),
             'The actuator input matrix is inconsistent with the physical plant')
    _require(np.array_equal(md['E'], np.eye(8)[:, 2:])
             and np.array_equal(md['C'], np.eye(8)[2:4]),
             'The closure embedding or physical-state readout is inconsistent')
    _require(np.array_equal(md['nonnull'], np.array([1, 2, 3, 4])),
             'The shaped channel must retain exactly residual components 2 through 5')
    return md

def verify_shape(md, shape, max_boxes=2000000, cap=1.0, batch=2048):
    """Enclose the nonlinear residual for the stored finite model A.

    This certifies gamma - A Z directly. It does not certify the error of
    the quadrature used to approximate the exact L2 Galerkin projection.
    """
    set_region_model(md)
    md = check_structure()
    D = np.asarray(shape, float)
    _require(D.shape == (4, 4) and np.all(np.isfinite(D)),
             'The residual shape must be a finite 4-by-4 matrix')
    _require(np.isfinite(cap) and cap > 0, 'The normalized residual cap must be positive and finite')
    W = np.linalg.inv(D)
    eta = inverse_defect(D, W)
    if not eta < 1:
        raise ValueError('Inverse defect not contractive')
    weights = np.zeros((6, 4))
    weights[np.asarray(md['nonnull'], int)] = W.T
    lim = md['bounds']
    lo = np.array([[-lim[0], -lim[2], -lim[3]], [lim[0], -lim[2], -lim[3]]])
    hi = np.array([[-lim[0], lim[2], lim[3]], [lim[0], lim[2], lim[3]]])
    start = time.time()
    tested = accepted = depth = 0
    max_upper = 0.0
    max_lower = 0.0
    badpoints = []
    worst_boxes = []
    target = float((I(cap) * (I(1) - I(eta))).l)
    while len(lo):
        newlo = []
        newhi = []
        for k in range(0, len(lo), batch):
            ll, hh = (lo[k:k + batch], hi[k:k + batch])
            ub = upper_bounds(ll, hh, 0.0, weights)
            tested += len(ll)
            ok = ub <= target
            accepted += int(ok.sum())
            if ok.any():
                max_upper = max(max_upper, float(np.max(ub[ok])))
            ll, hh = (ll[~ok], hh[~ok])
            if not len(ll):
                continue
            mid = (ll + hh) / 2
            rr = idot(residual_interval(mid, mid, 0.0), weights.T)
            minabs = np.maximum(0, np.maximum(rr.l, -rr.h))
            sqsum = down(np.sum(down(minabs ** 2), axis=1) * (1 - 8 * (minabs.shape[1] + 1) * np.finfo(float).eps))
            lb = np.maximum(0, down(np.sqrt(np.maximum(0, sqsum))))
            max_lower = max(max_lower, float(max(lb)))
            bad = lb > np.nextafter(cap * (1 + eta), np.inf)
            if bad.any():
                badpoints.extend(mid[bad].tolist())
            if len(badpoints):
                break
            dim = 1 + np.argmax((hh[:, 1:] - ll[:, 1:]) / (2 * lim[[2, 3]]), axis=1)
            l1 = ll.copy()
            h1 = hh.copy()
            l2 = ll.copy()
            h2 = hh.copy()
            ix = np.arange(len(ll))
            h1[ix, dim] = mid[ix, dim]
            l2[ix, dim] = mid[ix, dim]
            newlo.extend([l1, l2])
            newhi.extend([h1, h2])
        if badpoints:
            break
        lo = np.concatenate(newlo) if newlo else np.empty((0, 3))
        hi = np.concatenate(newhi) if newhi else np.empty((0, 3))
        depth += 1
        if tested >= max_boxes:
            break
    verified = not badpoints and len(lo) == 0
    corrected = float((I(max_upper) / (I(1) - I(eta))).h) if accepted else None
    if not verified:
        raise RuntimeError("The whole-region residual enclosure did not close.")
    return dict(verified=True, normalized_upper=float(corrected),
                boxes_tested=tested, seconds=time.time()-start)

def verify_auxiliary(model):
    """Enclose the four-channel residual of the cubic localization model."""
    _require(np.array_equal(model['D_closure'], np.eye(6)[:,model['nonnull']]@model['shape']),
             'The auxiliary closure embedding differs from its residual shape')
    previous = globals().get('model')
    try:
        report = verify_shape(model,model['shape'])
    finally:
        if previous is not None:
            set_region_model(previous)
    report.update(lift_degree=3, plant_states=8, closure_channels=4)
    return report
