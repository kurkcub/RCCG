from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import solve_continuous_are, eigvals, matrix_balance
from scipy.optimize import minimize_scalar

def close(G, K):
    A, B1, B2, C1, C2, D11, D12, D21 = (G[k] for k in ('A', 'B1', 'B2', 'C1', 'C2', 'D11', 'D12', 'D21'))
    AK, BK, CK, DK = (K[k] for k in ('AK', 'BK', 'CK', 'DK'))
    AC = np.block([[A + B2 @ DK @ C2, B2 @ CK], [BK @ C2, AK]])
    BC = np.vstack((B1 + B2 @ DK @ D21, BK @ D21))
    CC = np.hstack((C1 + D12 @ DK @ C2, D12 @ CK))
    DC = D11 + D12 @ DK @ D21
    return (AC, BC, CC, DC)

def care_res(A, B, Q, R, N, X):
    G1 = X @ B + N
    terms = [A.T @ X + X @ A, Q, -G1 @ np.linalg.solve(R, G1.T)]
    err = np.linalg.norm(sum(terms), 'fro') / max(1.0, sum((np.linalg.norm(t, 'fro') for t in terms)))
    eig = np.linalg.eigvalsh((X + X.T) * 0.5)
    return (err, float(eig.min()), float(eig.max()))

def synth_at(G, gamma):
    A, B1, B2, C1, C2, D11, D12, D21 = (G[k] for k in ('A', 'B1', 'B2', 'C1', 'C2', 'D11', 'D12', 'D21'))
    n = len(A)
    nw = B1.shape[1]
    nz = C1.shape[0]
    B = np.hstack((B1, B2))
    Ct = np.vstack((C1, C2))
    D1 = np.hstack((D11, D12))
    Dt1 = np.vstack((D11, D21))
    Rx = D1.T @ D1
    Rx[:nw, :nw] -= gamma ** 2 * np.eye(nw)
    Ry = Dt1 @ Dt1.T
    Ry[:nz, :nz] -= gamma ** 2 * np.eye(nz)
    sx = C1.T @ D1
    sy = B1 @ Dt1.T
    try:
        X = solve_continuous_are(A, B, C1.T @ C1, Rx, s=sx, balanced=True)
        Y = solve_continuous_are(A.T, Ct.T, B1 @ B1.T, Ry, s=sy, balanced=True)
        dx = care_res(A, B, C1.T @ C1, Rx, sx, X)
        dy = care_res(A.T, Ct.T, B1 @ B1.T, Ry, sy, Y)
        if dx[1] < -1e-07 * max(1, dx[2]) or dy[1] < -1e-07 * max(1, dy[2]):
            return (None, 'non-PSD')
        if dx[0] > 2e-07 or dy[0] > 2e-07:
            return (None, 'CARE accuracy')
        rad = max(abs(eigvals(Y @ X))) / gamma ** 2
        if rad >= 1:
            return (None, 'coupling')
        F = -np.linalg.solve(Rx, D1.T @ C1 + B.T @ X)
        H = -(B1 @ Dt1.T + Y @ Ct.T) @ np.linalg.inv(Ry)
        Z = np.linalg.solve(np.eye(n) - Y @ X / gamma ** 2, np.eye(n))
        F2 = F[nw:]
        H12 = H[:, nz - 1:nz]
        H2 = H[:, nz:]
        CK = F2 @ Z
        AK = A + H @ Ct + (B2 + H12) @ CK
        BK = -H2
        DK = np.zeros((1, 1))
        K = dict(AK=AK, BK=BK, CK=CK, DK=DK)
        CL = close(G, K)
        ab = float(max(eigvals(CL[0]).real))
        if ab >= -1e-08:
            return (None, 'CL unstable ' + str(ab))
        K.update(gamma=gamma, coupling=float(rad), care_x=dx, care_y=dy, abscissa=ab, X=X, Y=Y)
        return (K, 'ok')
    except (np.linalg.LinAlgError, ValueError) as exc:
        return (None, str(exc))

def synthesize(G, initial=1000.0, rel_tol=1e-05):
    high = initial
    best = None
    for i in range(12):
        best, msg = synth_at(G, high)
        if best is not None:
            break
        high *= 2
    if best is None:
        raise RuntimeError(msg)
    low = 0.0
    for j in range(55):
        mid = (low + high) / 2
        K, msg = synth_at(G, mid)
        if K is None:
            low = mid
        else:
            high = mid
            best = K
        if high - low < rel_tol * max(1, high):
            break
    K, msg = synth_at(G, high * 1.01)
    if K is None:
        K = best
    K['feasible_threshold'] = high
    return K

def freq_gain(CL, w):
    A, B, C, D = CL
    return float(np.linalg.svd(C @ np.linalg.solve(1j * w * np.eye(len(A)) - A, B) + D, compute_uv=False)[0])

def norm_grid(CL, nf=2200):
    w = np.r_[0, np.geomspace(1e-05, 1000000.0, nf)]
    gains = np.array([freq_gain(CL, o) for o in w])
    candidates = [(gains[0], 0.0), (np.linalg.svd(CL[3], compute_uv=False)[0], np.inf)]
    peak = (gains[1:-1] > gains[:-2]) & (gains[1:-1] >= gains[2:])
    for k in np.flatnonzero(peak) + 1:
        if k <= 1:
            continue
        opt = minimize_scalar(lambda lw: -freq_gain(CL, np.exp(lw)), bounds=np.log([w[k - 1], w[k + 1]]), method='bounded', options={'xatol': 1e-11})
        candidates.append((-opt.fun, np.exp(opt.x)))
    return (max(candidates), w, gains)

def generalized(md, D, w, cap=0.682, reg=None, wu=0.0):
    A, B, C, E = [md[k] for k in ('A', 'B', 'C', 'E')]
    C = C[:1]
    B = B.reshape(-1, 1)
    n, nf = len(A), len(w['A_WP'])
    nu = len(w.get('A_WU', []))
    du = float(np.asarray(w.get('D_WU', reg if reg is not None else wu)).item())
    effort = nu > 0 or reg is not None or du > 0
    ng, nw = n + nf + nu, D.shape[1] + 1
    fp, fu = slice(n, n + nf), slice(n + nf, ng)
    Ag = np.zeros((ng, ng))
    Ag[:n, :n] = A
    Ag[fp, :n] = -w['B_WP'] @ C
    Ag[fp, fp] = w['A_WP']
    B1 = np.zeros((ng, nw))
    B1[:n, :-1] = E @ D
    B1[fp, -1:] = w['B_WP'] * cap
    B2 = np.zeros((ng, 1))
    B2[:n] = B
    C1 = np.zeros((1 + int(effort), ng))
    C1[:1, :n] = -w['D_WP'] @ C
    C1[:1, fp] = w['C_WP']
    C2 = np.zeros((1, ng))
    C2[:, :n] = -C
    D11 = np.zeros((len(C1), nw))
    D11[:1, -1:] = w['D_WP'] * cap
    D12 = np.zeros((len(C1), 1))
    if effort:
        D12[1, 0] = du
    if nu:
        Ag[fu, fu] = w['A_WU']
        B2[fu] = w['B_WU']
        C1[1:2, fu] = w['C_WU']
    D21 = np.zeros((1, nw))
    D21[0, -1] = cap
    if reg is not None:
        B2 /= reg
        D12 /= reg
        C2 /= cap
        D21 /= cap
    diag = np.ones(ng)
    diag[:2] = [0.1, 0.01]
    if nf > 1:
        diag[n + 1] = max(float(abs(w['A_WP'][1, 0])), 1.0) ** (-0.5)
    if nu:
        diag[fu] = abs(du)
    S = np.diag(diag)
    return dict(A=S @ Ag / diag[None, :], B1=S @ B1, B2=S @ B2,
                C1=C1 / diag, C2=C2 / diag, D11=D11, D12=D12,
                D21=D21, S=S)

def design_controller(md, Dshape, wp):
    cap = 0.682
    output_scale = float(wp.get('output_scale', 1.0))
    wu = float(wp['WU'])
    normalized = {key: value.copy() for key, value in wp.items()}
    for key in ('C_WP', 'D_WP', 'WP_num', 'C_WU', 'D_WU', 'WU_num'):
        if key in normalized:
            normalized[key] /= output_scale
    input_scale = float(np.asarray(wp.get('D_WU', wu)).item()) / output_scale
    gp = generalized(md, Dshape, normalized, cap=cap, reg=input_scale)
    search = synthesize(gp, initial=1.0, rel_tol=1e-7)
    ratio = float(wp.get('synthesis_ratio', 1.001))
    seed, message = synth_at(gp, search['feasible_threshold'] * ratio)
    if seed is None:
        raise RuntimeError(message)
    A, T = matrix_balance(seed['AK'], permute=False)
    B = np.linalg.solve(T, seed['BK'] / cap)
    C = seed['CK'] / input_scale @ T
    D = seed['DK'] / (input_scale * cap)
    gain = float(wp.get('synthesis_gain', 1.0))
    C *= gain
    D *= gain
    k = dict(A=A, B=B, C=C, D=D, **wp)
    true_gp = generalized(md, Dshape, wp, cap=cap, wu=wu)
    cl = close(true_gp, {q + 'K': k[q] for q in 'ABCD'})
    peak, _, _ = norm_grid(cl, nf=1800)
    if max(eigvals(cl[0]).real) >= 0:
        raise RuntimeError("The synthesized feedback is not stable.")
    report = dict(order=len(A), WU=wu, WU_order=len(wp.get('A_WU', [])), WT=0.0,
                  seed_gamma=float(seed['gamma'] * output_scale),
                  normalized_seed_gamma=float(seed['gamma']),
                  output_normalization=output_scale, output_gain=gain,
                  synthesis_ratio=ratio, coupling=float(seed['coupling']),
                  CARE_X_relative_residual=float(seed['care_x'][0]),
                  CARE_Y_relative_residual=float(seed['care_y'][0]),
                  norm_estimate=float(peak[0]), peak_frequency=float(peak[1]))
    return k, report


def verify_galerkin(model):
    """Compare the stored coefficients with independent tensor quadrature."""
    nodes, weights = [], []
    for halfwidth, order in zip(model['bounds'], (4, 4, 64, 12)):
        x, w = leggauss(order)
        nodes.append(halfwidth*x)
        weights.append(halfwidth*w)
    points = np.array(np.meshgrid(*nodes, indexing='ij')).reshape(4, -1).T
    weights = np.einsum('i,j,k,l->ijkl', *weights).ravel()
    p, p2, q, v = points.T
    lift = np.column_stack((p,p2,q,v,4*v**3-6*v,4*q*v*v-2*q,
                            4*q*q*v-2*v,4*q**3-6*q))
    sn, cs = np.sin(q), np.cos(q)
    acc = (-9.81*sn-v/.9-.5*p*cs-.45*cs*sn*v*v)/(1-.45*cs*cs)
    field = np.column_stack((p2,-10000*p-200*p2,v,acc,(12*v*v-6)*acc,
        4*v**3-2*v+8*q*v*acc,8*q*v*v+(4*q*q-2)*acc,(12*q*q-6)*v))
    R = (lift.T*weights) @ lift
    Q = (field.T*weights) @ lift
    recalculated = np.linalg.solve(R.T, Q.T).T
    recalculated[:2] = 0
    recalculated[0,1] = 1
    recalculated[1,:2] = (-10000,-200)
    recalculated[2] = np.eye(8)[3]
    recalculated[7] = 3*np.eye(8)[6]
    recalculated[2:,1] = 0
    error = float(np.max(np.abs(recalculated-model['A'])))
    if error > 1e-8:
        raise RuntimeError("The Galerkin coefficients failed the quadrature check.")
    if not np.array_equal(model['C'], np.eye(8)[2:4]):
        raise ValueError("The physical-state readout is inconsistent with the dictionary.")
    return error


# Dimensionwise MDI-TP Galerkin construction.
"""Structure-exploiting MDI-TP for the cart-pendulum Galerkin integrals.

The quadrature rule is the original four-dimensional tensor-product
Gauss--Legendre rule.  Its sums are contracted dimension by dimension;
no tensor grid is constructed.  Polynomial dependence on p1, p2 and v is
eliminated using reusable one-dimensional quadrature moments.  The
remaining coefficient functions are evaluated at the q nodes only.

Public API: galerkin_model(bounds, degree=3, orders=(4, 4, 64, 12)).
R and Q use unnormalized Lebesgue measure; diagnostics use regional RMS.
"""
from functools import lru_cache
from time import perf_counter

import numpy as np
from numpy.polynomial.legendre import leggauss

LD = np.longdouble


def indices(degree):
    if degree < 1 or degree % 2 != 1:
        raise ValueError("degree must be a positive odd integer")
    return [(1, 0), (0, 1)] + [
        (a, d-a) for d in range(3, degree+1, 2) for a in range(d+1)]


def _poly_add(a, b):
    out = np.zeros(max(len(a), len(b)), dtype=LD)
    out[:len(a)] += a
    out[:len(b)] += b
    return out


@lru_cache(None)
def _hermite_coefficients(n):
    if n == 0:
        return np.array([1], dtype=LD)
    if n == 1:
        return np.array([0, 2], dtype=LD)
    return _poly_add(np.r_[LD(0), 2*_hermite_coefficients(n-1)],
                     -2*(n-1)*_hermite_coefficients(n-2))


@lru_cache(None)
def _legendre_coefficients(n):
    if n == 0:
        return np.array([1], dtype=LD)
    if n == 1:
        return np.array([0, 1], dtype=LD)
    return _poly_add(np.r_[LD(0), (2*n-1)*_legendre_coefficients(n-1)],
                     -(n-1)*_legendre_coefficients(n-2))/n


def _polyval(c, x):
    out = np.zeros_like(x, dtype=LD)
    for ci in c[::-1]:
        out = out*x + ci
    return out


def _poly_to_hermite(c):
    """Convert monomial coefficients by x H_n = H_(n+1)/2 + n H_(n-1)."""
    out = np.zeros(len(c), dtype=LD)
    power = np.zeros(len(c), dtype=LD)
    power[0] = 1
    for k, ci in enumerate(c):
        out += ci*power
        if k+1 < len(c):
            new = np.zeros_like(power)
            for j in range(k+1):
                new[j+1] += power[j]/2
                if j:
                    new[j-1] += j*power[j]
            power = new
    return out


class _Symbol:
    """Sparse powers of (p1,p2,v), with coefficient arrays on q nodes."""
    def __init__(self, terms):
        self.terms = {key: np.asarray(value, dtype=LD)
                      for key, value in terms.items() if np.any(value)}

    def __add__(self, other):
        if not isinstance(other, _Symbol):
            other = _Symbol({(0, 0, 0): LD(other)})
        out = dict(self.terms)
        for key, value in other.terms.items():
            out[key] = out.get(key, LD(0)) + value
        return _Symbol(out)

    __radd__ = __add__

    def __neg__(self):
        return _Symbol({key: -value for key, value in self.terms.items()})

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if not isinstance(other, _Symbol):
            return _Symbol({key: value*LD(other) for key, value in self.terms.items()})
        out = {}
        for ka, va in self.terms.items():
            for kb, vb in other.terms.items():
                key = tuple(a+b for a, b in zip(ka, kb))
                out[key] = out.get(key, LD(0)) + va*vb
        return _Symbol(out)

    __rmul__ = __mul__

    def at_zero_p(self):
        return _Symbol({key: value for key, value in self.terms.items()
                        if key[0] == 0 and key[1] == 0})


class _MDI:
    def __init__(self, bounds, orders):
        self.bounds = np.asarray(bounds, dtype=LD)
        self.orders = tuple(int(n) for n in orders)
        self.nodes, self.weights = [], []
        for b, n in zip(self.bounds, self.orders):
            nodes, weights = leggauss(n)
            # Match the original rule's binary64 nodes before accumulating in LD.
            self.nodes.append(np.asarray(float(b)*nodes, dtype=LD))
            self.weights.append(np.asarray(weights/2, dtype=LD))
        self.q = self.nodes[2]
        self.moments = {}
        self.moment_reuses = 0
        self.integrals = 0
        self.stage_terms = [0, 0, 0]

    def moment(self, dimension, exponent):
        key = (dimension, exponent)
        if key in self.moments:
            self.moment_reuses += 1
            return self.moments[key]
        # Odd moments vanish exactly by the symmetric GL rule; this removes
        # zero symbols before any downstream dimension is processed.
        value = LD(0) if exponent % 2 else np.sum(
            self.weights[dimension]*self.nodes[dimension]**exponent, dtype=LD)
        self.moments[key] = value
        return value

    def integrate(self, symbol):
        terms = symbol.terms
        # MDI stages: p1 -> p2 -> v.  Each stage combines equal remaining
        # symbols, so a later stage evaluates their common sum only once.
        for stage, dimension in enumerate((0, 1, 3)):
            partial = {}
            for key, value in terms.items():
                factor = self.moment(dimension, key[0])
                if factor != 0:
                    remaining = key[1:]
                    partial[remaining] = partial.get(remaining, LD(0)) + factor*value
            terms = partial
            self.stage_terms[stage] += len(terms)
        remaining = terms.get((), LD(0))
        self.integrals += 1
        return np.sum(self.weights[2]*remaining, dtype=LD)

    def tensor_polynomial(self, qcoeff, vcoeff, factor=1):
        cq = _polyval(qcoeff, self.q)*LD(factor)
        return _Symbol({(0, 0, b): cq*cb for b, cb in enumerate(vcoeff) if cb})


def galerkin_model(bounds, degree=3, orders=(4, 4, 64, 12)):
    """Return a fresh Galerkin realization evaluated with MDI-TP.

    Returned keys: A, B (column), C, E, R, Q, bounds, modes, diagnostics,
    metadata.  The coordinate order is (p1,p2), then indices(degree).
    R = integral Z Z^T and Q = integral Gamma Z^T on the given box.
    """
    started = perf_counter()
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (4,) or np.any(bounds <= 0) or len(orders) != 4:
        raise ValueError("positive four-dimensional halfwidths and four orders required")
    if min(int(orders[2]), int(orders[3])) <= degree:
        raise ValueError("state quadrature orders must exceed the dictionary degree")
    if min(int(orders[0]), int(orders[1])) < 2:
        raise ValueError("actuator quadrature orders must be at least two")
    modes = indices(degree)
    mdi = _MDI(bounds, orders)
    p = _Symbol({(1, 0, 0): LD(1)})
    p2 = _Symbol({(0, 1, 0): LD(1)})
    v = _Symbol({(0, 0, 1): LD(1)})
    s, c = np.sin(mdi.q), np.cos(mdi.q)
    den = 1-LD('0.45')*c*c
    accel = _Symbol({(0, 0, 0): -LD('9.81')*s/den,
                     (0, 0, 1): -1/(LD('0.9')*den),
                     (1, 0, 0): -LD('0.5')*c/den,
                     (0, 0, 2): -LD('0.45')*c*s/den})
    z = [p, p2]
    gamma = [p2, -10000*p-200*p2]
    for a, b in modes:
        hq, hv = _hermite_coefficients(a), _hermite_coefficients(b)
        z.append(mdi.tensor_polynomial(hq, hv, .5))
        da = (mdi.tensor_polynomial(_hermite_coefficients(a-1), hv, a)*v
              if a else _Symbol({}))
        db = (mdi.tensor_polynomial(hq, _hermite_coefficients(b-1), b)*accel
              if b else _Symbol({}))
        gamma.append(da+db)
    n = len(z)
    # Equivalent orthonormal Legendre basis.  It spans exactly the same
    # finite space as the odd-total-degree Hermite dictionary.
    transform = np.zeros((n, n), dtype=LD)
    transform[0, 0] = np.sqrt(LD(3))/LD(bounds[0])
    transform[1, 1] = np.sqrt(LD(3))/LD(bounds[1])
    orthogonal = [p*transform[0, 0], p2*transform[1, 1]]
    for row, (a, b) in enumerate(modes, 2):
        cq = _legendre_coefficients(a)/LD(bounds[2])**np.arange(a+1)
        cv = _legendre_coefficients(b)/LD(bounds[3])**np.arange(b+1)
        normalization = np.sqrt(LD((2*a+1)*(2*b+1)))
        orthogonal.append(mdi.tensor_polynomial(cq, cv, normalization))
        hq, hv = _poly_to_hermite(cq), _poly_to_hermite(cv)
        for col, (i, j) in enumerate(modes, 2):
            if i < len(hq) and j < len(hv):
                transform[row, col] = 2*normalization*hq[i]*hv[j]
    # The orthogonal Gram is the identity in exact arithmetic.  Solve its
    # actually evaluated, well-conditioned counterpart as well, retaining
    # the small finite-precision off-diagonal quadrature terms.
    ortho_gram = np.array([[mdi.integrate(oi*oj) for oj in orthogonal]
                           for oi in orthogonal], dtype=float)
    cross = np.array([[mdi.integrate(g*o) for o in orthogonal]
                      for g in gamma], dtype=float)
    orthogonal_coefficients = np.linalg.solve(ortho_gram, cross.T).T
    a_mdi = np.asarray(orthogonal_coefficients, dtype=LD) @ transform
    A = np.asarray(a_mdi, dtype=float)
    # Preserve analytically exact coordinate dynamics and absent p2 coupling.
    A[:2] = 0
    A[0, 1] = 1
    A[1, :2] = [-10000, -200]
    A[2:, 1] = 0
    A[2] = 0
    A[2, 3] = 1
    for order in range(3, degree+1, 2):
        row = modes.index((order, 0))+2
        A[row] = 0
        A[row, modes.index((order-1, 1))+2] = order
    B = np.zeros((n, 1))
    B[1, 0] = 10000
    C = np.eye(n)[[2, 3]]
    E = np.eye(n)[:, 2:]
    gram = np.array([[mdi.integrate(zi*zj) for zj in z] for zi in z], dtype=LD)
    generator_cross = np.array([[mdi.integrate(gi*zj) for zj in z] for gi in gamma], dtype=LD)
    projected_acc = sum((zi*coefficient for zi, coefficient in zip(z, A[3])), _Symbol({}))
    residual = accel-projected_acc
    drift_residual = residual.at_zero_p()
    bfield = _Symbol({(0, 0, 0): -LD('0.5')*c/den})
    bmean = mdi.integrate(bfield)
    coupling = p*(bfield-bmean)
    r2, f2, b2 = (mdi.integrate(expr*expr) for expr in (residual, drift_residual, coupling))
    scales = np.sqrt(np.asarray(np.diag(gram), dtype=float))
    # Singular values of the scaled basis transformation equal those of
    # its weighted quadrature design matrix, without constructing that grid.
    normalized_basis_map = np.linalg.solve(np.asarray(transform, dtype=float), np.eye(n))/scales[:, None]
    diagnostics = {
        'common_acceleration_residual_regional_rms': float(np.sqrt(max(r2, 0))),
        'state_drift_projection_regional_rms': float(np.sqrt(max(f2, 0))),
        'input_coupling_regional_rms': float(np.sqrt(max(b2, 0))),
        'mean_input_coefficient': float(bmean),
        'orthogonal_split_error': float(abs(r2-f2-b2)),
        'weighted_design_condition': float(np.linalg.cond(normalized_basis_map)),
        'max_real_eigenvalue': float(np.linalg.eigvals(A).real.max()),
    }
    normalized_error = (np.asarray(A, dtype=LD)@gram-generator_cross)/np.sqrt(np.diag(gram))[None, :]
    normalized_scale = np.maximum(LD(1), np.max(abs(
        generator_cross/np.sqrt(np.diag(gram))[None, :]), axis=1))
    volume = np.prod(2*np.asarray(bounds, dtype=LD))
    metadata = {
        'method': 'MDI-TP',
        'rule': 'tensor-product Gauss-Legendre',
        'orders': [int(k) for k in orders],
        'measure': 'unnormalized Lebesgue on the symmetric box',
        'volume': float(volume),
        'dimension_elimination_order': ['p1', 'p2', 'x2', 'x1'],
        'tensor_grid_materialized': False,
        'equivalent_tensor_node_count': int(np.prod(orders)),
        'nonpolynomial_coefficient_node_count': int(orders[2]),
        'cached_one_dimensional_moments': len(mdi.moments),
        'cached_moment_reuses': mdi.moment_reuses,
        'contracted_integrals': mdi.integrals,
        'retained_partial_symbols_per_stage_total': mdi.stage_terms,
        'accumulation_precision_bits': int(np.finfo(LD).nmant+1),
        'scaled_gram_cross_consistency': float(np.max(abs(normalized_error)/normalized_scale[:, None])),
        'wall_seconds': perf_counter()-started,
    }
    return {'A': A, 'B': B, 'C': C, 'E': E,
            'R': np.asarray(volume*gram, dtype=float),
            'Q': np.asarray(volume*generator_cross, dtype=float),
            'bounds': bounds, 'modes': modes,
            'diagnostics': diagnostics, 'metadata': metadata}


def validate_tensor_reference(bounds=(100., 1e5, .75, .5)):
    """Compare MDI-TP coefficients with full tensor quadrature."""
    from numpy.polynomial.hermite import hermval, hermder

    bounds = np.asarray(bounds, dtype=float)
    volume = float(np.prod(2*bounds))
    orders = (4, 4, 64, 12)
    grids, weights = [], []
    for b, order in zip(bounds, orders):
        nodes, w = leggauss(order)
        grids.append(b*nodes)
        weights.append(w/2)
    p, p2, q, v = np.array(np.meshgrid(*grids, indexing='ij')).reshape(4, -1)
    wt = np.einsum('i,j,k,l->ijkl', *weights).ravel()

    def hermite(n, x, derivative=False):
        c = np.zeros(n+1)
        c[n] = 1
        return hermval(x, hermder(c) if derivative else c)

    acceleration = (-9.81*np.sin(q)-v/.9-.5*p*np.cos(q)
                    -.45*np.cos(q)*np.sin(q)*v*v)/(1-.45*np.cos(q)**2)
    report = {}
    for degree in (1, 3, 5):
        model = galerkin_model(bounds, degree, orders)
        modes = model['modes']
        z = np.column_stack([p, p2] + [
            .5*hermite(a, q)*hermite(b, v) for a, b in modes])
        gamma = np.column_stack([p2, -10000*p-200*p2] + [
            .5*(hermite(a, q, True)*hermite(b, v)*v
                 +hermite(a, q)*hermite(b, v, True)*acceleration)
            for a, b in modes])
        scales = np.sqrt(wt @ (z*z))
        design = z/scales*np.sqrt(wt[:, None])
        reference = np.linalg.lstsq(design, gamma*np.sqrt(wt[:, None]),
                                   rcond=None)[0].T/scales
        reference[:2] = 0
        reference[0, 1] = 1
        reference[1, :2] = [-10000, -200]
        reference[2:, 1] = 0
        reference[2] = 0
        reference[2, 3] = 1
        for order in range(3, degree+1, 2):
            row = modes.index((order, 0))+2
            reference[row] = 0
            reference[row, modes.index((order-1, 1))+2] = order
        gram = (z.T*wt)@z*volume
        cross = (gamma.T*wt)@z*volume
        gamma_scale = np.sqrt(wt @ (gamma*gamma))
        gram_scale = volume*scales[:, None]*scales[None, :]
        cross_scale = volume*gamma_scale[:, None]*scales[None, :]
        coefficient_error = float(np.max(abs(reference-model['A'])))
        prediction_difference = z@(reference-model['A']).T
        prediction_error = float(np.max(np.sqrt(wt@(prediction_difference**2))
                                        /np.maximum(1, gamma_scale)))
        gram_error = float(np.max(abs(gram-model['R'])/np.maximum(1, gram_scale)))
        cross_error = float(np.max(abs(cross-model['Q'])/np.maximum(1, cross_scale)))
        rv = acceleration-z@model['A'][3]
        rms_error = abs(float(np.sqrt(wt@(rv*rv)))-
                        model['diagnostics']['common_acceleration_residual_regional_rms'])
        refined = galerkin_model(bounds, degree, (4, 4, 96, 12))
        refinement_difference = float(np.max(abs(model['A']-refined['A'])))
        assert np.array_equal(model['A'][:2], reference[:2])
        assert np.array_equal(model['A'][2], reference[2])
        assert np.count_nonzero(model['A'][2:, 1]) == 0
        for order in range(3, degree+1, 2):
            row = modes.index((order, 0))+2
            assert np.array_equal(model['A'][row], reference[row])
        assert coefficient_error < 5e-7, (degree, coefficient_error)
        assert prediction_error < 1e-10, (degree, prediction_error)
        assert max(gram_error, cross_error) < 1e-12
        assert rms_error < 1e-10
        assert model['metadata']['scaled_gram_cross_consistency'] < 1e-11
        report[str(degree)] = {
            'full_tensor_vs_mdi_max_coefficient_difference': coefficient_error,
            'full_tensor_vs_mdi_relative_prediction_rms_difference': prediction_error,
            'gram_scaled_difference': gram_error,
            'cross_gram_scaled_difference': cross_error,
            'regional_residual_rms_difference': rms_error,
            'mdi_64_vs_96_max_coefficient_difference': refinement_difference,
            'exact_structure_passed': True,
            'metadata': model['metadata'],
        }
    return {'passed': True, 'reference': 'independent full four-dimensional tensor grid with weighted SVD least squares',
            'degree': report}


