"""Nominal plant, output-feedback, backstepping and Koopman-based hybrid MPC kernels.

All closed loops are integrated with fixed-step RK4 (an exponential integrator
for the linear part of the H-infinity loops) in numba. Exit times from the
quadrature box BOX and the local box LOCAL are monitored on every substep.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.linalg import cho_factor, cho_solve, expm

BOX = np.array([100.0, 1e5, 0.75, 0.5])
LOCAL = np.array([32.0, 3200.0, 0.725, 0.47])
TS = 0.005


def physical_jacobian(q=0.0, p=None):
    if p is None:
        p = -19.62 * np.tan(q)
    s, c = np.sin(q), np.cos(q)
    den = 1 - 0.45 * c * c
    f = (-9.81 * s - 0.5 * p * c) / den
    aq = (-9.81 * c + 0.5 * p * s) / den - f * (0.9 * c * s) / den
    return np.array(
        [[0, 1, 0, 0], [-10000, -200, 0, 0], [0, 0, 0, 1], [-0.5 * c / den, 0, aq, -1 / 0.9 / den]],
        float,
    )


def cl_matrix(k, q=0.0):
    A = physical_jacobian(q)
    B = np.array([[0.0], [10000.0], [0.0], [0.0]])
    C = np.array([[0.0, 0.0, 1.0, 0.0]])
    AC = np.block([[A - B @ k["D"] @ C, B @ k["C"]], [-k["B"] @ C, k["A"]]])
    BR = np.vstack([B @ k["D"], k["B"]])
    return AC, BR


def phi_blocks(L, h):
    n = len(L)
    M = np.zeros((4 * n, 4 * n))
    M[:n, :n] = L * h
    M[:n, n : 2 * n] = np.eye(n)
    M[n : 2 * n, 2 * n : 3 * n] = np.eye(n)
    M[2 * n : 3 * n, 3 * n :] = np.eye(n)
    E = expm(M)
    return E[:n, :n], E[:n, n : 2 * n], E[:n, 2 * n : 3 * n], E[:n, 3 * n :]


def matrices(k, h):
    A, BR = cl_matrix(k)
    n = len(A)
    L = np.zeros((n + 2, n + 2))
    L[:n, :n] = A
    L[:n, n : n + 1] = BR
    L[n, n + 1] = 1
    E, p1, p2, p3 = phi_blocks(L, h)
    E2, ph, _, _ = phi_blocks(L, h / 2)
    return (
        E,
        E2,
        (h / 2 * ph)[:, 3],
        (h * (p1 - 3 * p2 + 4 * p3))[:, 3],
        (h * (p2 - 2 * p3))[:, 3],
        (h * (-p2 + 4 * p3))[:, 3],
    )


@njit(inline="always")
def nonlinear(x):
    p, _, q, v = x[:4]
    sn, cs = np.sin(q), np.cos(q)
    exact = (-9.81 * sn - v / 0.9 - 0.5 * p * cs - 0.45 * cs * sn * v * v) / (1 - 0.45 * cs * cs)
    return exact - (-0.5 / 0.55 * p - 9.81 / 0.55 * q - 1 / 0.9 / 0.55 * v)


@njit
def hinf_kernel(reference, refdt, initial, kC, kD, E, E2, Q, f1, f2, f3, h):
    n = len(E)
    steps = int(round(refdt / h))
    save_every = steps
    capacity = len(reference)
    states = np.empty((capacity, n - 2))
    rr = np.empty(capacity)
    uu = np.empty(capacity)
    x = np.zeros(n)
    x[:-2] = initial
    x[-2] = reference[0]
    u = (kC @ x[4:-2])[0] + kD[0, 0] * (reference[0] - x[2])
    states[0] = x[:-2]
    rr[0] = reference[0]
    uu[0] = u
    maximum = np.abs(x[:4])
    maxu = abs(u)
    error = reference[0] - x[2]
    maxerror = abs(error)
    integral = 0.0
    sumu2 = 0.0
    t = 0.0
    previous_error = error
    previous_u = u
    firstexit = -1.0
    firstlocal = -1.0
    stop = 0
    rec = 0
    for a in range(4):
        if abs(x[a]) > BOX[a] + 1e-12:
            firstexit = 0.0
        if abs(x[a]) > LOCAL[a] + 1e-12:
            firstlocal = 0.0
    for b in range(len(reference) - 1):
        x[-2] = reference[b]
        x[-1] = (reference[b + 1] - reference[b]) / refdt
        for j in range(steps):
            v0 = nonlinear(x)
            xa = E2 @ x + Q * v0
            va = nonlinear(xa)
            xb = E2 @ x + Q * va
            vb = nonlinear(xb)
            xc = E2 @ xa + Q * (2 * vb - v0)
            vc = nonlinear(xc)
            x = E @ x + f1 * v0 + 2 * f2 * (va + vb) + f3 * vc
            t = (b * steps + j + 1) * h
            r = reference[b] + (reference[b + 1] - reference[b]) * (j + 1) / steps
            u = (kC @ x[4:-2])[0] + kD[0, 0] * (r - x[2])
            error = r - x[2]
            integral += h * 0.5 * (error * error + previous_error * previous_error)
            sumu2 += h * 0.5 * (u * u + previous_u * previous_u)
            previous_error = error
            previous_u = u
            maxerror = max(maxerror, abs(error))
            maxu = max(maxu, abs(u))
            for a in range(4):
                maximum[a] = max(maximum[a], abs(x[a]))
                if firstexit < 0 and abs(x[a]) > BOX[a] + 1e-12:
                    firstexit = t
                if firstlocal < 0 and abs(x[a]) > LOCAL[a] + 1e-12:
                    firstlocal = t
            if not np.all(np.isfinite(x)) or not np.isfinite(u):
                stop = 2
            elif (
                abs(x[2]) > 3.0
                or abs(u) > 1e8
                or abs(x[0]) > 1e8
                or abs(x[1]) > 1e10
                or abs(x[3]) > 1e5
            ):
                stop = 1
            if (j + 1) % save_every == 0 or stop:
                rec += 1
                states[rec] = x[:-2]
                rr[rec] = r
                uu[rec] = u
            if stop:
                return (
                    states[: rec + 1],
                    rr[: rec + 1],
                    uu[: rec + 1],
                    maximum,
                    maxu,
                    maxerror,
                    integral,
                    sumu2,
                    t,
                    firstexit,
                    firstlocal,
                    stop,
                )
    return (
        states[: rec + 1],
        rr[: rec + 1],
        uu[: rec + 1],
        maximum,
        maxu,
        maxerror,
        integral,
        sumu2,
        t,
        firstexit,
        firstlocal,
        stop,
    )


@njit
def features(x, modes):
    ns, n = x.shape
    degree = np.max(modes)
    h = np.ones((ns, n, degree + 1))
    h[:, :, 1] = 2 * x
    for d in range(1, degree):
        h[:, :, d + 1] = 2 * x * h[:, :, d] - 2 * d * h[:, :, d - 1]
    p = np.empty((ns, len(modes)))
    for j in range(len(modes)):
        for k in range(ns):
            v = 0.5
            for a in range(n):
                v *= h[k, a, modes[j, a]]
            p[k, j] = v
    return p


@njit(inline="always")
def actuator_rhs(x, u):
    p, p2, q, v = x
    sn, cs = np.sin(q), np.cos(q)
    return np.array(
        [
            p2,
            -10000 * p - 200 * p2 + 10000 * u,
            v,
            (-9.81 * sn - v / 0.9 - 0.5 * p * cs - 0.45 * cs * sn * v * v) / (1 - 0.45 * cs * cs),
        ]
    )


@njit
def propagate_actuator(x, u, steps=40, h=0.0005):
    x = x.copy()
    for i in range(steps):
        a = actuator_rhs(x, u)
        b = actuator_rhs(x + h * a / 2, u)
        c = actuator_rhs(x + h * b / 2, u)
        d = actuator_rhs(x + h * c, u)
        x += h * (a + 2 * b + 2 * c + d) / 6
    return x


@njit
def mpc_kernel(ref, count, x0, prev, modes, Kx, Kr, Kprev, h):
    x = x0.copy()
    xs = np.empty((count + 1, 4))
    control = np.empty(count)
    xs[0] = x
    stop = count
    code = 0
    planned_peak = abs(prev)
    steps = int(round(TS / h))
    for i in range(count):
        z = features(x.reshape(1, 4), modes)[0]
        planned = Kx @ z + Kr @ ref[i + 1 : i + 1 + len(Kr)] + Kprev * prev
        planned_peak = max(planned_peak, abs(planned))
        control[i] = prev
        x = propagate_actuator(x, prev, steps=steps, h=h)
        xs[i + 1] = x
        prev = planned
        if not np.all(np.isfinite(x)) or not np.isfinite(prev):
            code = 1
        elif abs(x[2]) > 3.0:
            code = 2
        elif max(abs(x[0]), abs(x[1]), abs(prev)) > 1e8:
            code = 3
        if code:
            stop = i + 1
            break
    return xs[: stop + 1], control[:stop], stop, code, planned_peak


@njit(inline="always")
def _acceleration(x, k):
    sn, cs, v = np.sin(x[2]), np.cos(x[2]), x[3]
    return (-k[0] * sn - k[1] * v - cs / k[2] * (x[0] + k[3] * sn * v * v)) / (1 - k[4] * cs * cs)


@njit(inline="always")
def _bs_command(x, rjet, feedback):
    # Nominal Lie derivatives.
    p, p2, q, v = x
    sn, cs = np.sin(q), np.cos(q)
    denominator = 1 - 0.45 * cs * cs
    d1 = 0.9 * cs * sn
    d2 = 0.9 * (cs * cs - sn * sn)
    numerator = -9.81 * sn - v / 0.9 - 0.5 * p * cs - 0.45 * cs * sn * v * v
    n1 = -9.81 * cs + 0.5 * p * sn - 0.45 * (cs * cs - sn * sn) * v * v
    n2 = 9.81 * sn + 0.5 * p * cs + 1.8 * cs * sn * v * v
    nv = -1 / 0.9 - 0.9 * cs * sn * v
    nqv = -0.9 * (cs * cs - sn * sn) * v
    a = numerator / denominator
    aq = n1 / denominator - numerator * d1 / denominator**2
    av = nv / denominator
    aqq = (
        n2 / denominator
        - 2 * n1 * d1 / denominator**2
        - numerator * d2 / denominator**2
        + 2 * numerator * d1 * d1 / denominator**3
    )
    aqv = nqv / denominator - nv * d1 / denominator**2
    avv = -0.9 * cs * sn / denominator
    b = -0.5 * cs / denominator
    bq = 0.5 * sn / denominator + 0.5 * cs * d1 / denominator**2
    phi3 = b * p2 + aq * v + av * a
    phi4 = (
        2 * bq * p2 * v
        + aqq * v * v
        + 2 * aqv * v * a
        + avv * a * a
        + b * (-10000 * p - 200 * p2)
        + aq * a
        + av * phi3
    )
    return (
        rjet[4]
        - phi4
        + feedback[0] * (rjet[0] - q)
        + feedback[1] * (rjet[1] - v)
        + feedback[2] * (rjet[2] - a)
        + feedback[3] * (rjet[3] - phi3)
    ) / (10000 * b)


@njit(inline="always")
def _reference_jet(tau, slope, delta, r, w, out):
    # Exact dirty derivatives for the current linear reference interval.
    # The four low-pass states are driven by the constant reference slope.
    z = w * tau
    e = np.exp(-z)
    u1 = e * delta[0]
    u2 = e * (delta[1] + z * delta[0])
    u3 = e * (delta[2] + z * delta[1] + 0.5 * z * z * delta[0])
    u4 = e * (delta[3] + z * delta[2] + 0.5 * z * z * delta[1] + z * z * z / 6 * delta[0])
    out[0] = r
    out[1] = slope + u1
    out[2] = w * (u1 - u2)
    out[3] = w * w * (u1 - 2 * u2 + u3)
    out[4] = w * w * w * (u1 - 3 * u2 + 3 * u3 - u4)


@njit(inline="always")
def bs_rhs(x, jet, coeff, feedback):
    command = _bs_command(x, jet, feedback)
    return np.array(
        [x[1], -10000 * x[0] - 200 * x[1] + 10000 * command, x[3], _acceleration(x, coeff)]
    )


@njit(inline="always")
def filter_values(tau, slope, delta, w):
    z = w * tau
    e = np.exp(-z)
    return np.array(
        [
            slope + e * delta[0],
            slope + e * (delta[1] + z * delta[0]),
            slope + e * (delta[2] + z * delta[1] + 0.5 * z * z * delta[0]),
            slope
            + e * (delta[3] + z * delta[2] + 0.5 * z * z * delta[1] + z * z * z * delta[0] / 6),
        ]
    )


@njit
def bs_kernel(rk, coeff, step, output, w, feedback):
    width = 0.02
    count = int(round(width / step))
    h = width / count
    save_every = int(round(output / h))
    duration = (len(rk) - 1) * width
    nout = int(round(duration / output)) + 1
    xx = np.zeros((nout, 4))
    ff = np.zeros((nout, 4))
    jj = np.zeros((nout, 5))
    rr = np.zeros(nout)
    uu = np.zeros(nout)
    x = np.zeros(4)
    filters = np.zeros(4)
    jet0 = np.empty(5)
    jetm = np.empty(5)
    jete = np.empty(5)
    maxima = np.zeros(4)
    maxu = 0.0
    maxerror = abs(rk[0])
    integ = 0.0
    previous_error = rk[0]
    first_O = -1.0
    first_L = -1.0
    rec = 0
    code = 0
    t = 0.0
    rr[0] = rk[0]
    _reference_jet(0.0, 0.0, filters, rk[0], w, jet0)
    jj[0] = jet0
    uu[0] = _bs_command(x, jet0, feedback)
    for block in range(len(rk) - 1):
        slope = (rk[block + 1] - rk[block]) / width
        delta = filters - slope
        for j in range(count):
            tau = j * h
            r0 = rk[block] + (rk[block + 1] - rk[block]) * j / count
            rm = rk[block] + (rk[block + 1] - rk[block]) * (j + 0.5) / count
            re = rk[block] + (rk[block + 1] - rk[block]) * (j + 1) / count
            _reference_jet(tau, slope, delta, r0, w, jet0)
            _reference_jet(tau + 0.5 * h, slope, delta, rm, w, jetm)
            _reference_jet(tau + h, slope, delta, re, w, jete)
            k1 = bs_rhs(x, jet0, coeff, feedback)
            k2 = bs_rhs(x + 0.5 * h * k1, jetm, coeff, feedback)
            k3 = bs_rhs(x + 0.5 * h * k2, jetm, coeff, feedback)
            k4 = bs_rhs(x + h * k3, jete, coeff, feedback)
            x += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t = (block * count + j + 1) * h
            command = _bs_command(x, jete, feedback)
            error = re - x[2]
            integ += h * 0.5 * (error * error + previous_error * previous_error)
            previous_error = error
            maxu = max(maxu, abs(command))
            maxerror = max(maxerror, abs(error))
            for a in range(4):
                maxima[a] = max(maxima[a], abs(x[a]))
                if first_O < 0 and abs(x[a]) > BOX[a] + 1e-12:
                    first_O = t
                if first_L < 0 and abs(x[a]) > LOCAL[a] + 1e-12:
                    first_L = t
            if not np.all(np.isfinite(x)) or not np.isfinite(command):
                code = 1
            elif abs(x[2]) > 3.0 or abs(command) > 1e8 or max(abs(x[0]), abs(x[1])) > 1e8:
                code = 2
            if (j + 1) % save_every == 0 or code:
                rec += 1
                xx[rec] = x
                ff[rec] = filter_values(tau + h, slope, delta, w)
                jj[rec] = jete
                rr[rec] = re
                uu[rec] = command
            if code:
                return (
                    xx[: rec + 1],
                    ff[: rec + 1],
                    jj[: rec + 1],
                    rr[: rec + 1],
                    uu[: rec + 1],
                    maxima,
                    maxu,
                    maxerror,
                    integ,
                    t,
                    first_O,
                    first_L,
                    code,
                )
        filters = filter_values(width, slope, delta, w)
    return (
        xx[: rec + 1],
        ff[: rec + 1],
        jj[: rec + 1],
        rr[: rec + 1],
        uu[: rec + 1],
        maxima,
        maxu,
        maxerror,
        integ,
        t,
        first_O,
        first_L,
        code,
    )


def mpc_gains(K, H=200, R=1e-6):
    N = len(K) - 1
    A, B = K[:N, :N], K[:N, -1]
    C = np.eye(N)[2]
    F, mark = [], []
    for i in range(H):
        mark.append(float(C @ B))
        C = C @ A
        F.append(C.copy())
    F, mark = np.array(F), np.array(mark)
    T = np.zeros((H, H))
    for i in range(H):
        T[i, : i + 1] = mark[i::-1]
    Q = np.full(H, 10.0)
    Q[-1] = 100.0
    Tf = T[:, 1:]
    Hess = Tf.T @ (Q[:, None] * Tf) + R * np.eye(H - 1)
    Kplan = cho_solve(cho_factor(Hess), (Tf.T * Q))
    return -(Kplan @ F)[0], Kplan[0], -(Kplan @ T[:, 0])[0]
