"""Regional containment certificate for the cubic auxiliary closed loop.

The closed-loop transfer matrices are enclosed exactly (Rouche disks around
the poles, interval residues), the convolution kernels are integrated with
validated interval Taylor models, and the reference class is handled on its
exact uniform knot lattice.  Everything that decides a bound is computed with
Fractions or outward-rounded mpmath intervals; floating-point values only
propose subdivision points.
"""

from __future__ import annotations

from fractions import Fraction as F
from math import comb, factorial, inf, nextafter, sqrt
import time

import mpmath as mp
import numpy as np

from verification import require

iv = mp.iv

# Common reference class: amplitude, slope and slope-jump bounds on the exact
# uniform knot grid of KNOT_SPACING seconds.  The same values are stored in
# certificates/witnesses/common_reference_class.json.
REFERENCE_AMPLITUDE = F(5, 8)
REFERENCE_SLOPE = F(21, 200)
REFERENCE_SLOPE_JUMP = F(41, 5000)
KNOTS_PER_SECOND = 50
KNOT_SPACING = F(1, KNOTS_PER_SECOND)


def rat(x):
    a = np.asarray(x)
    return np.array([F(float(v)) for v in a.flat], dtype=object).reshape(a.shape)


def zeros(n, m=None):
    return np.full((n, n if m is None else m), F(0), dtype=object)


def eye(n):
    a = zeros(n)
    for i in range(n):
        a[i, i] = F(1)
    return a


# Complex numbers are pairs of Fractions.
def cadd(z, w):
    return z[0] + w[0], z[1] + w[1]


def cmul(z, w):
    return z[0] * w[0] - z[1] * w[1], z[0] * w[1] + z[1] * w[0]


def cdiv(z, w):
    d = w[0] * w[0] + w[1] * w[1]
    return (z[0] * w[0] + z[1] * w[1]) / d, (z[1] * w[0] - z[0] * w[1]) / d


def csub(z, w):
    return z[0] - w[0], z[1] - w[1]


def modulus(z, upper=True):
    square = z[0] ** 2 + z[1] ** 2
    if not square:
        return F(0)
    value = sqrt(float(square)) * (1 + 1e-14 if upper else 1 - 1e-14)
    bound = F(value)
    while (bound**2 < square) if upper else (bound**2 > square):
        value = nextafter(value, inf if upper else 0.0)
        bound = F(value)
    return bound


def directed_float(value, upper=True):
    result = float(value)
    if (F(result) < value) if upper else (F(result) > value):
        result = nextafter(result, inf if upper else -inf)
    return result


def shifted(poly, center):
    powers = [(F(1), F(0))]
    for _ in range(len(poly)):
        powers.append(cmul(powers[-1], center))
    result = []
    for k in range(len(poly)):
        z = (F(0), F(0))
        for j in range(k, len(poly)):
            coefficient = poly[j] * comb(j, k)
            z = cadd(z, (coefficient * powers[j - k][0], coefficient * powers[j - k][1]))
        result.append(z)
    return result


def exact_transfer(A, B):
    n = len(A)
    adj = eye(n)
    coefficients = [F(1)]
    numerators = [[[] for _ in range(B.shape[1])] for _ in range(4)]
    for k in range(1, n + 1):
        products = adj @ B
        for i in range(4):
            for j in range(B.shape[1]):
                numerators[i][j].append(products[i, j])
        product = A @ adj
        ck = -sum((product[j, j] for j in range(n)), F(0)) / k
        coefficients.append(ck)
        adj = product + ck * eye(n)
    if any((x != 0 for x in adj.flat)):
        raise ValueError("Exact characteristic-polynomial identity failed")
    return coefficients[::-1], [[a[::-1] for a in row] for row in numerators]


def qmp(x):
    return mp.mpf(x.numerator) / x.denominator


def upper(x):
    return nextafter(float(x.b), inf)


def lower(x):
    return nextafter(float(x.a), -inf)


def ivf(x):
    return iv.mpf(x.numerator) / x.denominator


def _enclose(ac, bc, progress=None):
    mp.mp.dps = 160
    iv.dps = 65
    if progress:
        progress(f"exact transfer polynomials for {bc.shape[1]} input columns")
    da, na = exact_transfer(ac, bc)
    dm = [qmp(v) for v in da]
    der = [(j + 1) * da[j + 1] for j in range(len(da) - 1)]
    scale = max(abs(v) for v in dm)

    def poly(z):
        return mp.polyval(dm[::-1], z) / scale

    mat = mp.matrix([[qmp(v) for v in row] for row in ac])
    centers = []
    if progress:
        progress("candidate poles at 160 digits")
    for pole in mp.eig(mat, left=False, right=False):
        root = mp.findroot(
            poly,
            (pole, pole * mp.mpf("1.00000001")),
            tol=mp.mpf("1e-130"),
            maxsteps=200,
            verify=False,
        )
        centers.append((F(mp.nstr(root.real, 95)), F(mp.nstr(root.imag, 95))))
    radius = F("1e-55")
    residues, errors, records = [], [], []
    for j, center in enumerate(centers):
        sh = shifted(da, center)
        linear = modulus(sh[1], False) * radius
        other = modulus(sh[0]) + sum((modulus(sh[v]) * radius**v for v in range(2, len(sh))), F(0))
        if linear <= other or center[0] + radius >= 0:
            raise ValueError("An exact stable Rouche disk check failed")
        if any(
            (center[0] - old[0]) ** 2 + (center[1] - old[1]) ** 2 <= (2 * radius) ** 2
            for old in centers[:j]
        ):
            raise ValueError("Candidate pole disks overlap")
        ds = shifted(der, center)
        dvar = sum((modulus(ds[v]) * radius**v for v in range(1, len(ds))), F(0))
        dl = modulus(ds[0], False) - dvar
        if dl <= 0:
            raise ValueError("Denominator derivative enclosure reaches zero")
        rr, ee = [], []
        for i in range(4):
            rows, errs = [], []
            for column in range(bc.shape[1]):
                ns = shifted(na[i][column], center)
                rv = cdiv(ns[0], ds[0])
                rf = (F(float(rv[0])), F(float(rv[1])))
                nv = sum((modulus(ns[v]) * radius**v for v in range(1, len(ns))), F(0))
                error = (nv + modulus(rv) * dvar) / dl + modulus(csub(rf, rv))
                # Compact dyadic upper enclosure, checked against the full exact fraction.
                rounded = F(directed_float(error))
                if rounded < error:
                    raise ValueError("Residue error was rounded in the wrong direction")
                rows.append(rf)
                errs.append(rounded)
            rr.append(rows)
            ee.append(errs)
        residues.append(rr)
        errors.append(ee)
        records.append(
            dict(
                center_exact=[str(v) for v in center],
                radius=str(radius),
                rouche_ratio_upper=directed_float(other / linear),
                real_upper=directed_float(center[0] + radius),
            )
        )
    if len(records) != len(ac):
        raise ValueError("The pole disks do not cover the characteristic degree")
    data = dict(centers=centers, radius=radius, residues=residues, errors=errors, records=records)
    # Reference DC/moment values from exact polynomial derivatives, not residue sums.
    dc = [na[i][4][0] / da[0] for i in range(4)]
    moment = [(na[i][4][0] * da[1] - na[i][4][1] * da[0]) / da[0] ** 2 for i in range(4)]
    exact = dict(
        characteristic=[str(v) for v in da],
        reference_DC_exact=[str(v) for v in dc],
        reference_first_moment_exact=[str(v) for v in moment],
    )
    return data, exact


def closed_loop_matrices(model, controller):
    """Exact closed-loop matrix and the eleven input columns.

    Columns 0-3 carry the closure channel, column 4 the physical reference r and
    columns 5-10 the coherent cubic initial-state directions.
    """
    a, b, c = rat(model["A"]), rat(model["B"]).reshape(8, 1), rat(model["C"])[:1]
    ak, bk, ck, dk = (rat(controller[key]) for key in "ABCD")
    require(
        a.shape == (8, 8) and ak.ndim == 2 and ak.shape[0] == ak.shape[1],
        "Expected eight cubic lift states and a square controller matrix",
    )
    ac = np.block([[a - b @ dk @ c, b @ ck], [-bk @ c, ak]])
    inputs = zeros(len(ac), 11)
    inputs[:8, :4] = rat(model["E"]) @ rat(model["D_closure"])
    inputs[:8, 4:5] = b @ dk
    inputs[8:, 4:5] = bk
    inputs[[2, 5, 7], 5] = [F(1), F(-2), F(-6)]
    inputs[[3, 4, 6], 6] = [F(1), F(-6), F(-2)]
    for row, column in [(4, 7), (5, 8), (6, 9), (7, 10)]:
        inputs[row, column] = F(4)
    return ac, inputs


def build_kernels(model, controller, progress=None):
    ac, inputs = closed_loop_matrices(model, controller)
    data, exact = _enclose(ac, inputs, progress)
    data["exact_transfers"] = exact
    data["metadata"] = dict(
        initial_coefficient_bounds_exact=[
            str(v)
            for v in [
                F("1e-7"),
                F(".002"),
                F(".002") ** 3,
                F("1e-7") * F(".002") ** 2,
                F("1e-7") ** 2 * F(".002"),
                F("1e-7") ** 3,
            ]
        ]
    )
    return data


class Kernel:
    """Modal expansion of one closed-loop kernel h_i exp(A t) b_l.

    Each term is (pole, residue) as mpmath complex intervals. gap_l1 and
    gap_sup bound the L1 and sup norm of the difference between the exact
    kernel and this interval modal sum, from the residue errors and the disk
    radii.
    """

    def __init__(self, data, i, column):
        self.terms = []
        self.gap_l1 = F(0)
        self.gap_sup = F(0)
        rad = data["radius"]
        for cc, rr, er in zip(data["centers"], data["residues"], data["errors"]):
            r = rr[i][column]
            re = er[i][column]
            a = -cc[0] - rad
            self.gap_l1 += re / a + modulus(r) * rad / a**2
            self.gap_sup += re + modulus(r) * rad / a
            z = iv.mpc(ivf(cc[0]), ivf(cc[1]))
            rv = iv.mpc(ivf(r[0]), ivf(r[1]))
            self.terms.append((z, rv))


def verify_reference(t, r):
    """Check a reference sequence against the common class on exact knots t_k = k/50."""
    t, r = np.asarray(t, float), np.asarray(r, float)
    require(
        t.ndim == 1
        and r.ndim == 1
        and len(t) == len(r)
        and len(t) >= 2
        and np.all(np.isfinite(t))
        and np.all(np.isfinite(r))
        and t[0] == 0,
        "The reference needs finite matching knot arrays starting at t=0",
    )
    canonical = np.arange(len(t), dtype=float) / KNOTS_PER_SECOND
    tolerance = 8 * np.spacing(np.maximum(1.0, canonical))
    require(
        np.all(np.abs(t - canonical) <= tolerance),
        "The reference must use the uniform 0.02-second knot grid",
    )
    rs = [F(float(x)) for x in r]
    slopes = [(rs[i + 1] - rs[i]) / KNOT_SPACING for i in range(len(r) - 1)]
    previous = F(0)
    if rs[0] != 0 or max(map(abs, rs)) > REFERENCE_AMPLITUDE:
        raise ValueError("The reference violates its amplitude or initial-value constraint.")
    for slope in slopes:
        if abs(slope) > REFERENCE_SLOPE or abs(slope - previous) > REFERENCE_SLOPE_JUMP:
            raise ValueError("The reference violates a slope or slope-jump constraint.")
        previous = slope
    if abs(slopes[-1]) > REFERENCE_SLOPE_JUMP:
        raise ValueError("The final reference slope violates the ordinary jump bound.")
    return canonical


class IntervalBundle:
    """Weighted vector of kernels sharing the same poles, with interval Taylor data."""

    def __init__(self, data, output, columns, weights=None):
        self.kernels = [Kernel(data, output, column) for column in columns]
        self.weights = [F(1)] * len(columns) if weights is None else [F(v) for v in weights]
        self.z = [term[0] for term in self.kernels[0].terms]
        self.r = [
            [kernel.terms[j][1] * ivf(w) for j in range(len(self.z))]
            for kernel, w in zip(self.kernels, self.weights)
        ]
        self.power = [[z**d / factorial(d) for z in self.z] for d in range(5)]
        self.fifth = [[abs(r) * abs(z) ** 5 for z, r in zip(self.z, row)] for row in self.r]
        self.gap_l1 = sum((w * k.gap_l1 for w, k in zip(self.weights, self.kernels)), F(0))
        self.gap_sup = sum((w * k.gap_sup for w, k in zip(self.weights, self.kernels)), F(0))
        self.primitive_cache = {}

    def coefficients(self, midpoint, order=4):
        exps = [iv.exp(z * midpoint) for z in self.z]
        return [
            [
                sum((r * e * p for r, e, p in zip(row, exps, self.power[d])), iv.mpc(0)).real
                for row in self.r
            ]
            for d in range(order + 1)
        ]

    def fifth_bound(self, left):
        exps = [iv.exp(z.real * left) for z in self.z]
        return [sum((a * e for a, e in zip(row, exps)), iv.mpf(0)) for row in self.fifth]

    def primitive(self, point):
        key = F(point)
        if key not in self.primitive_cache:
            t = ivf(key)
            exps = [iv.exp(z * t) for z in self.z]
            self.primitive_cache[key] = [
                sum((r * e / z for r, e, z in zip(row, exps, self.z)), iv.mpc(0)).real
                for row in self.r
            ]
        return self.primitive_cache[key]

    def norm(self, values):
        return iv.sqrt(sum((abs(x) ** 2 for x in values), iv.mpf(0)))

    def cell(self, left, right):
        """Cauchy upper bound from the integrated squared Taylor polynomial."""
        a, b = F(left), F(right)
        midpoint = ivf((a + b) / 2)
        half = ivf((b - a) / 2)
        width = ivf(b - a)
        coefficients = self.coefficients(midpoint)
        fifth = self.fifth_bound(ivf(a))
        polynomial_energy = iv.mpf(0)
        for i in range(5):
            for j in range(i, 5):
                if (i + j) % 2:
                    continue
                dot = sum((x * y for x, y in zip(coefficients[i], coefficients[j])), iv.mpf(0))
                polynomial_energy += (
                    dot * (1 if i == j else 2) * 2 * half ** (i + j + 1) / (i + j + 1)
                )
        polynomial_sup = sum(
            (self.norm(row) * half**d for d, row in enumerate(coefficients)), iv.mpf(0)
        )
        remainder_coefficient = self.norm(fifth) / factorial(5)
        remainder_l1 = remainder_coefficient * 2 * half**6 / 6
        remainder_l2_squared = remainder_coefficient**2 * 2 * half**11 / 11
        energy = polynomial_energy + 2 * polynomial_sup * remainder_l1 + remainder_l2_squared
        if upper(energy) < 0:
            raise ValueError("Squared-kernel energy enclosure is negative.")
        # Replacing an uncertain lower endpoint by zero only widens the enclosure.
        energy = iv.mpf([0, upper(energy)])
        cell_upper = iv.sqrt(width * energy)
        integral = [y - x for x, y in zip(self.primitive(a), self.primitive(b))]
        cell_lower = self.norm(integral)
        return cell_upper, cell_lower

    def tail_integral(self, horizon):
        t = ivf(F(horizon))
        return sum(
            (
                abs(r) * iv.exp(z.real * t) / (-z.real)
                for row in self.r
                for z, r in zip(self.z, row)
            ),
            iv.mpf(0),
        )

    def tail_sup(self, horizon):
        t = ivf(F(horizon))
        return sum(
            (abs(r) * iv.exp(z.real * t) for row in self.r for z, r in zip(self.z, row)),
            iv.mpf(0),
        )


def time_grid(horizon):
    return np.unique(
        np.r_[
            0,
            np.geomspace(1e-8, 0.1, 60),
            np.arange(0.1, 2, 0.05),
            np.arange(2, 8, 0.2),
            np.arange(8, float(horizon), 0.5),
            float(horizon),
        ]
    )


def bound_vector_l1(
    data,
    output,
    columns,
    horizon=16,
    rtol=1e-3,
    atol=1e-6,
    max_cells=150000,
    progress=None,
):
    """Bound integral_0^infinity ||h exp(A t) B_columns||_2 dt."""
    started = time.monotonic()
    bundle = IntervalBundle(data, output, columns)
    grid = time_grid(horizon)
    stack = list(zip(grid[:-1], grid[1:]))
    total, lower_total = iv.mpf(0), iv.mpf(0)
    accepted = 0
    evaluated = 0
    while stack:
        a, b = stack.pop()
        cell_upper, cell_lower = bundle.cell(a, b)
        ub, lb = upper(cell_upper), max(0.0, lower(cell_lower))
        evaluated += 1
        if ub - lb <= rtol * lb + atol * (b - a) / horizon:
            total += cell_upper
            lower_total += iv.mpf([lb, lb])
            accepted += 1
        else:
            mid = (F(a) + F(b)) / 2
            stack.extend([(a, mid), (mid, b)])
        if evaluated > max_cells:
            raise RuntimeError("Vector convolution interval subdivision exceeded its cell budget.")
        if progress and evaluated % 1000 == 0:
            progress(
                dict(output=output, evaluated=evaluated, accepted=accepted, pending=len(stack))
            )
    tail = bundle.tail_integral(horizon)
    result = total + tail + ivf(bundle.gap_l1)
    return dict(
        verified=True,
        upper=upper(result),
        nominal_finite_upper=upper(total),
        nominal_finite_lower=lower(lower_total),
        tail_upper=upper(tail),
        modal_gap_l1_upper=directed_float(bundle.gap_l1),
        horizon=horizon,
        evaluated_cells=evaluated,
        accepted_cells=accepted,
        rtol=rtol,
        atol=atol,
        seconds=time.monotonic() - started,
        method=(
            "Interval Taylor degree4 with degree5 remainder; cellwise Cauchy-Schwarz; "
            "exponential tail and modal gap."
        ),
    )


def bound_initial_sup(
    data,
    output,
    columns,
    coefficients,
    horizon=16,
    atol=5e-4,
    max_cells=100000,
    progress=None,
):
    """Bound the supremum of the coherent polynomial initial-state envelope."""
    started = time.monotonic()
    bundle = IntervalBundle(data, output, columns, weights=coefficients)
    grid = time_grid(horizon)
    # A numerical search proposes one point; its interval value supplies only
    # a validated lower anchor for adaptive subdivision, never an upper bound.
    poles = np.asarray([complex(float(c[0]), float(c[1])) for c in data["centers"]])
    residues = np.asarray(
        [
            [
                complex(float(r[output][j][0]), float(r[output][j][1])) * float(w)
                for j, w in zip(columns, coefficients)
            ]
            for r in data["residues"]
        ]
    )
    trial = np.unique(np.r_[grid, np.geomspace(1e-8, 0.1, 500), np.arange(0.1, 4, 0.002)])
    values = np.abs(np.real(np.exp(np.outer(trial, poles)) @ residues)).sum(axis=1)
    proposal = float(trial[int(np.argmax(values))])
    anchor = bundle.coefficients(ivf(F(proposal)), order=0)[0]
    anchor_lower = max(0.0, lower(sum((abs(v) for v in anchor), iv.mpf(0))))
    stack = list(zip(grid[:-1], grid[1:]))
    upper_global = upper(bundle.tail_sup(horizon))
    evaluated = 0
    while stack:
        a, b = stack.pop()
        midpoint = (F(a) + F(b)) / 2
        half = ivf((F(b) - F(a)) / 2)
        coefficients_iv = bundle.coefficients(ivf(midpoint))
        fifth = bundle.fifth_bound(ivf(F(a)))
        center = sum((abs(v) for v in coefficients_iv[0]), iv.mpf(0))
        variation = sum(
            (abs(v) * half**d for d, row in enumerate(coefficients_iv[1:], start=1) for v in row),
            iv.mpf(0),
        )
        variation += sum(fifth, iv.mpf(0)) * half**5 / factorial(5)
        ub = upper(center + variation)
        anchor_lower = max(anchor_lower, lower(center))
        evaluated += 1
        if ub <= anchor_lower + atol:
            upper_global = max(upper_global, ub)
        else:
            stack.extend([(a, midpoint), (midpoint, b)])
        if evaluated > max_cells:
            raise RuntimeError("Initial-envelope subdivision exceeded its cell budget.")
        if progress and evaluated % 1000 == 0:
            progress(
                dict(output=output, evaluated=evaluated, pending=len(stack), lower=anchor_lower)
            )
    bound = F(float(upper_global)) + bundle.gap_sup
    return dict(
        verified=True,
        upper=directed_float(bound),
        nominal_sup_upper=upper_global,
        nominal_lower_anchor=anchor_lower,
        modal_gap_sup_upper=directed_float(bundle.gap_sup),
        horizon=horizon,
        evaluated_cells=evaluated,
        atol=atol,
        seconds=time.monotonic() - started,
        method=(
            "Interval supremum of sum_j coefficient_bound_j |h exp(A t) initial_column_j|; "
            "coherent polynomial initialization."
        ),
    )


def down(x):
    return np.nextafter(np.asarray(x, float), -np.inf)


def up(x):
    return np.nextafter(np.asarray(x, float), np.inf)


def imul(al, ah, bl, bh):
    vals = np.stack((al * bl, al * bh, ah * bl, ah * bh))
    return down(vals.min(axis=0)), up(vals.max(axis=0))


def positive_sum_up(values):
    """Directed pairwise reduction, including every addition explicitly."""
    q = np.asarray(values, float)
    while q.shape[-1] > 1:
        pairs = q.shape[-1] // 2
        v = up(q[..., : 2 * pairs : 2] + q[..., 1 : 2 * pairs : 2])
        q = np.concatenate((v, q[..., -1:]), axis=-1) if q.shape[-1] % 2 else v
    return q[..., 0]


def ivparts(z):
    return lower(z.real), upper(z.real), lower(z.imag), upper(z.imag)


def certify_reference(data, phase_intervals=200, lattice_terms=800, progress=None):
    """Upper bound of the reference contribution on the exact uniform knot lattice."""
    if phase_intervals < 2 or lattice_terms < 1:
        raise ValueError("The phase grid and lattice must be nonempty")
    iv.dps = 65
    h = KNOT_SPACING
    qiv = []
    ziv = [iv.mpc(ivf(c[0]), ivf(c[1])) for c in data["centers"]]
    lip = [iv.mpf(0) for _ in range(4)]
    tail = [iv.mpf(0) for _ in range(4)]
    gap = [iv.mpf(0) for _ in range(4)]
    for i in range(4):
        qi = []
        for j, c in enumerate(data["centers"]):
            r = data["residues"][j][i][4]
            q = cdiv(r, cmul(c, c))
            qi.append(iv.mpc(ivf(q[0]), ivf(q[1])))
            nominal_geo = iv.exp(ivf(c[0] * h))
            denom = 1 - nominal_geo
            lip[i] += ivf(modulus(cdiv(r, c))) / denom
            tail[i] += ivf(modulus(q)) * iv.exp(ivf(c[0] * h * lattice_terms)) / denom
            rad = data["radius"]
            mlo = modulus(c, False)
            mhi = modulus(c)
            if mlo <= rad or c[0] + rad >= 0:
                raise ValueError("The reciprocal/stable modal enclosure is invalid")
            coeff_error = data["errors"][j][i][4] / (mlo - rad) ** 2 + modulus(r) * rad * (
                2 * mhi + rad
            ) / (mlo**2 * (mlo - rad) ** 2)
            a = -c[0] - rad
            true_geo = iv.exp(-ivf(a * h))
            true_den = 1 - true_geo
            gap[i] += ivf(coeff_error) / true_den + ivf(modulus(q) * rad * h) / true_den**2
        qiv.append(qi)
    if progress:
        progress("Enclosing the geometric modal powers")
    # Direct interval exponentials avoid wrapping growth from repeated complex
    # interval multiplication near slowly damped oscillatory poles.
    powers = np.empty((len(ziv), 4, lattice_terms))
    for j, z in enumerate(ziv):
        for n in range(lattice_terms):
            powers[j, :, n] = ivparts(iv.exp(z * ivf(n * h)))
    if not np.all(np.isfinite(powers)):
        raise ValueError("A geometric modal enclosure overflowed")
    maxima = np.zeros(4)
    worst = np.zeros(4, dtype=int)
    nodes = []
    if progress:
        progress("Enclosing all phase-node lattice sums")
    for k in range(phase_intervals + 1):
        theta = F(k) * h / phase_intervals
        real_lo = np.zeros((4, lattice_terms))
        real_hi = np.zeros((4, lattice_terms))
        for j, z in enumerate(ziv):
            factor = iv.exp(z * ivf(theta))
            coefficients = np.array([ivparts(qiv[i][j] * factor) for i in range(4)])
            al, ah, bl, bh = (coefficients[:, c, None] for c in range(4))
            prl, prh, pil, pih = (powers[j, c][None, :] for c in range(4))
            rrlo, rrhi = imul(al, ah, prl, prh)
            iilo, iihi = imul(bl, bh, pil, pih)
            term_lo, term_hi = down(rrlo - iihi), up(rrhi - iilo)
            real_lo = down(real_lo + term_lo)
            real_hi = up(real_hi + term_hi)
        absolute = up(np.maximum(abs(real_lo), abs(real_hi)))
        sums = positive_sum_up(absolute)
        changed = sums > maxima
        maxima[changed] = sums[changed]
        worst[changed] = k
        nodes.append(sums.tolist())
    half_width = h / (2 * phase_intervals)
    dc = [F(v) for v in data["exact_transfers"]["reference_DC_exact"]]
    moment = [F(v) for v in data["exact_transfers"]["reference_first_moment_exact"]]
    R, nu, J = REFERENCE_AMPLITUDE, REFERENCE_SLOPE, REFERENCE_SLOPE_JUMP
    coordinate = []
    for i in range(4):
        node = ivf(F(float(maxima[i])))
        allowance = lip[i] * ivf(half_width)
        lattice = node + allowance + tail[i] + gap[i]
        bound = ivf(R * abs(dc[i]) + nu * abs(moment[i])) + ivf(J) * lattice
        coordinate.append(
            dict(
                reference_upper=upper(bound),
                exact_DC=str(dc[i]),
                exact_first_moment=str(moment[i]),
                max_phase_node_upper=float(maxima[i]),
                worst_node_index=int(worst[i]),
                worst_node_phase_exact=str(F(int(worst[i])) * h / phase_intervals),
                phase_lipschitz_upper=upper(lip[i]),
                phase_allowance_upper=upper(allowance),
                nominal_infinite_lattice_tail_upper=upper(tail[i]),
                true_nominal_lattice_gap_upper=upper(gap[i]),
                true_lattice_envelope_upper=upper(lattice),
            )
        )
    return dict(
        verified=True,
        kind="Uniform-grid all-time reference convolution upper bound",
        reference_upper=[q["reference_upper"] for q in coordinate],
        knot_spacing_exact=str(h),
        phase_intervals=phase_intervals,
        phase_nodes=phase_intervals + 1,
        nearest_node_half_width_exact=str(half_width),
        lattice_terms=lattice_terms,
        nominal_tail_start_exact=str(h * lattice_terms),
        reference_amplitude_exact=str(R),
        slope_bound_exact=str(nu),
        slope_jump_bound_exact=str(J),
        coordinate_details=coordinate,
        phase_node_lattice_upper=nodes,
        scope=(
            f"Zero-past reference, r(0)=0, exact uniform {float(h)}-second knots, "
            f"amplitude<={float(R)}, slope<={float(nu)}, slope jumps<={float(J)} including "
            "initial and terminal jumps. Exact representation y=G(0)r-Q(0)r'+Q*dr'. "
            "Every phase is covered by the nearest-node Lipschitz allowance. Infinite "
            "lattice tails and pole/residue disk discrepancies are included."
        ),
    )


def tracking_error_bound(data, closure_bound, initial_bound, lattice_bound):
    """Bound r-x1 using the auxiliary angle transfer and the same knot class.

    For G: r -> x1, m1=-G'(0) and Q=(G-G(0)+m1*s)/s**2 give
    e_ref=(1-G(0))*r+m1*r'-Q*dr'. The lattice envelope of Q and
    the angle closure/initial envelopes therefore also bound the error.
    """
    dc = F(data["exact_transfers"]["reference_DC_exact"][2])
    moment = F(data["exact_transfers"]["reference_first_moment_exact"][2])
    closure, initial, lattice = map(F, (closure_bound, initial_bound, lattice_bound))
    require(min(closure, initial, lattice) >= 0, "Tracking envelopes must be nonnegative")
    reference = (
        REFERENCE_AMPLITUDE * abs(1 - dc)
        + REFERENCE_SLOPE * abs(moment)
        + REFERENCE_SLOPE_JUMP * lattice
    )
    total = reference + closure + initial
    return dict(
        readout="r-x1",
        upper=directed_float(total),
        total_exact=str(total),
        reference_upper=directed_float(reference),
        reference_exact=str(reference),
        closure_upper=directed_float(closure),
        closure_exact=str(closure),
        initial_upper=directed_float(initial),
        initial_exact=str(initial),
        error_DC_exact=str(1 - dc),
        error_first_moment_exact=str(-moment),
        slope_jump_coefficient_upper_exact=str(lattice),
    )


def certify_localization(model, controller, progress=None):
    """Verify the cubic auxiliary bound for the paper's reference/initial class."""
    data = build_kernels(model, controller, progress)
    iv.dps = 45
    closure, initial = [], []
    for output in range(4):
        if progress:
            progress("closure" + str(output))
        closure.append(bound_vector_l1(data, output, [0, 1, 2, 3], rtol=1e-3))
    coefficients = [F(v) for v in data["metadata"]["initial_coefficient_bounds_exact"]]
    for output, tolerance in enumerate([5e-4, 0.05, 1e-5, 1e-5]):
        if progress:
            progress("initial" + str(output))
        initial.append(
            bound_initial_sup(data, output, list(range(5, 11)), coefficients, atol=tolerance)
        )
    if progress:
        progress("uniform-grid reference")
    reference = certify_reference(data, progress=progress)
    cl = [F(q["upper"]) for q in closure]
    ic = [F(q["upper"]) for q in initial]
    rf = [F(q) for q in reference["reference_upper"]]
    total = [cl[i] + ic[i] + rf[i] for i in range(4)]
    box = [F(float(v)) for v in model["bounds"]]
    require(
        all(total[i] < box[i] for i in range(4)),
        "The cubic auxiliary localization bounds do not close the region",
    )
    error = tracking_error_bound(
        data, cl[2], ic[2],
        F(reference["coordinate_details"][2]["true_lattice_envelope_upper"]),
    )
    error["all_time"] = True
    return dict(
        verified=True,
        physical_bounds=[directed_float(q) for q in total],
        ratios=[directed_float(total[i] / box[i]) for i in range(4)],
        beta=directed_float(max(total[i] / box[i] for i in range(4))),
        reference_bounds=[directed_float(q) for q in rf],
        closure_bounds=[directed_float(q) for q in cl],
        initial_bounds=[directed_float(q) for q in ic],
        auxiliary_box=[directed_float(q) for q in box],
        auxiliary_dimension=8 + len(controller["A"]),
        auxiliary_lift_degree=3,
        auxiliary_blend=float(model["blend"]),
        closure_integrals=closure,
        coherent_initial_suprema=initial,
        reference_certificate=reference,
        tracking_error=error,
        all_pole_disks_strictly_stable=True,
        verified_pole_disks=data["records"],
    )
