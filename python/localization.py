from __future__ import annotations

from fractions import Fraction as F
from math import comb, sqrt, nextafter, inf
import math
import sys
import time
import numpy as np
import mpmath as mp
iv = mp.iv

def _require(condition, message):
    # Root, stability and containment checks must survive Python's -O option.
    if not condition:
        raise ValueError(message)

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

def add(z, w):
    return (z[0] + w[0], z[1] + w[1])

def mul(z, w):
    return (z[0] * w[0] - z[1] * w[1], z[0] * w[1] + z[1] * w[0])

def modulus(z, upper=True):
    square = z[0] ** 2 + z[1] ** 2
    if not square:
        return F(0)
    value = sqrt(float(square)) * (1 + 1e-14 if upper else 1 - 1e-14)
    bound = F(value)
    while bound ** 2 < square if upper else bound ** 2 > square:
        value = nextafter(value, inf if upper else 0.0)
        bound = F(value)
    return bound

def directed_float(value, upper=True):
    result = float(value)
    if F(result) < value if upper else F(result) > value:
        result = nextafter(result, inf if upper else -inf)
    return result

def shifted(poly, center):
    powers = [(F(1), F(0))]
    for _ in range(len(poly)):
        powers.append(mul(powers[-1], center))
    result = []
    for k in range(len(poly)):
        z = (F(0), F(0))
        for j in range(k, len(poly)):
            coefficient = poly[j] * comb(j, k)
            z = add(z, (coefficient * powers[j - k][0], coefficient * powers[j - k][1]))
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
        raise ValueError('Exact characteristic-polynomial identity failed')
    return (coefficients[::-1], [[a[::-1] for a in row] for row in numerators])

def configure_precision():
    mp.mp.dps = 100
    iv.dps = 45

def qmp(x):
    return mp.mpf(x.numerator) / x.denominator

def cmul(z, w):
    return (z[0] * w[0] - z[1] * w[1], z[0] * w[1] + z[1] * w[0])

def cdiv(z, w):
    d = w[0] * w[0] + w[1] * w[1]
    return ((z[0] * w[0] + z[1] * w[1]) / d, (z[1] * w[0] - z[0] * w[1]) / d)

def csub(z, w):
    return (z[0] - w[0], z[1] - w[1])

def upper(x):
    return math.nextafter(float(x.b), math.inf)

def lower(x):
    return math.nextafter(float(x.a), -math.inf)

def ivf(x):
    return iv.mpf(x.numerator) / x.denominator

def _enclose(ac, bc):
    mp.mp.dps = 160
    iv.dps = 65
    print('Constructing exact transfer polynomials for 11 input columns', flush=True)
    da, na = exact_transfer(ac, bc)
    dm = [qmp(v) for v in da]
    der = [(j + 1) * da[j + 1] for j in range(len(da) - 1)]
    scale = max(abs(v) for v in dm)
    poly = lambda z: mp.polyval(dm[::-1], z) / scale
    mat = mp.matrix([[qmp(v) for v in row] for row in ac])
    centers = []
    print('Locating candidate poles at 160 digits', flush=True)
    for pole in mp.eig(mat, left=False, right=False):
        root = mp.findroot(poly, (pole, pole * mp.mpf('1.00000001')),
                           tol=mp.mpf('1e-130'), maxsteps=200, verify=False)
        centers.append((F(mp.nstr(root.real, 95)), F(mp.nstr(root.imag, 95))))
    radius = F('1e-55')
    residues, errors, records = [], [], []
    for j, center in enumerate(centers):
        sh = shifted(da, center)
        linear = modulus(sh[1], False) * radius
        other = modulus(sh[0]) + sum((modulus(sh[v]) * radius**v
                                        for v in range(2, len(sh))), F(0))
        if linear <= other or center[0] + radius >= 0:
            raise ValueError('An exact stable Rouche disk check failed')
        if any((center[0]-old[0])**2 + (center[1]-old[1])**2 <= (2*radius)**2
               for old in centers[:j]):
            raise ValueError('Candidate pole disks overlap')
        ds = shifted(der, center)
        dvar = sum((modulus(ds[v]) * radius**v for v in range(1, len(ds))), F(0))
        dl = modulus(ds[0], False) - dvar
        if dl <= 0:
            raise ValueError('Denominator derivative enclosure reaches zero')
        rr, ee = [], []
        for i in range(4):
            rows, errs = [], []
            for column in range(bc.shape[1]):
                ns = shifted(na[i][column], center)
                rv = cdiv(ns[0], ds[0])
                rf = (F(float(rv[0])), F(float(rv[1])))
                nv = sum((modulus(ns[v]) * radius**v for v in range(1, len(ns))), F(0))
                error = ((nv + modulus(rv)*dvar)/dl
                         + modulus(csub(rf, rv)))
                # Compact dyadic upper enclosure, checked against the full exact fraction.
                rounded = F(directed_float(error))
                if rounded < error:
                    raise ValueError('Residue error was rounded in the wrong direction')
                rows.append(rf)
                errs.append(rounded)
            rr.append(rows)
            ee.append(errs)
        residues.append(rr)
        errors.append(ee)
        records.append(dict(center_exact=[str(v) for v in center], radius=str(radius),
                            rouche_ratio_upper=directed_float(other/linear),
                            real_upper=directed_float(center[0]+radius)))
    if len(records) != len(ac):
        raise ValueError('The pole disks do not cover the characteristic degree')
    data = dict(centers=centers, radius=radius, residues=residues,
                errors=errors, records=records)
    # Reference DC/moment values from exact polynomial derivatives, not residue sums.
    dc = [na[i][4][0]/da[0] for i in range(4)]
    moment = [(na[i][4][0]*da[1]-na[i][4][1]*da[0])/da[0]**2 for i in range(4)]
    exact = dict(characteristic=[str(v) for v in da],
                 reference_DC_exact=[str(v) for v in dc],
                 reference_first_moment_exact=[str(v) for v in moment])
    return data, exact


def build_kernels(model, controller):
    a, b, c = rat(model['A']), rat(model['B']).reshape(8, 1), rat(model['C'])[:1]
    ak, bk, ck, dk = (rat(controller[key]) for key in 'ABCD')
    _require(a.shape == (8, 8) and ak.shape == (14, 14),
             'Expected eight cubic lift states and the fourteen-state controller')
    ac = np.block([[a-b@dk@c,b@ck],[-bk@c,ak]])
    inputs = zeros(22, 11)
    inputs[:8,:4] = rat(model['E'])@rat(model['D_closure'])
    inputs[:8,4:5] = b@dk
    inputs[8:,4:5] = bk
    inputs[[2,5,7],5] = [F(1),F(-2),F(-6)]
    inputs[[3,4,6],6] = [F(1),F(-6),F(-2)]
    for row,column in [(4,7),(5,8),(6,9),(7,10)]:
        inputs[row,column] = F(4)
    data,exact = _enclose(ac,inputs)
    data['exact_transfers'] = exact
    data['metadata'] = dict(initial_coefficient_bounds_exact=[str(v) for v in
        [F('1e-7'),F('.002'),F('.002')**3,F('1e-7')*F('.002')**2,
         F('1e-7')**2*F('.002'),F('1e-7')**3]])
    return data


class Kernel:

    def __init__(self, data, i, l, derivative=0, tail=False):
        self.terms = []
        self.gap_l1 = F(0)
        self.gap_sup = F(0)
        self.minimum_decay = min((-z[0] for z in data['centers']))
        for cc, rr, er in zip(data['centers'], data['residues'], data['errors']):
            r = rr[i][l]
            rad = data['radius']
            re = er[i][l]
            if tail:
                r = cdiv((-r[0], -r[1]), cc)
                re = re / (modulus(cc, False) - rad) + modulus(rr[i][l]) * rad / (modulus(cc, False) * (modulus(cc, False) - rad))
            for _ in range(derivative):
                re = re * (modulus(cc) + rad) + modulus(r) * rad
                r = cmul(r, cc)
            a = -cc[0] - rad
            self.gap_l1 += re / a + modulus(r) * rad / a ** 2
            self.gap_sup += re + modulus(r) * rad / a
            z = iv.mpc(ivf(cc[0]), ivf(cc[1]))
            rv = iv.mpc(ivf(r[0]), ivf(r[1]))
            self.terms.append((z, rv))

    def evaluate(self, t, d=0):
        tt = iv.mpf(t)
        return sum((r * z ** d * iv.exp(z * tt) for z, r in self.terms), iv.mpc(0)).real

    def primitive(self, t):
        tt = iv.mpf(t)
        return sum((r / z * iv.exp(z * tt) for z, r in self.terms), iv.mpc(0)).real

    def deriv_bound(self, left, d):
        return sum((abs(r) * abs(z) ** d * iv.exp(z.real * left) for z, r in self.terms), iv.mpf(0))

    def sup(self, T=32, rtol=2e-05):
        grid = np.unique(np.r_[0, np.geomspace(1e-07, 0.1, 70), np.arange(0.1, T + 0.1, 0.1)])
        stack = list(zip(grid[:-1], grid[1:]))
        lo = 0.0
        hi = 0.0
        count = 0
        while stack:
            a, b = stack.pop()
            mid = iv.mpf(a) / 2 + iv.mpf(b) / 2
            hh = (iv.mpf(b) - iv.mpf(a)) / 2
            vals = [self.evaluate(mid, d) for d in range(5)]
            lo = max(lo, lower(abs(vals[0])))
            variation = sum((abs(vals[d]) * hh ** d / math.factorial(d) for d in range(1, 5)), iv.mpf(0)) + self.deriv_bound(a, 5) * hh ** 5 / 120
            ub = upper(abs(vals[0]) + variation)
            count += 1
            if ub <= lo * (1 + rtol) + 1e-09 or b - a < 1e-12:
                hi = max(hi, ub)
            else:
                m = (a + b) / 2
                stack.extend([(a, m), (m, b)])
        return dict(sup=math.nextafter(max(hi, upper(self.deriv_bound(T, 0))) + directed_float(self.gap_sup), math.inf), cells=count)

    def bound(self, T=32, tol=1e-07):
        grid = np.unique(np.r_[0, np.geomspace(1e-07, 0.1, 80), np.arange(0.1, T + 0.1, 0.1)])
        total = iv.mpf(0)
        peak = iv.mpf(0)
        count = 0
        stack = list(zip(grid[:-1], grid[1:]))
        while stack:
            a, b = stack.pop()
            mid = iv.mpf(a) / 2 + iv.mpf(b) / 2
            hh = (iv.mpf(b) - iv.mpf(a)) / 2
            vals = [self.evaluate(mid, d) for d in range(5)]
            variation = sum((abs(vals[d]) * hh ** d / math.factorial(d) for d in range(1, 5)), iv.mpf(0)) + self.deriv_bound(a, 5) * hh ** 5 / 120
            enc = vals[0] + iv.mpf([-1, 1]) * variation
            dt = iv.mpf(b) - iv.mpf(a)
            enup = abs(enc)
            count += 1
            if lower(enc) >= 0 or upper(enc) <= 0:
                total += abs(self.primitive(b) - self.primitive(a))
            elif upper(enup * dt) <= tol * (b - a) / T or b - a < 1e-11:
                total += enup * dt
            else:
                m = (a + b) / 2
                stack.extend([(a, m), (m, b)])
                continue
            if upper(enup) > upper(peak):
                peak = enup
        tail = self.deriv_bound(T, 0) / ivf(self.minimum_decay)
        total += tail + ivf(self.gap_l1)
        return dict(l1=upper(total), sup_coarse=upper(peak + ivf(self.gap_sup)), cells=count, gap_l1=directed_float(self.gap_l1), tail=upper(tail))

def verify_reference(t, r):
    """Interpret the supplied values at exact uniform knots t_k=k/50."""
    t, r = np.asarray(t, float), np.asarray(r, float)
    _require(t.ndim == 1 and r.ndim == 1 and len(t) == len(r) and len(t) >= 2
             and np.all(np.isfinite(t)) and np.all(np.isfinite(r)) and t[0] == 0,
             'The reference needs finite matching knot arrays starting at t=0')
    canonical = np.arange(len(t), dtype=float)/50
    tolerance = 8*np.spacing(np.maximum(1.,canonical))
    _require(np.all(np.abs(t-canonical) <= tolerance),
             'The reference must use the uniform 0.02-second knot grid')
    rs = [F(float(x)) for x in r]
    h = F('.02')
    slopes = [(rs[i+1]-rs[i])/h for i in range(len(r)-1)]
    previous = F(0)
    if rs[0] != 0 or max(map(abs,rs)) > F('.682'):
        raise ValueError('The reference violates its amplitude or initial-value constraint.')
    for slope in slopes:
        if abs(slope) > F('.136') or abs(slope-previous) > F('.00078'):
            raise ValueError('The reference violates a slope or slope-jump constraint.')
        previous = slope
    if abs(slopes[-1]) > F('.00078'):
        raise ValueError('The final reference slope violates the ordinary jump bound.')
    return canonical


class IntervalBundle:
    def __init__(self, data, output, columns, backend, weights=None):
        self.backend = backend
        self.iv = backend.iv
        self.kernels = [backend.Kernel(data, output, column) for column in columns]
        self.weights = [F(1)]*len(columns) if weights is None else [F(v) for v in weights]
        self.z = [term[0] for term in self.kernels[0].terms]
        self.r = [[kernel.terms[j][1]*backend.ivf(w)
                   for j in range(len(self.z))]
                  for kernel,w in zip(self.kernels,self.weights)]
        self.power = [[z**d/math.factorial(d) for z in self.z] for d in range(5)]
        self.fifth = [[abs(r)*abs(z)**5 for z,r in zip(self.z,row)] for row in self.r]
        self.gap_l1 = sum((w*k.gap_l1 for w,k in zip(self.weights,self.kernels)),F(0))
        self.gap_sup = sum((w*k.gap_sup for w,k in zip(self.weights,self.kernels)),F(0))
        self.primitive_cache = {}

    def coefficients(self, midpoint, order=4):
        iv = self.iv
        exps = [iv.exp(z*midpoint) for z in self.z]
        return [[sum((r*e*p for r,e,p in zip(row,exps,self.power[d])),iv.mpc(0)).real
                 for row in self.r] for d in range(order+1)]

    def fifth_bound(self, left):
        exps = [self.iv.exp(z.real*left) for z in self.z]
        return [sum((a*e for a,e in zip(row,exps)),self.iv.mpf(0)) for row in self.fifth]

    def primitive(self, point):
        key = F(point)
        if key not in self.primitive_cache:
            t = self.backend.ivf(key)
            exps = [self.iv.exp(z*t) for z in self.z]
            self.primitive_cache[key] = [sum((r*e/z for r,e,z in zip(row,exps,self.z)),self.iv.mpc(0)).real
                                         for row in self.r]
        return self.primitive_cache[key]

    def norm(self, values):
        return self.iv.sqrt(sum((abs(x)**2 for x in values),self.iv.mpf(0)))

    def cell(self, left, right):
        """Cauchy upper bound from the integrated squared Taylor polynomial."""
        iv = self.iv
        a,b = F(left),F(right)
        midpoint = self.backend.ivf((a+b)/2)
        half = self.backend.ivf((b-a)/2)
        width = self.backend.ivf(b-a)
        coefficients = self.coefficients(midpoint)
        fifth = self.fifth_bound(self.backend.ivf(a))
        polynomial_energy = iv.mpf(0)
        for i in range(5):
            for j in range(i,5):
                if (i+j)%2:
                    continue
                dot = sum((x*y for x,y in zip(coefficients[i],coefficients[j])),iv.mpf(0))
                polynomial_energy += dot*(1 if i==j else 2)*2*half**(i+j+1)/(i+j+1)
        polynomial_sup = sum((self.norm(row)*half**d for d,row in enumerate(coefficients)),iv.mpf(0))
        remainder_coefficient = self.norm(fifth)/math.factorial(5)
        remainder_l1 = remainder_coefficient*2*half**6/6
        remainder_l2_squared = remainder_coefficient**2*2*half**11/11
        energy = polynomial_energy+2*polynomial_sup*remainder_l1+remainder_l2_squared
        if self.backend.upper(energy)<0:
            raise ValueError('Squared-kernel energy enclosure is negative.')
        # Replacing an uncertain lower endpoint by zero only widens the enclosure.
        energy = iv.mpf([0,self.backend.upper(energy)])
        upper = iv.sqrt(width*energy)
        integral = [y-x for x,y in zip(self.primitive(a),self.primitive(b))]
        lower = self.norm(integral)
        return upper,lower

    def tail_integral(self, horizon):
        t = self.backend.ivf(F(horizon))
        return sum((abs(r)*self.iv.exp(z.real*t)/(-z.real)
                    for row in self.r for z,r in zip(self.z,row)),self.iv.mpf(0))

    def tail_sup(self, horizon):
        t = self.backend.ivf(F(horizon))
        return sum((abs(r)*self.iv.exp(z.real*t)
                    for row in self.r for z,r in zip(self.z,row)),self.iv.mpf(0))


def time_grid(horizon):
    return np.unique(np.r_[0,np.geomspace(1e-8,.1,60),np.arange(.1,2,.05),
                            np.arange(2,8,.2),np.arange(8,float(horizon),.5),float(horizon)])


def bound_vector_l1(data, output, columns, backend, horizon=16, rtol=1e-3,
                    atol=1e-6, max_cells=150000, progress=None):
    """Bound integral_0^infinity ||h exp(A t) B_columns||_2 dt."""
    started=time.monotonic()
    bundle=IntervalBundle(data,output,columns,backend)
    grid=time_grid(horizon)
    stack=list(zip(grid[:-1],grid[1:]))
    total,lower_total=backend.iv.mpf(0),backend.iv.mpf(0)
    accepted=0
    evaluated=0
    while stack:
        a,b=stack.pop()
        upper,lower=bundle.cell(a,b)
        ub,lb=backend.upper(upper),max(0.,backend.lower(lower))
        evaluated+=1
        if ub-lb<=rtol*lb+atol*(b-a)/horizon:
            total+=upper
            lower_total+=backend.iv.mpf([lb,lb])
            accepted+=1
        else:
            mid=(F(a)+F(b))/2
            stack.extend([(a,mid),(mid,b)])
        if evaluated>max_cells:
            raise RuntimeError('Vector convolution interval subdivision exceeded its cell budget.')
        if progress and evaluated%1000==0:
            progress(dict(output=output,evaluated=evaluated,accepted=accepted,pending=len(stack)))
    tail=bundle.tail_integral(horizon)
    result=total+tail+backend.ivf(bundle.gap_l1)
    return dict(verified=True,upper=backend.upper(result),nominal_finite_upper=backend.upper(total),
        nominal_finite_lower=backend.lower(lower_total),tail_upper=backend.upper(tail),
        modal_gap_l1_upper=backend.directed_float(bundle.gap_l1),horizon=horizon,
        evaluated_cells=evaluated,accepted_cells=accepted,rtol=rtol,atol=atol,
        seconds=time.monotonic()-started,
        method='Interval Taylor degree4 with degree5 remainder; cellwise Cauchy-Schwarz; exponential tail and modal gap.')


def bound_initial_sup(data,output,columns,coefficients,backend,horizon=16,
                      atol=5e-4,max_cells=100000,progress=None):
    """Bound the supremum of the coherent polynomial initial-state envelope."""
    started=time.monotonic()
    bundle=IntervalBundle(data,output,columns,backend,weights=coefficients)
    grid=time_grid(horizon)
    # A numerical search proposes one point; its interval value supplies only
    # a validated lower anchor for adaptive subdivision, never an upper bound.
    poles=np.asarray([complex(float(c[0]),float(c[1])) for c in data['centers']])
    residues=np.asarray([[complex(float(r[output][j][0]),float(r[output][j][1]))*float(w)
                         for j,w in zip(columns,coefficients)] for r in data['residues']])
    trial=np.unique(np.r_[grid,np.geomspace(1e-8,.1,500),np.arange(.1,4,.002)])
    values=np.abs(np.real(np.exp(np.outer(trial,poles))@residues)).sum(axis=1)
    proposal=float(trial[int(np.argmax(values))])
    anchor=bundle.coefficients(backend.ivf(F(proposal)),order=0)[0]
    lower=max(0.,backend.lower(sum((abs(v) for v in anchor),backend.iv.mpf(0))))
    stack=list(zip(grid[:-1],grid[1:]))
    upper_global=backend.upper(bundle.tail_sup(horizon))
    evaluated=0
    while stack:
        a,b=stack.pop()
        midpoint=(F(a)+F(b))/2
        half=backend.ivf((F(b)-F(a))/2)
        coefficients_iv=bundle.coefficients(backend.ivf(midpoint))
        fifth=bundle.fifth_bound(backend.ivf(F(a)))
        center=sum((abs(v) for v in coefficients_iv[0]),backend.iv.mpf(0))
        variation=sum((abs(v)*half**d for d,row in enumerate(coefficients_iv[1:],start=1)
                       for v in row),backend.iv.mpf(0))
        variation+=sum(fifth,backend.iv.mpf(0))*half**5/math.factorial(5)
        ub=backend.upper(center+variation)
        lower=max(lower,backend.lower(center))
        evaluated+=1
        if ub<=lower+atol:
            upper_global=max(upper_global,ub)
        else:
            stack.extend([(a,midpoint),(midpoint,b)])
        if evaluated>max_cells:
            raise RuntimeError('Initial-envelope subdivision exceeded its cell budget.')
        if progress and evaluated%1000==0:
            progress(dict(output=output,evaluated=evaluated,pending=len(stack),lower=lower))
    bound=F(float(upper_global))+bundle.gap_sup
    return dict(verified=True,upper=backend.directed_float(bound),nominal_sup_upper=upper_global,
        nominal_lower_anchor=lower,modal_gap_sup_upper=backend.directed_float(bundle.gap_sup),
        horizon=horizon,evaluated_cells=evaluated,atol=atol,seconds=time.monotonic()-started,
        method='Interval supremum of sum_j coefficient_bound_j |h exp(A t) initial_column_j|; coherent polynomial initialization.')

loc = sys.modules[__name__]

def down(x):
    return np.nextafter(np.asarray(x, float), -np.inf)


def up(x):
    return np.nextafter(np.asarray(x, float), np.inf)


def imul(al, ah, bl, bh):
    vals=np.stack((al*bl, al*bh, ah*bl, ah*bh))
    return down(vals.min(axis=0)),up(vals.max(axis=0))


def positive_sum_up(values):
    """Directed pairwise reduction, including every addition explicitly."""
    q=np.asarray(values,float)
    while q.shape[-1]>1:
        pairs=q.shape[-1]//2
        v=up(q[..., :2*pairs:2]+q[..., 1:2*pairs:2])
        q=np.concatenate((v,q[..., -1:]),axis=-1) if q.shape[-1]%2 else v
    return q[...,0]


def ivparts(z):
    return loc.lower(z.real),loc.upper(z.real),loc.lower(z.imag),loc.upper(z.imag)


def certify_reference(data, phase_intervals=200, lattice_terms=800, progress=None):
    if phase_intervals<2 or lattice_terms<1:
        raise ValueError('The phase grid and lattice must be nonempty')
    loc.iv.dps=65
    h=F('.02')
    qs=[]
    qiv=[]
    ziv=[loc.iv.mpc(loc.ivf(c[0]),loc.ivf(c[1])) for c in data['centers']]
    lip=[loc.iv.mpf(0) for _ in range(4)]
    tail=[loc.iv.mpf(0) for _ in range(4)]
    gap=[loc.iv.mpf(0) for _ in range(4)]
    for i in range(4):
        qr=[]; qi=[]
        for j,c in enumerate(data['centers']):
            r=data['residues'][j][i][4]
            q=loc.cdiv(r,loc.cmul(c,c))
            qr.append(q)
            qi.append(loc.iv.mpc(loc.ivf(q[0]),loc.ivf(q[1])))
            nominal_geo=loc.iv.exp(loc.ivf(c[0]*h))
            denom=1-nominal_geo
            lip[i]+=loc.ivf(loc.modulus(loc.cdiv(r,c)))/denom
            tail[i]+=loc.ivf(loc.modulus(q))*loc.iv.exp(loc.ivf(c[0]*h*lattice_terms))/denom
            rad=data['radius']; mlo=loc.modulus(c,False); mhi=loc.modulus(c)
            if mlo<=rad or c[0]+rad>=0:
                raise ValueError('The reciprocal/stable modal enclosure is invalid')
            coeff_error=(data['errors'][j][i][4]/(mlo-rad)**2
                         +loc.modulus(r)*rad*(2*mhi+rad)/(mlo**2*(mlo-rad)**2))
            a=-c[0]-rad
            true_geo=loc.iv.exp(-loc.ivf(a*h))
            true_den=1-true_geo
            gap[i]+=(loc.ivf(coeff_error)/true_den
                     +loc.ivf(loc.modulus(q)*rad*h)/true_den**2)
        qs.append(qr);qiv.append(qi)
    if progress: progress('Enclosing the geometric modal powers')
    # Direct interval exponentials avoid wrapping growth from repeated complex
    # interval multiplication near slowly damped oscillatory poles.
    powers=np.empty((len(ziv),4,lattice_terms))
    for j,z in enumerate(ziv):
        for n in range(lattice_terms):
            powers[j,:,n]=ivparts(loc.iv.exp(z*loc.ivf(n*h)))
    if not np.all(np.isfinite(powers)):
        raise ValueError('A geometric modal enclosure overflowed')
    maxima=np.zeros(4)
    worst=np.zeros(4,dtype=int)
    nodes=[]
    if progress: progress('Enclosing all phase-node lattice sums')
    for k in range(phase_intervals+1):
        theta=F(k)*h/phase_intervals
        real_lo=np.zeros((4,lattice_terms))
        real_hi=np.zeros((4,lattice_terms))
        for j,z in enumerate(ziv):
            factor=loc.iv.exp(z*loc.ivf(theta))
            coefficients=np.array([ivparts(qiv[i][j]*factor) for i in range(4)])
            al,ah,bl,bh=(coefficients[:,c,None] for c in range(4))
            prl,prh,pil,pih=(powers[j,c][None,:] for c in range(4))
            rrlo,rrhi=imul(al,ah,prl,prh)
            iilo,iihi=imul(bl,bh,pil,pih)
            term_lo,term_hi=down(rrlo-iihi),up(rrhi-iilo)
            real_lo=down(real_lo+term_lo)
            real_hi=up(real_hi+term_hi)
        absolute=up(np.maximum(abs(real_lo),abs(real_hi)))
        sums=positive_sum_up(absolute)
        changed=sums>maxima
        maxima[changed]=sums[changed]
        worst[changed]=k
        nodes.append(sums.tolist())
    half_width=h/(2*phase_intervals)
    dc=[F(v) for v in data['exact_transfers']['reference_DC_exact']]
    moment=[F(v) for v in data['exact_transfers']['reference_first_moment_exact']]
    R,nu,J=F('.682'),F('.136'),F('.00078')
    coordinate=[]
    for i in range(4):
        node=loc.ivf(F(float(maxima[i])))
        allowance=lip[i]*loc.ivf(half_width)
        lattice=node+allowance+tail[i]+gap[i]
        bound=loc.ivf(R*abs(dc[i])+nu*abs(moment[i]))+loc.ivf(J)*lattice
        coordinate.append(dict(reference_upper=loc.upper(bound),
            exact_DC=str(dc[i]),exact_first_moment=str(moment[i]),
            max_phase_node_upper=float(maxima[i]),worst_node_index=int(worst[i]),
            worst_node_phase_exact=str(F(int(worst[i]))*h/phase_intervals),
            phase_lipschitz_upper=loc.upper(lip[i]),
            phase_allowance_upper=loc.upper(allowance),
            nominal_infinite_lattice_tail_upper=loc.upper(tail[i]),
            true_nominal_lattice_gap_upper=loc.upper(gap[i]),
            true_lattice_envelope_upper=loc.upper(lattice)))
    return dict(verified=True,kind='Uniform-grid all-time reference convolution upper bound',
        reference_upper=[q['reference_upper'] for q in coordinate],
        knot_spacing_exact=str(h),phase_intervals=phase_intervals,
        phase_nodes=phase_intervals+1,nearest_node_half_width_exact=str(half_width),
        lattice_terms=lattice_terms,nominal_tail_start_exact=str(h*lattice_terms),
        reference_amplitude_exact=str(R),slope_bound_exact=str(nu),slope_jump_bound_exact=str(J),
        coordinate_details=coordinate,phase_node_lattice_upper=nodes,
        scope='Zero-past reference, r(0)=0, exact uniform .02-second knots, amplitude<=.682, slope<=.136, slope jumps<=.00078 including initial and terminal jumps. Exact representation y=G(0)r−Q(0)rprime+Q*drprime. Every phase is covered by the nearest-node Lipschitz allowance. Infinite lattice tails and pole/residue disk discrepancies are included.')


def certify_localization(model, controller, progress=None):
    """Verify the cubic auxiliary bound for the paper's reference/initial class."""
    data = build_kernels(model,controller)
    backend = sys.modules[__name__]
    iv.dps = 45
    closure, initial = [], []
    for output in range(4):
        if progress: progress('closure'+str(output))
        closure.append(bound_vector_l1(data,output,[0,1,2,3],backend,rtol=1e-3))
    coefficients = [F(v) for v in data['metadata']['initial_coefficient_bounds_exact']]
    for output,tolerance in enumerate([5e-4,.05,1e-5,1e-5]):
        if progress: progress('initial'+str(output))
        initial.append(bound_initial_sup(data,output,list(range(5,11)),coefficients,backend,
                                         atol=tolerance))
    if progress: progress('uniform-grid reference')
    reference = certify_reference(data,progress=progress)
    cl = [F(q['upper']) for q in closure]
    ic = [F(q['upper']) for q in initial]
    rf = [F(q) for q in reference['reference_upper']]
    total = [cl[i]+ic[i]+rf[i] for i in range(4)]
    box = [F(float(v)) for v in model['bounds']]
    _require(all(total[i]<box[i] for i in range(4)),
             'The cubic auxiliary localization bounds do not close the region')
    return dict(verified=True,physical_bounds=[directed_float(q) for q in total],
        ratios=[directed_float(total[i]/box[i]) for i in range(4)],
        beta=directed_float(max(total[i]/box[i] for i in range(4))),
        reference_bounds=[directed_float(q) for q in rf],
        closure_bounds=[directed_float(q) for q in cl],
        initial_bounds=[directed_float(q) for q in ic],
        auxiliary_box=[directed_float(q) for q in box],
        auxiliary_dimension=22,auxiliary_lift_degree=3,
        auxiliary_blend=float(model['blend']),
        closure_integrals=closure,coherent_initial_suprema=initial,
        reference_certificate=reference,all_pole_disks_strictly_stable=True,
        verified_pole_disks=data['records'])
