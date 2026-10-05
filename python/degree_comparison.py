from pathlib import Path
import json
import os
import tempfile
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'galerkin-degree-mpl'))
import numpy as np
from numpy.polynomial.hermite import hermval, hermder
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from numba import njit
from design import galerkin_model

ROOT = Path(__file__).resolve().parent
DATA = dict(np.load(ROOT / 'paper_data.npz', allow_pickle=False))
BOX = DATA['plant_bounds']

def indices(degree):
    return [(1,0),(0,1)] + [(a,d-a) for d in range(3,degree+1,2) for a in range(d+1)]

def herm(n, x, derivative=False):
    c = np.zeros(n+1)
    c[n] = 1
    return hermval(x, hermder(c) if derivative else c)

def lift(x, modes):
    x = np.atleast_2d(x)
    q,v = x[:,2:].T
    return np.column_stack([x[:,:2]] + [.5*herm(a,q)*herm(b,v) for a,b in modes])

def acceleration(x):
    x = np.atleast_2d(x)
    p,_,q,v = x.T
    s,c = np.sin(q),np.cos(q)
    return (-9.81*s-v/.9-.5*p*c-.45*c*s*v*v)/(1-.45*c*c)

def drift(x, modes):
    x = np.atleast_2d(x)
    q,v = x[:,2:].T
    a = acceleration(x)
    return np.column_stack([.5*(herm(i,q,True)*herm(j,v)*v + herm(i,q)*herm(j,v,True)*a) for i,j in modes])

def projection_tensor_reference(degree, qorder=64):
    """Full tensor quadrature baseline for the degree study."""
    modes = indices(degree)
    grid, weights = [], []
    for b,n in zip(BOX[[0,2,3]], (4,qorder,12)):
        g,w = leggauss(n)
        grid.append(b*g)
        weights.append(w/2)
    p,q,v = np.array(np.meshgrid(*grid,indexing='ij')).reshape(3,-1)
    x = np.column_stack((p,np.zeros_like(p),q,v))
    wt = np.einsum('i,j,k->ijk',*weights).ravel()
    z = lift(x,modes)
    active = np.r_[0,np.arange(2,z.shape[1])]
    za = z[:,active]
    scales = np.sqrt(np.sum(wt[:,None]*za**2,axis=0))
    weighted = za/scales*np.sqrt(wt[:,None])
    coeff = np.linalg.lstsq(weighted,drift(x,modes)*np.sqrt(wt[:,None]),rcond=None)[0].T/scales
    A = np.zeros((len(modes)+2,len(modes)+2))
    A[0,1] = 1
    A[1,:2] = [-10000,-200]
    A[2:,active] = coeff
    A[2] = 0
    A[2,3] = 1
    for n in range(3,degree+1,2):
        row = modes.index((n,0))+2
        A[row] = 0
        A[row,modes.index((n-1,1))+2] = n
    B = np.zeros(len(A))
    B[1] = 10000
    rv = drift(x,modes)[:,1] - z@A[3]
    xzero = x.copy()
    xzero[:,0] = 0.
    aerror = acceleration(xzero) - lift(xzero,modes) @ A[3]
    bfield = -.5*np.cos(x[:,2])/(1-.45*np.cos(x[:,2])**2)
    bmean = float(wt@bfield)
    coupling = x[:,0]*(bfield-bmean)
    diagnostics = {'common_acceleration_residual_regional_rms':float(np.sqrt(wt@rv**2)),
                   'state_drift_projection_regional_rms':float(np.sqrt(wt@aerror**2)),
                   'input_coupling_regional_rms':float(np.sqrt(wt@coupling**2)),
                   'mean_input_coefficient':bmean,
                   'orthogonal_split_error':float(abs(wt@rv**2-wt@aerror**2-wt@coupling**2)),
                   'weighted_design_condition':float(np.linalg.cond(weighted)),
                   'max_real_eigenvalue':float(np.linalg.eigvals(A).real.max())}
    return modes,A,B,diagnostics

def projection(degree, qorder=64):
    model = galerkin_model(BOX, degree, (4,4,qorder,12))
    diagnostics = dict(model['diagnostics'])
    diagnostics['quadrature'] = model['metadata']
    return model['modes'], model['A'], model['B'][:,0], diagnostics

@njit
def corrected_rhs(t,z,A,B,pairs):
    q,v=z[2],z[3]
    n=pairs.max()
    hq,hv=np.ones(n+1),np.ones(n+1)
    if n>0:
        hq[1],hv[1]=2*q,2*v
    for k in range(1,n):
        hq[k+1]=2*q*hq[k]-2*k*hq[k-1]
        hv[k+1]=2*v*hv[k]-2*k*hv[k-1]
    sn,cs=np.sin(q),np.cos(q)
    acc=(-9.81*sn-v/.9-.5*z[0]*cs-.45*cs*sn*v*v)/(1-.45*cs*cs)
    phi=np.empty(len(z))
    phi[:2]=z[:2]
    exact=np.empty(len(pairs))
    for j in range(len(pairs)):
        a,b=pairs[j]
        phi[j+2]=.5*hq[a]*hv[b]
        exact[j]=0.
        if a>0:
            exact[j]+=a*hq[a-1]*hv[b]*v
        if b>0:
            exact[j]+=b*hq[a]*hv[b-1]*acc
    rhs=A@z+B*(12*np.sin(.2*t)+2*np.sin(.47*t))
    rhs[2:]+=exact-A[2:]@phi
    return rhs

def command(t):
    return 12*np.sin(.2*t)+2*np.sin(.47*t)

def physical(t,x):
    return [x[1], -10000*x[0]-200*x[1]+10000*command(t),x[3],acceleration(x)[0]]

def run():
    t = np.linspace(0,40,8001)
    x0 = np.array([0.,0.,0.,.1])
    opts = dict(t_eval=t,rtol=2e-12,atol=2e-14,method='DOP853')
    physical_solution = solve_ivp(physical,(0,40),x0,**opts)
    assert physical_solution.success
    x = physical_solution.y.T
    assert np.all(np.abs(x)<BOX)
    stored = {'t':t,'xi':x}
    report = {'box':BOX.tolist(),'command':'12 sin(0.2 t) + 2 sin(0.47 t)',
              'initial_state':x0.tolist(),'time_horizon':40.,'quadrature_orders':[4,4,64,12],
              'integration_algorithm':'MDI-TP with staged polynomial contractions',
              'physical_state_max':np.max(np.abs(x),axis=0).tolist(),'degree':{}}
    for d in (1,3,5):
        modes,A,B,summary = projection(d)
        z0 = lift(x0,modes)[0]
        nominal = solve_ivp(lambda t,z:A@z+B*command(t),(0,40),z0,**opts)
        pairs=np.asarray(modes,dtype=np.int64)
        replay = solve_ivp(lambda t,z:corrected_rhs(t,z,A,B,pairs),(0,40),z0,**opts)
        assert nominal.success and replay.success
        error = nominal.y[2:4].T-x[:,2:4]
        res = drift(x,modes)-lift(x,modes)@A[2:].T
        summary.update(states=len(A),modes=modes,
            nominal_max_error=np.max(np.abs(error),axis=0).tolist(),
            nominal_rms_error=np.sqrt(np.trapezoid(error**2,t,axis=0)/t[-1]).tolist(),
            corrected_max_error=np.max(np.abs(replay.y[2:4].T-x[:,2:4]),axis=0).tolist(),
            common_acceleration_residual_path_max=float(np.abs(res[:,1]).max()),
            common_acceleration_residual_path_rms=float(np.sqrt(np.trapezoid(res[:,1]**2,t)/t[-1])))
        if d==3:
            summary['existing_model_max_coefficient_difference'] = float(np.max(np.abs(A-DATA['plant_A'])))
        modes2,A2,B2,diag2 = projection(d,96)
        summary['quadrature_64_96_max_matrix_difference'] = float(np.max(np.abs(A-A2)))
        stored.update({f'A_{d}':A,f'B_{d}':B,f'nominal_{d}':nominal.y.T,
                       f'corrected_{d}':replay.y.T,f'residual_{d}':res})
        report['degree'][str(d)] = summary
        print(d,summary,flush=True)
    np.savez_compressed(ROOT/'degree_results.npz',**stored)
    (ROOT/'degree_results.json').write_text(json.dumps(report,indent=2))
    from run import residual_family
    family=residual_family()
    y=np.where(family['active'],family['normalized'],np.nan)
    np.savez_compressed(ROOT/'degree_figure_data.npz',
        t=t,xi=x,nominal_1=stored['nominal_1'],nominal_3=stored['nominal_3'],
        nominal_5=stored['nominal_5'],corrected_3=stored['corrected_3'],
        family_t=family['t'],family_lo=np.nanmin(y,axis=1),family_hi=np.nanmax(y,axis=1))

if __name__=='__main__':
    run()
