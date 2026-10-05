from pathlib import Path
import argparse
import json
import sys
import numpy as np
from numba import njit
from scipy.special import hyp1f1

from localization import verify_reference


@njit(inline='always')
def energy_rhs(x, r, ak, bk, ck, dk, aw, bw, cw, dw, au, bu, cu, du, qr, qa, out):
    run._feedback_rhs(x, r, ak, bk, ck, dk, aw, bw, au, bu, out)
    nk, nw = len(ak), len(aw)
    e = r-x[2]
    u = dk*e
    for j in range(nk):
        u += ck[j]*x[4+j]
    zp = dw*e
    for j in range(nw):
        zp += cw[j]*x[4+nk+j]
    zu = du*u
    for j in range(len(au)):
        zu += cu[j]*x[4+nk+nw+j]
    p, pd, q, v, acc = x[0], x[1], x[2], x[3], out[3]
    z4, z5, z6, z7 = 4*v**3-6*v, 4*q*v*v-2*q, 4*q*q*v-2*v, 4*q**3-6*q
    f2 = (12*v*v-6)*acc
    f3 = 4*v**3-2*v+8*q*v*acc
    f4 = 8*q*v*v+(4*q*q-2)*acc
    f5 = (12*q*q-6)*v
    power = (r/.682)**2
    for j in range(len(qr)):
        wc = qr[j,0]*v+qr[j,1]*acc+qr[j,2]*f2+qr[j,3]*f3+qr[j,4]*f4+qr[j,5]*f5
        wc -= qa[j,0]*p+qa[j,1]*pd+qa[j,2]*q+qa[j,3]*v+qa[j,4]*z4+qa[j,5]*z5+qa[j,6]*z6+qa[j,7]*z7
        power += wc*wc
    out[-2] = zp*zp+zu*zu
    out[-1] = power


@njit
def energy_closed_loop(tk, rk, ak, bk, ck, dk, aw, bw, cw, dw, au, bu, cu, du, qr, qa, step, marks, initial):
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
            energy_rhs(x, rr, ak, bk, ck, dk, aw, bw, cw, dw, au, bu, cu, du, qr, qa, k1)
            tmp[:] = x+.5*h*k1
            energy_rhs(tmp, rm, ak, bk, ck, dk, aw, bw, cw, dw, au, bu, cu, du, qr, qa, k2)
            tmp[:] = x+.5*h*k2
            energy_rhs(tmp, rm, ak, bk, ck, dk, aw, bw, cw, dw, au, bu, cu, du, qr, qa, k3)
            tmp[:] = x+h*k3
            energy_rhs(tmp, re, ak, bk, ck, dk, aw, bw, cw, dw, au, bu, cu, du, qr, qa, k4)
            x += h*(k1+2*k2+2*k3+k4)/6.
            for k in range(4):
                coordinate_maxima[k] = max(coordinate_maxima[k], abs(x[k]))
            error_peak = max(error_peak, abs(re-x[2]))
            control_peak = max(control_peak, abs(dk*(re-x[2])+np.dot(ck,x[4:4+len(ak)])))
            save = mark < len(marks) and j+1 == marks[mark]
            if save:
                mark += 1
            if save or block == 0:
                tout[rec] = tk[block]+(j+1)*h
                rout[rec], states[rec] = re, x
                rec += 1
    return tout[:rec], rout[:rec], states[:rec], coordinate_maxima, error_peak, control_peak

def reference_profile():
    active = 60.
    stop = 65.
    tk = np.arange(round(stop/.02)+1)*.02
    sweep_time = np.minimum(tk, active)
    low, high = .22, 10.
    log_ratio = np.log(high/low)
    frequency_shape = log_ratio*(sweep_time/active)**4
    omega = low*np.exp(frequency_shape)
    phase = low*sweep_time*hyp1f1(.25, 1.25, frequency_shape)
    window = np.ones_like(tk)
    ramp = 12.
    left = tk < ramp
    right = tk > active-ramp
    window[left] = np.sin(.5*np.pi*tk[left]/ramp)**2
    window[right] = np.sin(.5*np.pi*np.maximum(active-tk[right], 0.)/ramp)**2
    window[tk >= active] = 0.
    modulation = .65+.35*np.cos(2*np.pi*(sweep_time-8.)/25.)
    shape = window*modulation/np.sqrt(1+(omega/.5)**4)
    scale = .28
    amplitude = scale*shape
    reference = amplitude*np.sin(phase)
    tk = verify_reference(tk, reference)
    return dict(tk=tk, reference=reference, amplitude=amplitude, omega=omega,
                active=active, stop=stop, low=low, high=high, scale=scale, ramp=ramp)


def main():
    global run
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--step', type=float, default=1e-5)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code.resolve()))
    import run


    saved = dict(np.load(args.code/'results.npz', allow_pickle=False))
    paper = dict(np.load(args.code/'paper_data.npz', allow_pickle=False))
    args.output.mkdir(parents=True, exist_ok=True)
    profile = reference_profile()
    tk,reference,amplitude,omega = (profile[key] for key in ('tk','reference','amplitude','omega'))
    active,stop,low,high,scale,ramp = (profile[key] for key in ('active','stop','low','high','scale','ramp'))
    ak,bk,ck,dk = (saved['controller_'+q] for q in 'ABCD')
    aw,bw,cw,dw = (saved['controller_'+q+'_WP'] for q in 'ABCD')
    au,bu,cu,du = (saved['controller_'+q+'_WU'] for q in 'ABCD')
    offsets = np.array([.0001,.0002,.0005,.001,.0015,.002,.003,.005,.0075,.01,.015,.02])
    marks = run.output_marks(args.step, offsets)
    initial = np.zeros(4+len(ak)+len(aw)+len(au)+2)
    qr = np.linalg.solve(paper['shape_Draw'], paper['shape_U'].T)
    qa = qr@saved['plant_A'][2:]
    t,r,augmented,maxima,epeak,upeak = energy_closed_loop(tk,reference,ak,bk.ravel(),ck.ravel(),dk.item(),aw,bw.ravel(),cw.ravel(),dw.item(),au,bu.ravel(),cu.ravel(),du.item(),qr,qa,args.step,marks,initial)
    if not np.isfinite(augmented).all() or np.any(maxima >= paper['plant_bounds']):
        raise RuntimeError('The sweep trajectory is nonfinite or outside the region.')
    x = augmented[:,:-2]
    Iz, Iw = augmented[:,-2], augmented[:,-1]
    xi=x[:,:4]
    xk=x[:,4:4+len(ak)]
    xw=x[:,4+len(ak):4+len(ak)+len(aw)]
    xu=x[:,4+len(ak)+len(aw):]
    e=r-xi[:,2]
    u=xk@ck.ravel()+dk.item()*e
    z=np.column_stack((xw@cw.ravel()+dw.item()*e,xu@cu.ravel()+du.item()*u))
    zeta=run.lift(xi)
    field=run.drift(xi)
    residual=field-zeta@saved['plant_A'][2:].T
    wc=np.linalg.solve(paper['shape_Draw'],(residual@paper['shape_U']).T).T
    w=np.column_stack((wc,r/.682))
    z2=np.sum(z*z,axis=1)
    w2=np.sum(w*w,axis=1)
    gain=np.sqrt(np.divide(Iz,Iw,out=np.zeros_like(Iz),where=Iw>0))
    chi=np.column_stack((zeta,xw,xu,xk))
    derivative=np.column_stack((xi[:,1],-10000*xi[:,0]-200*xi[:,1]+10000*u,field,xw@aw.T+e[:,None]*bw.ravel(),xu@au.T+u[:,None]*bu.ravel(),xk@ak.T+e[:,None]*bk.ravel()))
    transform=saved['certificate_analysis_transform']
    xhat=chi@transform.T
    dhat=derivative@transform.T
    P=saved['P_performance']
    V=np.einsum('ij,ij->i',xhat@P,xhat)
    dV=2*np.einsum('ij,ij->i',xhat@P,dhat)
    gamma=float(saved['gamma_performance'])
    epsilon=float(saved['epsilon_performance'])
    lhs=dV+z2+epsilon*np.sum(xhat*xhat,axis=1)-gamma**2*w2
    slopes=np.diff(reference)/np.diff(tk)
    jumps=np.diff(np.r_[0.,slopes,0.])
    report=dict(profile='oscillating_envelope',initial='zero',frequency_interval_rad_s=[low,high],active_duration=active,stop_time=stop,amplitude_scale=scale,max_reference=float(max(abs(reference))),max_slope=float(max(abs(slopes))),max_slope_jump=float(max(abs(jumps))),reference_verified=True,finite_horizon_gain=float(gain[-1]),certified_gamma=gamma,gamma_utilization=float(gain[-1]/gamma),physical_maxima=maxima.tolist(),peak_tracking_error=float(epeak),peak_control=float(upeak),peak_normalized_closure=float(np.linalg.norm(wc,axis=1).max()),dissipation_maximum=float(lhs.max()),energy_output=float(Iz[-1]),energy_input=float(Iw[-1]),terminal_performance_storage=float(V[-1]),step=args.step)
    report['frequency_exponent'] = 4
    report['phase_definition'] = 'integral from 0 to t of omega(s), evaluated as 0.22*t*hyp1f1(0.25,1.25,log(10/0.22)*(t/60)**4)'
    report['amplitude_modulation'] = dict(mean=.65, depth=.35, period=25., phase_shift=8.)
    report['amplitude_rolloff_rad_s'] = .5
    report['endpoint_ramp_duration'] = ramp
    report['gain_at_times'] = {str(q): float(np.interp(q,t,gain)) for q in (20.,30.,35.,40.,45.,50.,55.,60.,65.)}
    report['energy_integration'] = 'RK4 passive integral states evaluated at every ODE stage'
    report['closure_stage_readout_difference'] = float(np.max(abs(field@qr.T-zeta@qa.T-wc)))
    np.savez_compressed(args.output/'sweep_results.npz',t=t,reference=r,xi=xi,error=e,control=u,z=z,w=w,gain=gain,Iz=Iz,Iw=Iw,V=V,dissipation=lhs,reference_t=tk,reference_r=reference,amplitude=amplitude,omega=omega)
    (args.output/'sweep_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report


if __name__ == "__main__":
    main()
