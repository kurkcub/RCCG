from pathlib import Path
import argparse
import json
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'galerkin-mpl'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

PURPLE, ORANGE, RED, INK = '#634592', '#E69A37', '#DE5C1E', '#242424'

def style():
    plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],
        'mathtext.fontset':'stix','font.size':8.1,'axes.labelsize':8.4,
        'xtick.labelsize':7.4,'ytick.labelsize':7.4,'legend.fontsize':7.1,
        'axes.linewidth':.65,'lines.linewidth':1.,'xtick.direction':'in',
        'ytick.direction':'in','xtick.top':True,'ytick.right':True,
        'xtick.major.width':.55,'ytick.major.width':.55,'pdf.fonttype':42,
        'axes.unicode_minus':False,'path.simplify':False})

def axis(width=3.43,height=2.05,rect=None):
    fig,ax=plt.subplots(figsize=(width,height))
    if rect is not None:ax.set_position(rect)
    ax.grid(True,alpha=.22,linewidth=.4)
    ax.set_axisbelow(True)
    return fig,ax

def save(fig,path,tight=False,book=None):
    if tight:fig.tight_layout(pad=.4)
    fig.savefig(path,bbox_inches=None,pad_inches=0.,metadata={
        'Title':path.stem,'Subject':'Regenerated MDI-TP numerical study'})
    if book is not None:book.savefig(fig,bbox_inches=None,pad_inches=0.)
    plt.close(fig)

def realization(data,out,book=None):
    t,x=data['t'],data['xi'][:,2:]
    fig=plt.figure(figsize=(2.50,2.18))
    for j in (0,1):
        ax=fig.add_axes([.20,.54 if j==0 else .16,.77,.285])
        ax.grid(True,alpha=.22,linewidth=.4);ax.set_axisbelow(True)
        ax.plot(t,data['nominal_3'][:,j+2],color=ORANGE,ls='--',lw=1.1,label='Nominal LTI')
        ax.plot(t,data['corrected_3'][:,j+2],color=PURPLE,lw=.95,label=r'LTI + $\Delta_N$')
        idx=np.unique(np.r_[np.arange(0,len(t),400),len(t)-1])
        ax.plot(t[idx],x[idx,j],ls='none',marker='o',mfc='white',mec=INK,mew=.55,ms=2.7,label='Nonlinear',zorder=4)
        ax.set(xlim=(0,40),ylabel=rf'$x_{j+1}$ ('+('rad)' if j==0 else 'rad/s)'))
        ax.set_xticks([0,10,20,30,40])
        if j==0:
            ax.set_yticks([-.5,0,.5]);ax.tick_params(labelbottom=False)
            ax.legend(frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.5,1.03),
                      fontsize=6.8,columnspacing=.9,labelspacing=.16,handlelength=1.5)
        else:
            ax.set_yticks([-.2,0,.2]);ax.set_xlabel('Time (s)',labelpad=1.)
    save(fig,out/'fig1a_realization.pdf',book=book)

def _plots(root,out,book):
    d=dict(np.load(root/'degree_figure_data.npz', allow_pickle=False))
    f=dict(np.load(root/'results.npz', allow_pickle=False))
    sw=dict(np.load(root/'sweep_results.npz', allow_pickle=False))
    summary=json.loads((root/'results.json').read_text())
    realization(d,out,book)
    fig,ax=axis(2.15,2.18,[.26,.16,.71,.63])
    t=d['family_t']
    ax.fill_between(t,d['family_lo'],d['family_hi'],color=PURPLE,alpha=.18,lw=0,label='Observed range')
    ax.plot(t,d['family_lo'],color=PURPLE,lw=.7);ax.plot(t,d['family_hi'],color=PURPLE,lw=.9)
    ax.axhline(1,color=INK,ls='--',lw=.85,label='Certified bound')
    ax.set_yscale('symlog',linthresh=1e-3,linscale=.6)
    ax.set(xlim=(0,10),ylim=(0,1.15),xlabel='Time (s)',ylabel=r'$\|R_D^{-1}U_N^\top\mathbf{r}_N(\xi(t))\|_2$')
    ax.set_xticks([0,2,4,6,8,10]);ax.set_yticks([0,1e-3,1e-2,1e-1,1])
    ax.set_yticklabels(['0',r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$','1'])
    ax.legend(frameon=False,loc='lower center',bbox_to_anchor=(.5,1.04),labelspacing=.2,handlelength=2.)
    save(fig,out/'fig1b_residual.pdf',book=book)
    fig,ax=axis(2.29,2.18,[.24,.16,.73,.69])
    for degree,color,ls in [(1,ORANGE,'-.'),(3,PURPLE,'-'),(5,RED,'--')]:
        ax.plot(d['t'],np.abs(d[f'nominal_{degree}'][:,2]-d['xi'][:,2]),color=color,ls=ls,lw=1.,label=rf'$d={degree}$')
    ax.set(xlim=(0,40),ylim=(-.003,.15),xlabel='Time (s)',ylabel=r'$|x_1-x_1^{\rm nom}|$ (rad)')
    ax.set_xticks([0,10,20,30,40]);ax.set_yticks([0,.05,.10,.15])
    ax.legend(frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.04),fontsize=7.,columnspacing=.6,handlelength=1.4)
    save(fig,out/'fig1c_degree.pdf',book=book)
    t=f['t'];xi=f['xi']
    fig,ax=axis(3.43,1.90,[.19,.22,.77,.66])
    ax.plot(t,f['dissipation'],color=PURPLE,lw=.9)
    ax.axhline(0,color=INK,ls=':',lw=.65)
    ax.set(xlim=(0,175),xlabel='Time (s)',ylabel='Left side of (66)')
    ax.set_xticks([0,35,70,105,140,175]);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
    save(fig,out/'fig2_dissipation.pdf',book=book)
    fig,ax=axis(3.43,2.02,[.16,.18,.81,.65])
    good=sw['Iw']>0
    ax.plot(sw['t'][good],sw['gain'][good],color=PURPLE,lw=1.05,label=r'Finite-horizon gain $g(T)$')
    gamma=summary['storage']['performance']['gamma']
    ax.axhline(gamma,color=ORANGE,ls='--',lw=1.,label=rf'Certified level ${gamma:g}$')
    ax.annotate(f"{sw['gain'][-1]:.3f}",(sw['t'][-1],sw['gain'][-1]),xytext=(-4,6),textcoords='offset points',ha='right',color=PURPLE,fontsize=7.6)
    gain_upper=1.11*max(float(gamma),float(np.max(sw['gain'][good])))
    ax.set(xlim=(0,65),ylim=(0,gain_upper),xlabel=r'Horizon $T$ (s)',ylabel=r'$g(T)$')
    ax.set_xticks([0,15,30,45,60,65]);ax.set_yticks([0,5,10])
    ax.legend(frameon=False,loc='lower center',bbox_to_anchor=(.5,1.035),ncol=1,labelspacing=.18)
    save(fig,out/'fig3a_finite_horizon_gain.pdf',book=book)
    box=np.array(summary['localization']['auxiliary_box']);bounds=np.array(summary['localization']['physical_bounds'])
    fig,ax=axis(3.43,2.02,[.16,.18,.81,.65])
    for j,color in [(2,PURPLE),(3,ORANGE)]:
        ax.plot(t,np.abs(xi[:,j])/box[j],color=color,lw=.95,label=rf'$|x_{j-1}(t)|/b_{{{j+1},\rm loc}}$')
        ax.axhline(bounds[j]/box[j],color=color,ls='--',lw=.9,label=rf'$d_{j+1}/b_{{{j+1},\rm loc}}$')
    ax.axhline(1,color=INK,ls=':',lw=.7)
    ax.set(xlim=(0,175),ylim=(-.025,1.055),xlabel='Time (s)',ylabel='Normalized plant coordinate')
    ax.set_xticks([0,35,70,105,140,175]);ax.set_yticks([0,.25,.5,.75,1])
    ax.legend(frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.5,1.015),columnspacing=1.5,labelspacing=.18)
    save(fig,out/'fig3b_coordinate_bounds.pdf',book=book)
    fig,ax=axis(3.43,2.12,[.16,.20,.81,.65])
    ax.plot(xi[:,2],xi[:,3],color=PURPLE,lw=.9,label='Nonlinear trajectory')
    bq,bv=box[2:]
    ax.plot([-bq,bq,bq,-bq,-bq],[-bv,-bv,bv,bv,-bv],color=ORANGE,ls='--',lw=.9,label='Localization region')
    for tt,offset in [(0,(6,6)),(90,(6,5)),(175,(-29,-11))]:
        q,v=[np.interp(tt,t,xi[:,j]) for j in (2,3)]
        ax.plot(q,v,ls='none',marker='o',ms=3.4,color=ORANGE,zorder=5)
        ax.annotate(f'{tt} s',(q,v),xytext=offset,textcoords='offset points',fontsize=7.3,color=INK)
    ax.set(xlim=(-.78,.78),ylim=(-.51,.51),xlabel=r'$x_1$ (rad)',ylabel=r'$x_2$ (rad/s)')
    ax.set_xticks([-.75,-.5,0,.5,.75]);ax.set_yticks([-.5,-.25,0,.25,.5])
    ax.legend(frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.5,1.04),fontsize=7.1,columnspacing=1.,handlelength=1.8)
    fig.canvas.draw()
    for tt in [20.,34.,114.]:
        pts=np.array([[np.interp(s,t,xi[:,j]) for j in (2,3)] for s in [tt-.08,tt+.08]])
        tangent=np.diff(ax.transData.transform(pts),axis=0)[0]
        angle=np.degrees(np.arctan2(tangent[1],tangent[0]))
        marker=MarkerStyle('>');path=marker.get_path().transformed(marker.get_transform()).transformed(Affine2D().rotate_deg(angle))
        q,v=[np.interp(tt,t,xi[:,j]) for j in (2,3)]
        ax.scatter([q],[v],marker=path,s=25,color=ORANGE,linewidths=0,zorder=4)
    save(fig,out/'fig4_physical_path.pdf',book=book)

def run(root=None):
    root=Path(__file__).resolve().parent if root is None else Path(root)
    out=root/'figures';out.mkdir(exist_ok=True);style()
    _plots(root,out,None)
    print('Seven manuscript PDF figures written.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    run(p.parse_args().root)
