from pathlib import Path
import argparse

import numpy as np
from plot_figures import style
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import FormatStrFormatter


GOLD, PURPLE, INK = '#E69A37', '#634592', '#141414'
STYLES, MARKERS = ['-', '--', '-.'], ['o', 's', '^']


def run(root=None):
    root = Path(__file__).resolve().parent if root is None else Path(root)
    with np.load(root/'comparison_results.npz', allow_pickle=False) as data:
        t, y = np.asarray(data['t']).ravel(), np.asarray(data['y'])
    if t.size < 2 or y.shape != (t.size, 7):
        raise ValueError('Expected time samples and seven comparison columns.')
    if not np.isfinite(t).all() or not np.isfinite(y).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Comparison samples must be finite with increasing times.')

    style()
    fig = plt.figure(figsize=(18/2.54, 6.8/2.54))
    ax = fig.add_axes([.075, .24, .91, .735])
    r = y[:, 6]
    colors = [GOLD, PURPLE]
    for controller in [1, 0]:
        for model in range(3):
            col = 3*controller+model
            ax.plot(t, y[:, col], color=colors[controller], ls=STYLES[model],
                    lw=2.2 if controller == 0 else 1.4)
    for col in range(6):
        controller, model = divmod(col, 3)
        phase = .15+.70*col/5
        marker_times = t[0]+(np.arange(7)+phase)*(t[-1]-t[0])/7
        right = np.minimum(np.searchsorted(t, marker_times), len(t)-1)
        left = np.maximum(right-1, 0)
        indices = np.unique(np.where(abs(t[left]-marker_times) < abs(t[right]-marker_times), left, right))
        ax.plot(t[indices], y[indices, col], ls='none', marker=MARKERS[model],
                ms=5.6, mew=1.4, mec=colors[controller],
                mfc=colors[controller] if controller == 0 else 'white')
    ax.plot(t, r, color=INK, ls='--', lw=1.9)
    lo, hi = float(y.min()), float(y.max())
    pad = .06*(hi-lo) if hi > lo else .05*max(abs(lo), 1e-3)
    ax.set(xlim=(t[0], t[-1]), ylim=(lo-pad, hi+pad), xlabel='t (s)',
           ylabel=r'$y_1$ (rad)')
    ax.grid(alpha=.13, linewidth=.4)

    # Matplotlib fills legend columns downwards; interleave the two rows.
    controller_handles = [Line2D([], [], color=GOLD, lw=2.2),
                          Line2D([], [], color=PURPLE, lw=1.4),
                          Line2D([], [], color=INK, ls='--', lw=1.9)]
    model_handles = [Line2D([], [], color='#3D3D3D', ls=STYLES[i],
                           marker=MARKERS[i], mfc='white', ms=5.6, lw=1.4)
                     for i in range(3)]
    handles = [item for pair in zip(controller_handles, model_handles) for item in pair]
    labels = [r'$H_\infty$', 'Nominal', 'Backstepping', 'P1', 'Reference', 'P2']
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .01),
               ncol=3, frameon=False, fontsize=8, columnspacing=2., handlelength=3.,
               labelspacing=.16, borderpad=0.)

    zoom_x = np.array([34.70, 34.80])
    selected = (t >= zoom_x[0]) & (t <= zoom_x[1])
    if np.count_nonzero(selected) < 2:
        raise ValueError('The comparison must include the nominal zoom interval.')
    zoom_values = y[selected][:, [0, 3, 6]]
    zoom_pad = max(.12*float(np.ptp(zoom_values)), 1e-6)
    zoom_y = np.array([zoom_values.min()-zoom_pad, zoom_values.max()+zoom_pad])
    inset_position = [.66, .2988, .32, .28]
    inset = fig.add_axes(inset_position, facecolor='white')
    inset.set_title('Nominal', fontsize=6, fontweight='normal', pad=2)
    for col, color in [(0, GOLD), (3, PURPLE)]:
        inset.plot(t[selected], y[selected, col], color=color,
                   lw=1.4 if col == 0 else 1.2)
    inset.plot(t[selected], r[selected], color=INK, ls='--', lw=1.3)
    inset.set(xlim=zoom_x, xticks=[34.70, 34.75, 34.80],
              yticks=[.54130, .54135, .54140])
    inset.set_ylim(zoom_y)
    inset.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    inset.yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
    inset.tick_params(labelsize=6, length=2)
    inset.grid(alpha=.2, linewidth=.4)
    for spine in inset.spines.values():
        spine.set_linewidth(.8)

    # The box marks the zoom location; its minimum size keeps it visible.
    xhalf = max(.5*np.ptp(zoom_x)*2*np.sqrt(2), .005*(t[-1]-t[0]))
    yhalf = max(.5*np.ptp(zoom_y)*6.4*np.sqrt(2), .010*(hi-lo+2*pad))
    center_x, center_y = zoom_x.mean(), zoom_y.mean()
    box_x = [max(center_x-xhalf, t[0]), min(center_x+xhalf, t[-1])]
    box_y = [max(center_y-yhalf, lo-pad), min(center_y+yhalf, hi+pad)]
    ax.add_patch(Rectangle((box_x[0], box_y[0]), np.ptp(box_x), np.ptp(box_y),
                           edgecolor='red', facecolor=(1., .75, .75, .16), lw=2.2,
                           zorder=5))
    fig.add_artist(ConnectionPatch(
        xyA=(box_x[1], box_y[0]), coordsA='data', axesA=ax,
        xyB=(inset_position[0]+.02*inset_position[2],
             inset_position[1]+.02*inset_position[3]),
        coordsB='figure fraction', arrowstyle='->', color='red', lw=1.,
        mutation_scale=9., clip_on=False, zorder=10))

    out = root/'figures'
    out.mkdir(exist_ok=True)
    path = out/'fig5_tracking_comparison.pdf'
    fig.savefig(path, metadata={'Title': 'Tracking comparison',
                               'Subject': 'Nominal and perturbed physical plant simulations'})
    plt.close(fig)
    print('Figure 5 PDF written.', flush=True)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    run(parser.parse_args().root)
