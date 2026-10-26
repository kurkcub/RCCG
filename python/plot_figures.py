"""Figures 1a-1c and 2 from degree_figure_data.npz and the sweep results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache/mpl"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PURPLE, ORANGE, RED, INK = "#634592", "#E69A37", "#DE5C1E", "#242424"


def style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 8.1,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.1,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
            "path.simplify": False,
        }
    )


def axis(width=3.43, height=2.05, rect=None):
    fig, ax = plt.subplots(figsize=(width, height))
    if rect is not None:
        ax.set_position(rect)
    ax.grid(True, alpha=0.22, linewidth=0.4)
    ax.set_axisbelow(True)
    return fig, ax


def save(fig, path):
    fig.savefig(
        path,
        bbox_inches=None,
        pad_inches=0.0,
        metadata={"Title": path.stem, "Subject": "Regenerated MDI-TP numerical study"},
    )
    plt.close(fig)


def realization(data, out):
    t, x = data["t"], data["xi"][:, 2:]
    fig = plt.figure(figsize=(2.50, 2.18))
    for j in (0, 1):
        ax = fig.add_axes([0.20, 0.54 if j == 0 else 0.16, 0.77, 0.285])
        ax.grid(True, alpha=0.22, linewidth=0.4)
        ax.set_axisbelow(True)
        ax.plot(t, data["nominal_3"][:, j + 2], color=ORANGE, ls="--", lw=1.1, label="Nominal LTI")
        ax.plot(t, data["corrected_3"][:, j + 2], color=PURPLE, lw=0.95, label=r"LTI + $\Delta_N$")
        idx = np.unique(np.r_[np.arange(0, len(t), 400), len(t) - 1])
        ax.plot(
            t[idx],
            x[idx, j],
            ls="none",
            marker="o",
            mfc="white",
            mec=INK,
            mew=0.55,
            ms=2.7,
            label="Nonlinear",
            zorder=4,
        )
        ax.set(xlim=(0, 40), ylabel=rf"$x_{j + 1}$ (" + ("rad)" if j == 0 else "rad/s)"))
        ax.set_xticks([0, 10, 20, 30, 40])
        if j == 0:
            ax.set_yticks([-0.5, 0, 0.5])
            ax.tick_params(labelbottom=False)
            ax.legend(
                frameon=False,
                ncol=2,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.03),
                fontsize=6.8,
                columnspacing=0.9,
                labelspacing=0.16,
                handlelength=1.5,
            )
        else:
            ax.set_yticks([-0.2, 0, 0.2])
            ax.set_xlabel("Time (s)", labelpad=1.0)
    save(fig, out / "fig1a_realization.pdf")


def _plots(root, out):
    d = dict(np.load(root / "degree_figure_data.npz", allow_pickle=False))
    realization(d, out)
    fig, ax = axis(2.15, 2.18, [0.26, 0.16, 0.71, 0.63])
    t = d["family_t"]
    ax.fill_between(
        t, d["family_lo"], d["family_hi"], color=PURPLE, alpha=0.18, lw=0, label="Observed range"
    )
    ax.plot(t, d["family_lo"], color=PURPLE, lw=0.7)
    ax.plot(t, d["family_hi"], color=PURPLE, lw=0.9)
    ax.axhline(1, color=INK, ls="--", lw=0.85, label="Certified bound")
    ax.set_yscale("symlog", linthresh=1e-3, linscale=0.6)
    ax.set(
        xlim=(0, 10),
        ylim=(0, 1.15),
        xlabel="Time (s)",
        ylabel=r"$\|R_D^{-1}U_N^\top\mathbf{r}_N(\xi(t))\|_2$",
    )
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_yticks([0, 1e-3, 1e-2, 1e-1, 1])
    ax.set_yticklabels(["0", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", "1"])
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        labelspacing=0.2,
        handlelength=2.0,
    )
    save(fig, out / "fig1b_residual.pdf")
    fig, ax = axis(2.29, 2.18, [0.24, 0.16, 0.73, 0.69])
    for degree, color, ls in [(1, ORANGE, "-."), (3, PURPLE, "-"), (5, RED, "--")]:
        ax.plot(
            d["t"],
            np.abs(d[f"nominal_{degree}"][:, 2] - d["xi"][:, 2]),
            color=color,
            ls=ls,
            lw=1.0,
            label=rf"$d={degree}$",
        )
    ax.set(
        xlim=(0, 40), ylim=(-0.003, 0.15), xlabel="Time (s)", ylabel=r"$|x_1-x_1^{\rm nom}|$ (rad)"
    )
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_yticks([0, 0.05, 0.10, 0.15])
    ax.legend(
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        fontsize=7.0,
        columnspacing=0.6,
        handlelength=1.4,
    )
    save(fig, out / "fig1c_degree.pdf")
    sw = dict(np.load(root / "sweep_results.npz", allow_pickle=False))
    fig, ax = axis(3.43, 1.72, [0.16, 0.22, 0.81, 0.74])
    good = sw["Iw"] > 0
    ax.plot(sw["t"][good], sw["gain"][good], color=PURPLE, lw=1.05)
    gamma = float(json.loads((root / "sweep_summary.json").read_text())["certified_gamma"])
    ax.axhline(gamma, color=ORANGE, ls="--", lw=1.0)
    ax.annotate(
        f"{sw['gain'][-1]:.3f}",
        (sw["t"][-1], sw["gain"][-1]),
        xytext=(-4, 6),
        textcoords="offset points",
        ha="right",
        color=PURPLE,
        fontsize=7.6,
    )
    ax.set(xlim=(0, 65), ylim=(0, 13.431), xlabel=r"Horizon $T$ (s)", ylabel=r"$g(T)$")
    ax.set_xticks([0, 15, 30, 45, 60, 65])
    ax.set_yticks([0, 5, 10, 12.1])
    ax.set_yticklabels(["0", "5", "10", "12.1"])
    save(fig, out / "fig3a_finite_horizon_gain.pdf")


def run(root=None):
    root = Path(__file__).resolve().parent if root is None else Path(root)
    out = root / "figures"
    out.mkdir(exist_ok=True)
    style()
    _plots(root, out)
    print("Figures 1a, 1b, 1c and 3a written to", out, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    run(p.parse_args().root)
