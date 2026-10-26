"""Single-column figures from the saved four-controller results."""

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
from matplotlib.lines import Line2D

from design_data import fingerprint, load_data

HERE = Path(__file__).resolve().parent
ROWS = []
CERT = {}
OUT = HERE / "figures"
RESULTS = HERE / "results"
CONFIG = [
    ("galerkin3", r"Galerkin $d=3$ + $\mathcal{H}_\infty$", "#E69A37", "-", "o"),
    ("jacobian", r"Jacobian + $\mathcal{H}_\infty$", "#0072B2", ":", "s"),
    ("backstepping", "Backstepping", "#634592", "-.", "^"),
    ("hybrid_mpc", "Koopman-based\nhybrid MPC [49]", "#DE5C1E", "--", "D"),
]
STYLE = {key: (label, color, line, marker) for key, label, color, line, marker in CONFIG}
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 7.6,
        "axes.labelsize": 7.7,
        "axes.titlesize": 7.7,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "axes.linewidth": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def handles(markers):
    return [
        Line2D(
            [],
            [],
            color=color,
            ls=line,
            lw=1.1,
            label=label,
            marker=marker if markers else None,
            ms=3.2,
            mec="#333333",
            mew=0.35,
        )
        for key, label, color, line, marker in CONFIG
    ]


def save(fig, name):
    for extension in ["pdf", "png"]:
        fig.savefig(OUT / f"{name}.{extension}", dpi=220)
    plt.close(fig)


def aggregate():
    fig, axes = plt.subplots(3, 2, figsize=(3.45, 3.65))
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.105, top=0.84, hspace=0.58, wspace=0.40)
    frequencies = [0.5, 2.0, 4.0]
    fields = ["total_RMSE", "total_peak_error"]
    limits = [np.ceil(max(r[f] for r in ROWS) * 1000 * 1.1 * 20) / 20 for f in fields]
    for row, w in enumerate(frequencies):
        for col, field in enumerate(fields):
            ax = axes[row, col]
            for key, label, color, line, marker in CONFIG:
                selected = [
                    next(
                        r
                        for r in ROWS
                        if r["controller"] == key and r["mean"] == b and r["omega"] == w
                    )
                    for b in [0, 0.3, 0.6]
                ]
                ax.plot(
                    [0, 0.3, 0.6],
                    [r[field] * 1000 for r in selected],
                    color=color,
                    ls=line,
                    marker=marker,
                    lw=1.1,
                    ms=3.1,
                    mec="#333333",
                    mew=0.35,
                )
            ax.set(
                xlim=(-0.03, 0.63),
                ylim=(0, limits[col]),
                xticks=[0, 0.3, 0.6],
                ylabel=["RMSE (mrad)", "Peak error (mrad)"][col],
            )
            ax.set_title(f"({'abcdef'[row * 2 + col]}) $\\omega={w:g}$ rad/s", loc="left", pad=3)
            if row == 2:
                ax.set_xlabel(r"Mean reference $b$ (rad)")
            ax.grid(axis="y", color="#DDDDDD", lw=0.4)
            ax.set_axisbelow(True)
    fig.legend(
        handles=handles(True),
        loc="upper center",
        bbox_to_anchor=(0.53, 0.995),
        ncol=2,
        frameon=False,
        fontsize=7.0,
        handlelength=2.0,
        columnspacing=0.9,
        handletextpad=0.5,
        labelspacing=0.3,
    )
    save(fig, "common_controller_comparison")


def temporal():
    fig, axes = plt.subplots(
        3, 1, figsize=(3.45, 3.95), gridspec_kw={"height_ratios": [0.65, 1, 0.95]}
    )
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.15, top=0.83, hspace=0.83)
    response, error, enclosure = axes
    start, end = 35.0, 35.0 + np.pi
    for key in ["backstepping", "jacobian", "hybrid_mpc", "galerkin3"]:
        label, color, line, marker = STYLE[key]
        row = next(r for r in ROWS if r["controller"] == key and r["case"] == "b0.6_w4")
        trace = np.load(RESULTS / row["trace_path"])
        t = trace["time"]
        q = trace["state"][:, 2]
        response.plot(t, q, color=color, ls=line, lw=1.05)
        mask = (t >= start) & (t <= end)
        error.plot(
            t[mask], 1000 * abs(trace["reference"][mask] - q[mask]), color=color, ls=line, lw=1.05
        )
    reference_style = (0, (6, 3))
    response.plot(t, trace["reference"], color="#222222", ls=reference_style, lw=0.8)
    response.axvspan(start, end, color="#BBBBBB", alpha=0.35, zorder=0)
    response.set(
        xlim=(0, 80),
        ylim=(-0.025, 0.67),
        xticks=[0, 20, 40, 60, 80],
        yticks=[0, 0.3, 0.6],
        ylabel=r"$x_1$ (rad)",
        xlabel="Time (s)",
    )
    response.set_title("(a) Reference and responses", loc="left", pad=3)
    error.set(
        xlim=(start, end),
        ylim=(0, 0.65),
        yticks=[0, 0.3, 0.6],
        xlabel="Time (s)",
        ylabel=r"$|r-x_1|$ (mrad)",
    )
    error.set_title("(b) Absolute tracking error", loc="left", pad=3)
    bounds = [r["physical_bounds_upper"][3] for r in CERT["rows"][:2]]
    lower = CERT["rows"][2]["velocity_closure_component_strict_lower"]
    colors = [STYLE[k][1] for k in ["galerkin3", "jacobian", "hybrid_mpc"]]
    enclosure.bar([0, 1], bounds, color=colors[:2], width=0.48, edgecolor=colors[:2], lw=0.5)
    enclosure.bar(
        2, lower, color=colors[2], width=0.48, edgecolor=colors[2], alpha=0.2, hatch="///", lw=0.5
    )
    for i, b in enumerate(bounds):
        enclosure.text(
            i,
            b + (-0.055 if i == 0 else 0.023),
            rf"$\leq {np.ceil(b * 1e4) / 1e4:.4f}$",
            ha="center",
            fontsize=6.7,
            bbox=dict(fc="white", ec="none", pad=0.2),
        )
    enclosure.annotate(
        "",
        xy=(2, 0.61),
        xytext=(2, lower),
        arrowprops=dict(arrowstyle="-|>", color=colors[2], lw=1.05),
    )
    enclosure.text(2, 0.63, r"$>0.507$", ha="center", fontsize=6.8, color=colors[2])
    enclosure.axhline(0.47, color="#222222", ls="--", lw=0.7)
    enclosure.text(
        0.015, 0.97, r"Limit $0.47$ rad/s", transform=enclosure.transAxes, va="top", fontsize=6.7
    )
    enclosure.set(
        xticks=[0, 1, 2],
        xticklabels=[
            "Galerkin\nupper bound",
            "Jacobian\nupper bound",
            "Koopman-based\nhybrid MPC\nclosure lower bound",
        ],
        ylim=(0, 0.71),
        yticks=[0, 0.3, 0.6],
        ylabel="Velocity term (rad/s)",
    )
    enclosure.tick_params(axis="x", labelsize=6.7)
    enclosure.set_title("(c) Containment test", loc="left", pad=3)
    for ax in axes:
        ax.grid(axis="y", color="#DDDDDD", lw=0.4)
        ax.set_axisbelow(True)
    legend = handles(False) + [
        Line2D([], [], color="#222222", ls=reference_style, lw=0.8, label="Reference")
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.995),
        ncol=2,
        frameon=False,
        fontsize=7.0,
        handlelength=2.0,
        columnspacing=0.9,
        handletextpad=0.5,
        labelspacing=0.3,
    )
    save(fig, "common_time_response")


def main():
    global ROWS, CERT, OUT, RESULTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--output", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    RESULTS = args.results
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=True)
    ROWS = json.loads((RESULTS / "comparison_results.json").read_text())
    CERT = json.loads((HERE / "certificate_summary.json").read_text())
    current = load_data()
    used = dict(np.load(RESULTS / "controllers_used.npz", allow_pickle=False))
    if fingerprint(current) != fingerprint(used) or CERT["design_fingerprint"] != fingerprint(used):
        raise ValueError("Figures, simulations and certificates use different controller inputs")
    aggregate()
    temporal()


if __name__ == "__main__":
    main()
