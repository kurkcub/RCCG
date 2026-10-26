"""Reproduce the numerical example and Figures 1-4."""

from pathlib import Path
import argparse
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def execute(relative, *arguments):
    script = ROOT / relative
    subprocess.run([sys.executable, str(script), *arguments], cwd=script.parent, check=True)


def collect_figures():
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    sources = {
        "python": [
            "fig1a_realization.pdf",
            "fig1b_residual.pdf",
            "fig1c_degree.pdf",
            "fig3a_finite_horizon_gain.pdf",
        ],
        "comparison": ["common_controller_comparison.pdf", "common_time_response.pdf"],
    }
    for folder, names in sources.items():
        for name in names:
            shutil.copy2(ROOT / folder / "figures" / name, out / name)
    print("Figures 1-4 saved in figures/.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--figures-only", action="store_true", help="Redraw figures from a completed run."
    )
    mode.add_argument(
        "--certificates-only",
        action="store_true",
        help="Check the regional and performance witnesses from a completed run.",
    )
    args = parser.parse_args()
    if args.certificates_only:
        execute("certificates/check.py")
        return
    if args.figures_only:
        execute("python/run.py", "--figures-only")
    else:
        execute("python/run.py")
        execute("comparison/design_jacobian.py")
        execute("comparison/fit_mpc.py")
        execute("certificates/build.py")
        execute("certificates/check.py")
        execute("comparison/run.py")
    execute("comparison/plot.py")
    collect_figures()


if __name__ == "__main__":
    main()
