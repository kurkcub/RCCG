"""Check the regional and performance certificates used in the paper.

Every check raises on failure; the summary is written to results.json.  The
shared exact-arithmetic modules live in ../python.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "checks"))
sys.path.insert(1, str(ROOT.parent / "python"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-localization",
        action="store_true",
        help="Recompute the convolution, initial and reference envelopes.",
    )
    args = parser.parse_args()
    from build import check_sources

    check_sources()
    import brl
    import gain_lower
    import localization_check
    import mpc_prefix
    import mpc_split
    import reference_check
    import residual_check

    jobs = [
        ("bounded_real", brl.main),
        ("residual_enclosures", residual_check.main),
        ("reference_class", reference_check.main),
        ("localization", lambda: localization_check.main(args.full_localization)),
        ("jacobian_gain_lower", gain_lower.main),
        ("mpc_feedback_split", mpc_split.main),
        ("mpc_closure_lower", mpc_prefix.main),
    ]
    report = {}
    for name, check in jobs:
        started = time.time()
        print(name, flush=True)
        report[name] = check()
        print(f"  passed ({time.time() - started:.1f} s)", flush=True)
    report["all_checks_passed"] = True
    from build import write_plot_summary

    write_plot_summary(report)
    (ROOT / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
