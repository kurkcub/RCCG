# Model-Based Galerkin Control

Code for [*Model-Based Galerkin Lifting with Exact LTI Decomposition for Guaranteed H-infinity Output Feedback Control of Nonlinear Systems*](https://arxiv.org/abs/2610.07299), by Burak Kürkçü, Christopher Phan, Aleksandar Zecevic, and Maryam Khanbaghi.

Python code for the Galerkin design, regional certificates, nonlinear controller comparison, and Figs. 1–5. Localization uses a cubic lifted auxiliary realization with the same controller.

## Python

Tested with Python 3.12. From the repository folder:

```sh
python -m pip install -r python/requirements.txt
python python/run.py
```

This rebuilds the Galerkin models and controller, checks the regional certificates, runs the nonlinear simulations and degree study, and saves Figs. 1–5 in `python/figures`.

To redraw saved results:

```sh
python python/run.py --figures-only
```

To run only the controller comparison and Fig. 5:

```sh
python python/comparison.py
```

The comparison runs the nominal and two perturbed plants with H-infinity feedback and full-state backstepping. It uses RK4 with a `1e-5` s step, exact propagation of the reference derivative filters, and zero initial states. Backstepping uses a 500 rad/s derivative filter. Both controllers use the same comparison reference, scaled by 40/41 relative to the tracking experiment in Figs. 3(b) and 4. The figure and RMSE table are saved in `python/figures` and `python/comparison_metrics.csv`.

## Files

- `python/run.py`: controller design, certificates, and nonlinear simulations.
- `python/design.py`, `verification.py`, `localization.py`: model construction and certificate calculations.
- `python/degree_comparison.py`, `sweep_test.py`: degree and finite-horizon studies.
- `python/plot_figures.py`, `plot_comparison.py`: paper figures.
- `python/comparison.py`: nominal and perturbed-plant controller comparison.
- `python/paper_data.npz`: stored model, controller, reference, and certificates.
