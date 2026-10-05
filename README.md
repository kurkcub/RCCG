# Model-Based Galerkin Control

Code for *Model-Based Galerkin Lifting with Exact LTI Decomposition for Guaranteed H-infinity Output Feedback Control of Nonlinear Systems*, by Burak Kürkçü, Christopher Phan, Aleksandar Zecevic, and Maryam Khanbaghi.

The paper uses Python for the controller comparison. An alternative MATLAB/Simulink implementation reproduces the comparison and Fig. 5. Python also provides the Galerkin design, regional certificates, and Figs. 1–4. Localization uses a cubic lifted auxiliary realization with the same controller.

## Python

No MATLAB or Simulink license is needed. Tested with Python 3.12. From the repository folder:

```sh
python -m pip install -r python/requirements.txt
python python/run.py
```

This rebuilds the Galerkin models and controller, checks the regional certificates, runs the nonlinear simulations and degree study, and saves Figs. 1–5 in `python/figures`. The comparison uses the same physical plants, reference, and controllers as the Simulink model. It also exports `python/Galerkin_Control.mat` for MATLAB.

To redraw saved results:

```sh
python python/run.py --figures-only
```

To run only the alternative Python comparison and Fig. 5:

```sh
python python/comparison.py
```

## MATLAB / Simulink

Tested with MATLAB R2026a and Simulink. From the repository folder in MATLAB:

```matlab
cd matlab
[out, metrics] = run_comparison;
```

This runs the nominal and two perturbed-plant comparisons used in the paper with H-infinity feedback and full-state backstepping. The figure and RMSE table are saved in `matlab/figures`. Close another open model named `Hinf_BS_Comparison` before running this copy.

The model uses `ode23tb` with relative tolerance `1e-9`, absolute tolerance `1e-12`, and a maximum step of 0.02 s. Outputs are recorded every 0.001 s. These tolerances resolve the small nominal tracking errors.

`init_comparison.m` initializes the model workspace when the model opens and before each run. It loads the reference from `comparison_data.mat` and the controller from the Python export when available, otherwise from the same MAT file. Backstepping uses a 500 rad/s derivative filter. The nominal small-signal tracking bandwidths (-3 dB, reference to angle) are 225.57 rad/s for backstepping and 255.54 rad/s for H-infinity. Both controllers use the same comparison reference, scaled by 40/41 relative to the tracking experiment in Figs. 3(b) and 4. Parameter-perturbation tests are empirical comparisons outside the nominal regional guarantees.

## Python–Simulink agreement

The alternative Python comparison uses RK4 with a `1e-5` s step and exact propagation of the reference derivative filters. Both implementations use the same physical models, controller matrices, reference, zero initial states, and 500 rad/s backstepping filter.

Over 175 s, on the common 0.001 s output grid, the largest angle difference across the six nominal/P1/P2 trajectories is `4.66e-8` rad. The nominal backstepping time-weighted RMSE is `3.062538e-5` rad in Simulink and `3.062783e-5` rad in Python, a difference of `2.45e-9` rad (about 0.008%). RMSE and maximum absolute errors agree at the precision reported in the paper. The manuscript reports the Simulink results, while Python also exports Fig. 5 from its own simulated trajectories.

## Files

- `python/run.py`: controller design, certificates, and nonlinear simulations.
- `python/design.py`, `verification.py`, `localization.py`: model construction and certificate calculations.
- `python/degree_comparison.py`, `sweep_test.py`: degree and finite-horizon studies.
- `python/plot_figures.py`, `plot_comparison.py`: paper figures.
- `python/comparison.py`: nominal and perturbed-plant controller comparison.
- `python/export_matlab.py`: MATLAB export.
- `python/paper_data.npz`: stored model, controller, reference, and certificates.
- `matlab/Hinf_BS_Comparison.slx`: comparison model.
- `matlab/run_comparison.m`, `init_comparison.m`, `plotCartPendComparison.m`: setup, simulation, metrics, and figure.
- `matlab/comparison_data.mat`: controller matrices and comparison reference.
