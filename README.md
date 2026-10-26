# RCCG

Code for [*Model-Based Koopman-Galerkin Lifting for Guaranteed H-Infinity
Output Feedback Control of Nonlinear Systems*](https://arxiv.org/abs/2610.07299),
by Burak Kürkçü, Christopher Phan, Aleksandar Zecevic, and Maryam Khanbaghi.

Use Python 3.12 or 3.13. From a terminal:

```sh
git clone https://github.com/kurkcub/RCCG.git
cd RCCG
python -m pip install -r requirements.txt
python run.py
```

This builds the models and controllers, runs the numerical experiments,
checks the regional certificates, and saves the manuscript figures in
`figures/`. The fixed inputs and MPC training data are included.
The interval calculations can take several minutes.

After a completed run, redraw the figures with:

```sh
python run.py --figures-only
```

Recheck the certificates with:

```sh
python run.py --certificates-only
```

The Galerkin implementation is in `python/`, the controller comparisons
in `comparison/`, and the certificate checks in `certificates/`.
[Comparison details](comparison/README.md) include the controller settings
and training-data regeneration.
