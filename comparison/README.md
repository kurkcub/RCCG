# Four-controller comparison

Use Python 3.13.5 with the packages in the root `requirements.txt`. From this
folder, after the root `python run.py` has completed, run:

```sh
python run.py
python plot.py
```

The nine 80 s tests use the same nominal nonlinear plant, reference knots and
zero initial states. The scripts write the error table and trajectories to
`results/`, and the manuscript figures to `figures/`.

The controllers are Galerkin degree-3 H-infinity feedback, ordinary Jacobian
H-infinity feedback, backstepping with all four gains equal to 8 and filter
frequency 500 rad/s, and Koopman-based hybrid MPC [49]. Both H-infinity
controllers use only the tracking error. Backstepping and Koopman-based hybrid
MPC use all four physical states. MPC receives one second of reference preview.

`inputs.npz` contains the reference knots and backstepping parameters.
`design_data.py` loads the Galerkin controller from `../python/model_data.npz`
and the Jacobian controller and MPC gains from `generated/`. The simulations
save their exact inputs in `results/controllers_used.npz`. Backstepping uses the polynomial
`(s+8)^4+3*(s+8)^2+1 = s^4+32*s^3+387*s^2+2096*s+4289`.
References have 0.02 s knots. Output records use 0.005 s spacing. The integration
steps are 0.0005 s for H-infinity feedback and MPC, and 0.00001 s for
backstepping.

To reproduce the Jacobian controller and the Koopman-based hybrid MPC model:

```sh
python design_jacobian.py
python fit_mpc.py
```

The generated matrices are saved in `generated/` and are used by the comparison.
After changing a design, run `python ../certificates/build.py` and
`python ../certificates/check.py` before regenerating the figures.
The model fit uses the
supplied noiseless one-step pairs in `mpc_training.npz`: 72,000 pairs fit the
data model and 8,000 determine the mixing weight. The physics predictor uses
the nominal model, the two actuator coordinates and 27 Hermite products of
total degrees 1 through 6 in the angle and angular velocity. Its Gaussian
integrals are evaluated by quadrature. The hybrid predictor is
`Kp + gamma*(Ke-Kp)`, with `gamma` approximately 0.981. The MPC settings are
H=200, running tracking weight 10, terminal weight 100 and command weight 1e-6.
Its first command is fixed to the previously applied input, and the next
command is planned from the measured state and future reference.

The training pairs can also be regenerated:

```sh
python fit_mpc.py --regenerate-data
```

State coverage comes from ten 40 s nominal simulations with seed 481910.
At each stored state a new input is drawn independently from [-16,16] with
seed 481911 and held for 0.005 s to obtain the next state. No measurement noise
is added.

The containment panel reads `certificate_summary.json`. The certificate
calculation is provided in `../certificates/`. `design_jacobian.py` uses the
synthesis routines in `../python/design.py`.

[49] D. Bruder, D. Bombara and R. J. Wood, “A Koopman-based residual modeling
approach for the control of a soft robot arm,” *The International Journal of
Robotics Research*, 44(3), 388–406, 2025.
[Paper](https://doi.org/10.1177/02783649241272114) ·
[Authors' code](https://github.com/free-laboratory/koopman-residual).
