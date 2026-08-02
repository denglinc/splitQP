# PIPG

[`PIPG.ipynb`](PIPG.ipynb) starts from the solver architecture used in splitQP and asks a
simple question: when should a reusable implicit solve give way to
explicit proportional-integral primal-dual feedback?

The notebook develops PIPGeq, general-cone PIPG, xPIPG, infeasibility
signals, and projection-preserving preconditioning through a sequence of
JAX experiments.

Interestingly, this line of work has a very concrete onboard application.
[Kamath et al., *Customized Real-Time First-Order Methods for Onboard Dual
Quaternion-based 6-DoF Powered-Descent Guidance* (AIAA
2023-2003)](https://doi.org/10.2514/6.2023-2003) place a customized PIPG
solver inside sequential conic optimization for real-time rocket landing
guidance. In the later [full
study](https://arxiv.org/abs/2508.10439), the generated C implementation
was run on the NASA SPLICE Descent and Landing Computer in
hardware-in-the-loop tests, averaging about 0.59 seconds per guidance
update.

The companion
[Convexification](https://github.com/denglinc/Convexification) repository
explores the lossless and successive convexification techniques that
produce these structured trajectory-optimization subproblems.
