# splitQP

splitQP solves families of dense convex quadratic programs that share the same
quadratic and constraint matrices while their linear term and bounds change.
It is intended for readers who want a compact, modifiable implementation of
ADMM factorization reuse, batching, and warm starts in JAX. The solver,
reference implementation, correctness tests, and portfolio example are
complete and have been checked on CPU and CUDA; this is an educational research
implementation, not a production solver.

## Quick start

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run python portfolio.py --points 64
```

A representative CPU run prints:

```text
64 QPs, 1 factorization
max KKT residual: 2.00e-06
batch/scalar difference: 3.44e-15
cold iterations vs warm sequence: 48435 vs 36184
```

The example asserts every reported result. The final floating-point digits may
vary across devices or separate XLA compilations. On a compatible NVIDIA
system, install the optional CUDA backend with `uv sync --extra cuda`.

## Problem

splitQP accepts the box-constrained form

```math
\begin{aligned}
\min_x \quad & \frac{1}{2}x^\top P x + q^\top x \\
\text{subject to} \quad & l \leq A x \leq u.
\end{aligned}
```

where $P \succeq 0$. A problem family keeps $P$ and $A$ fixed while $q$, $l$,
and $u$ vary. Equalities, one-sided constraints, and two-sided constraints all
use the same projection onto $[l,u]$.

## Main features

- one Cholesky factorization shared by every ADMM iteration and every member of
  a problem family;
- a single scalar `step` transformed into a leading-axis batch with `jax.vmap`;
- a compiled `jax.lax.while_loop` with independently stopped batch lanes;
- warm starts through the state $(x,z,y)$;
- optional named iteration traces for inspecting the solve, projection, and
  stopping decisions;
- a correctness ladder from a direct reference implementation to OSQP.

## Project structure

Read the files in this order:

1. [`reference.py`](reference.py) (61 lines) implements the six ADMM equations
   in eager JAX and calls a fresh dense solve at every iteration.
2. [`splitqp.py`](splitqp.py) (238 lines) adds the cached Cholesky factor, pure
   step, `vmap`, compiled loop, per-lane stopping, trace, and warm start.
3. [`portfolio.py`](portfolio.py) (87 lines) solves a 64-member Markowitz
   family and checks batch/scalar agreement and cold/warm iteration counts.
4. [`test.py`](test.py) (177 lines) contains the hand-audited step, trajectory
   comparisons, JAX transform checks, batch checks, and OSQP oracle.

## Design

For positive fixed scalars $\rho$ and $\sigma$, and $\alpha\in(0,2)$, setup
forms

```math
H = P + \sigma I + \rho A^\top A
```

and factorizes $H$ once. Each iteration then evaluates

```math
\begin{aligned}
H\tilde{x}^{k+1}
    &= \sigma x^k-q+A^\top(\rho z^k-y^k), \\
\tilde{z}^{k+1} &= A\tilde{x}^{k+1}, \\
x^{k+1} &= \alpha\tilde{x}^{k+1}+(1-\alpha)x^k, \\
\bar{z}^{k+1} &= \alpha\tilde{z}^{k+1}+(1-\alpha)z^k, \\
z^{k+1} &= \Pi_{[l,u]}\left(\bar{z}^{k+1}+y^k/\rho\right), \\
y^{k+1} &= y^k+\rho\left(\bar{z}^{k+1}-z^{k+1}\right).
\end{aligned}
```

The first equation is solved with the cached Cholesky factor; the implementation
never forms $H^{-1}$. Fixed $\rho$ is deliberate: changing it would change $H$
and require a new factorization.

Termination is based on the original-coordinate KKT residuals

```math
r_{\mathrm{pri}} = Ax-z,
\qquad
r_{\mathrm{dual}} = Px+q+A^\top y,
```

with absolute-plus-relative infinity-norm thresholds

```math
\begin{aligned}
\epsilon_{\mathrm{pri}}
    &= \epsilon_{\mathrm{abs}}
       +\epsilon_{\mathrm{rel}}
        \max\!\left(\lVert Ax\rVert_\infty,\lVert z\rVert_\infty\right), \\
\epsilon_{\mathrm{dual}}
    &= \epsilon_{\mathrm{abs}}
       +\epsilon_{\mathrm{rel}}
        \max\!\left(\lVert Px\rVert_\infty,
                     \lVert A^\top y\rVert_\infty,
                     \lVert q\rVert_\infty\right).
\end{aligned}
```

The projection also enforces the normal-cone condition
$z=\Pi_{[l,u]}(z+y/\rho)$ at every accepted iterate.

### JAX implementation

The module enables float64 before creating arrays
([`splitqp.py:17`](splitqp.py#L17)). Setup stores only the factor array returned
by `cho_factor` ([`splitqp.py:112`](splitqp.py#L112)), and the step reuses it via
`cho_solve((factor, True), rhs)` ([`splitqp.py:122`](splitqp.py#L122)). The
literal `True` keeps the triangular orientation static under JAX transforms.

`jax.vmap` shares $P$, $A$, and the factor while mapping each problem's data
and state ([`splitqp.py:147`](splitqp.py#L147)). The compiled solve uses a
fixed-shape `lax.while_loop`; a done mask freezes each lane at its first
accepted stopping point ([`splitqp.py:181`](splitqp.py#L181)). The trace path
calls the same jitted step from a Python loop and performs an explicit
device-to-host synchronization ([`splitqp.py:230`](splitqp.py#L230)).

## Validation

Run the compact correctness suite on CPU and on the default JAX device:

```bash
JAX_PLATFORMS=cpu uv run python test.py
uv run python test.py
```

The CPU run prints:

```text
trajectory okay (3751 iterations, 30 audited against reference)
batch okay (5 QPs, iterations [722, 3041, 1886, 1189, 23410], 1 factorization)
OSQP okay (objectives and KKT residuals agree)
all okay
```

The tests compare an exact-fraction first step and every named intermediate of
the cached path with `reference.py`. They also compare eager and jitted steps,
`jit(vmap(step))` and scalar stacks, compiled and traced stopping behavior, and
batched and independently stopped scalar solves. OSQP is used only as an
external final-point oracle.

## Limitations

- dense, feasible, well-scaled convex QPs only;
- fixed scalar $\rho$ and fixed $P,A$ after setup;
- no sparse matrices, scaling, adaptive penalty, infeasibility certificate,
  polishing, or matrix updates;
- no autodiff layer, custom GPU kernel, multi-device execution, or performance
  guarantee;
- reaching `max_iter` returns `converged=False`; infeasibility is not inferred.

No correctness issue is known for the documented problem class. Inputs outside
these assumptions are not characterized as production solver statuses.

## References

- [Boyd et al., *Distributed Optimization and Statistical Learning via the
  Alternating Direction Method of Multipliers*](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf)
  for the ADMM state, residual interpretation, stopping tests, and warm starts.
- [Stellato et al., *OSQP: an Operator Splitting Solver for Quadratic
  Programs*](https://arxiv.org/abs/1711.08013) for the box form, proximal term,
  relaxation, factor reuse, and final-point oracle.
- [Bishop et al., *ReLU-QP*](https://arxiv.org/abs/2311.18056) for the
  fixed-point and batched interpretation of this iteration.
- [JAX documentation](https://docs.jax.dev/) and
  [JAXopt BoxOSQP at the inspected commit](https://github.com/google/jaxopt/tree/fd852822e8aed93c2e067a348ce5cf3f236d3ed2)
  for public transform and factor/solve patterns.

## License

No license file is currently included. Choose a license before redistribution
or publication.
