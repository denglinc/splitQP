# splitQP

A tiny JAX solver for families of dense convex quadratic programs that share a
fixed quadratic and constraint matrix. It implements proximal ADMM and is built
around two ideas: the metric that iteration runs in, and factorization reuse —
`Solver(P, A)` factors a single matrix at construction and reuses it for every
iteration and every member of the family. This is a compact educational research
implementation, not a production solver.

## Problem & Method

```math
\min_x \ \tfrac{1}{2}x^\top P x + q^\top x \quad\text{s.t.}\quad l \leq A x \leq u,
```

with $P \succeq 0$. A family keeps $P$ and $A$ fixed while $q$, $l$, $u$ vary.
Equalities ($l = u$), one-sided, two-sided, infinite, and empty constraint blocks
all use the same projection onto $[l, u]$.

`Solver(P, A)` forms $H = P + \sigma I + A^\top \mathrm{diag}(\rho) A$ and
factors it once with a Cholesky. Every ADMM iteration, every $(q, l, u)$,
`solve_batch`, and `solve_sequence` reuse that one triangular factor. A single scalar
`step` is mapped over a family with `jax.vmap` and driven by a compiled
`jax.lax.while_loop`; `solve_sequence` threads an ordered warm-started family through
a single `jax.lax.scan`. $\rho$ is fixed and may be scalar or diagonal.

## Preconditioning

![The same QP under raw and equilibrated ADMM metrics.](assets/preconditioning.png)

As a first-order method, ADMM carries no curvature information, so its iteration count is set by the geometry
it is given rather than by the problem itself. The QP above is one feasible set
written twice. In raw coordinates its constraint rows differ by two orders of
magnitude, so the two penalty directions differ by $10^4$, and no scalar $\rho$ suits
both: the default takes 7256 iterations to reach
$\lVert x^k - x^\star\rVert_2 < 10^{-5}$, and the best of a 49-point grid still takes
174 — tuning moves the bottleneck from one row to the other instead of removing it.
Equilibrate the rows, or equivalently pass that same rescaling as the diagonal ADMM
penalty $\mathrm{diag}(0.01, 100)$; the two runs trace the same trajectory to
$7\times10^{-16}$ and finish in 23. Preconditioning is the choice of a coordinate
system, or a metric, in which the problem is isotropic; for a first-order method that
choice is not a detail, it is most of the cost.

## Install

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python -c "import splitqp; print(splitqp.Solver)"
```

`uv sync --extra cuda` installs the optional CUDA backend; CPU float64 is the
canonical numerical setting.

## Usage

```python
import splitqp

solver = splitqp.Solver(P, A)               # the one factorization
metric = splitqp.Solver(P, A, rho=rho_m)    # a fixed diagonal penalty, shape (m,)
result = solver.solve(q, l, u)              # one QP
family = solver.solve_batch(qs, ls, us)     # B independent QPs sharing the factor
seq = solver.solve_sequence(qs, ls, us)     # an ordered warm family, one compiled program
warm = solver.solve(q2, l2, u2, init=result.state)

result.status         # "solved" | "max_iter" | "numerical_error" | "invalid_problem"
solver.factorizations # 1 after construction
```

## Notebook

[`Preconditioning.ipynb`](Preconditioning.ipynb) builds a small ordered family from
one factorization and solves it with `solve`, `solve_batch`, and `solve_sequence`,
showing that batch treats members independently while the sequence follows an ordered
warm start. It then runs the preconditioning experiment shown at the top of this page.
Both halves print iteration logs rather than only final counts:

```bash
uv run --group demo jupyter lab Preconditioning.ipynb
```

A small optional local timing script is included in `bench.py`:

```bash
uv run --group bench python bench.py          # portfolio + one synthetic family
uv run --group bench python bench.py --full   # more families
```

## Limitations

- dense, well-scaled convex QPs only; a fixed scalar or diagonal $\rho$ and fixed
  $P, A$ after construction;
- the fixed $\rho$ must match the problem scale: on ill-conditioned families the
  default $\rho$ reaches `max_iter`, and the API lets a caller set $\rho$;
- no sparse matrices, equilibration, adaptive penalty, infeasibility certificate,
  polishing, autodiff, or custom GPU kernel;
- non-finite iterates surface as `numerical_error` and invalid or non-convex data
  as `invalid_problem`; neither is returned as a successful solve.

## References

- [Boyd et al., *Distributed Optimization and Statistical Learning via the
  Alternating Direction Method of Multipliers*](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf)
  for the ADMM state, residual interpretation, stopping tests, warm starts, and the
  general augmenting term behind the diagonal penalty.
- [Stellato et al., *OSQP: an Operator Splitting Solver for Quadratic
  Programs*](https://arxiv.org/abs/1711.08013) for the box form, proximal term,
  relaxation, factor reuse, final-point comparison, and its data scaling and
  diagonal penalty selection.
- [Bishop et al., *ReLU-QP*](https://arxiv.org/abs/2311.18056) for the
  fixed-point and batched interpretation of this iteration.
- [JAX documentation](https://docs.jax.dev/) and
  [JAXopt's BoxOSQP](https://github.com/google/jaxopt) for transform and
  factor/solve patterns in JAX.
- [qpbenchmark](https://github.com/qpsolvers/qpbenchmark), a broader benchmark
  framework for quadratic-programming solvers.
