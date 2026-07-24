# splitQP

A tiny JAX solver for families of dense convex quadratic programs that share a
fixed quadratic and constraint matrix. It implements proximal ADMM, and its one
idea is factorization reuse: `Solver(P, A)` factors a single matrix at
construction and reuses it for every iteration and every member of the family.
This is a compact educational research implementation, not a production solver.

## Problem

```math
\min_x \ \tfrac{1}{2}x^\top P x + q^\top x \quad\text{s.t.}\quad l \leq A x \leq u,
```

with $P \succeq 0$. A family keeps $P$ and $A$ fixed while $q$, $l$, $u$ vary.
Equalities ($l = u$), one-sided, two-sided, infinite, and empty constraint blocks
all use the same projection onto $[l, u]$.

## Method

`Solver(P, A)` forms $H = P + \sigma I + \rho A^\top A$ and factors it once with a
Cholesky. Every ADMM iteration, every $(q, l, u)$, `solve_batch`, and
`solve_sequence` reuse that one triangular factor. A single scalar `step` is mapped
over a family with `jax.vmap` and driven by a compiled `jax.lax.while_loop`;
`solve_sequence` threads an ordered warm-started family through a single
`jax.lax.scan`. $\rho$ is fixed.

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
result = solver.solve(q, l, u)              # one QP
family = solver.solve_batch(qs, ls, us)     # B independent QPs sharing the factor
seq = solver.solve_sequence(qs, ls, us)     # an ordered warm family, one compiled program
warm = solver.solve(q2, l2, u2, init=result.state)

result.status         # "solved" | "max_iter" | "numerical_error" | "invalid_problem"
solver.factorizations # 1 after construction
```

## Notebook

[`split.ipynb`](split.ipynb) builds a small ordered family from one factorization
and solves it with `solve`, `solve_batch`, and `solve_sequence`, showing that batch
treats members independently while the sequence follows an ordered warm start:

```bash
uv run --group demo jupyter lab split.ipynb
```

A small optional local timing script is included in `bench.py`:

```bash
uv run --group bench python bench.py          # portfolio + one synthetic family
uv run --group bench python bench.py --full   # more families
```

## Limitations

- dense, well-scaled convex QPs only; a fixed scalar $\rho$ and fixed $P, A$ after
  construction;
- the fixed $\rho$ must match the problem scale: on ill-conditioned families the
  default $\rho$ reaches `max_iter`, and the API lets a caller set $\rho$;
- no sparse matrices, equilibration, adaptive penalty, infeasibility certificate,
  polishing, autodiff, or custom GPU kernel;
- non-finite iterates surface as `numerical_error` and invalid or non-convex data
  as `invalid_problem`; neither is returned as a successful solve.

## Status

This is a compact educational research implementation, not a production solver.
The implementation is feature-complete for its intended scope.

## References

- [Stellato et al., *OSQP*](https://arxiv.org/abs/1711.08013)
- [Boyd et al., *ADMM*](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf)
- [JAX documentation](https://docs.jax.dev/)
- [qpbenchmark](https://github.com/qpsolvers/qpbenchmark), a broader benchmark
  framework for quadratic-programming solvers.
