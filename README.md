# splitQP

**One Cholesky, many QPs.**

splitQP is a tiny JAX ADMM solver for a family of dense box QPs with fixed
`P, A` and changing `q, l, u`. Factor once; then every iteration is
`solve -> clip -> dual correction`, for one QP or a whole batch. I wanted that
reuse to be visible in the code and the trace — not buried under a general
solver API.

## feel the magic

```bash
uv sync
uv run python portfolio.py --points 64
```

A representative CUDA run prints:

```text
64 QPs, 1 factorization
max KKT residual: 2.00e-06
batch/scalar difference: 9.49e-15
cold iterations vs warm sequence: 48435 vs 36184
```

Every number above is asserted by the demo before it is printed. The last few
floating-point digits can vary across devices or separate XLA compilations.

## the files that matter

Read them in this order:

1. [`reference.py`](reference.py) (61 lines) — the six scalar ADMM equations,
   eager JAX, a fresh dense solve every iteration;
2. [`splitqp.py`](splitqp.py) (238 lines) — the same equations with one cached
   Cholesky, one pure step, leading-axis `vmap`, a compiled `while_loop`, and
   warm-startable `(x, z, y)` state;
3. [`portfolio.py`](portfolio.py) (87 lines) — one complete family experiment:
   batch versus scalar solves and a cold/warm sweep over 64 target returns;
4. [`test.py`](test.py) (177 lines) — hand-audited first step, reference
   trajectory, batch, and OSQP oracle.

## how it works

The only problem form is OSQP's box QP:

```text
minimize    1/2 x'Px + q'x
subject to  l <= Ax <= u
```

For fixed scalars `rho > 0`, `sigma > 0`, `alpha in (0, 2)`, setup forms

```text
H = P + sigma*I + rho*A'A
```

and Cholesky-factorizes it once. Every iteration is then

```text
x_tilde = solve(H, sigma*x - q + A'(rho*z - y))
z_tilde = A x_tilde
x       = alpha*x_tilde + (1-alpha)*x
z_bar   = alpha*z_tilde + (1-alpha)*z
z       = clip(z_bar + y/rho, l, u)
y       = y + rho*(z_bar - z)
```

with OSQP's absolute-plus-relative infinity-norm residual test deciding when
to stop. Fixed `rho` is an opinion, not an oversight: adaptive `rho` changes
`H` and would invalidate the one-factorization experiment this project exists
to show. The projection line also enforces the normal-cone KKT condition by
construction — `z = clip(z + y/rho, l, u)` holds at every iterate, and
`test.py` checks it.

**JAX in this file.** `jnp` supplies immutable float64 arrays
([`splitqp.py:17`](splitqp.py#L17)), so a step returns a new `(x, z, y)` state
instead of mutating one. Setup keeps only the factor array from `cho_factor`
([`splitqp.py:112`](splitqp.py#L112)) and the step reuses it through
`cho_solve((factor, True), rhs)` ([`splitqp.py:122`](splitqp.py#L122)).
`jax.vmap` turns that one scalar step into a leading-axis family, with
`in_axes=None` sharing `P`, `A`, and the factor while `0` maps each QP's data
([`splitqp.py:147`](splitqp.py#L147)). The compiled solve is a fixed-shape
`lax.while_loop` whose per-lane done mask freezes each lane at its first
accepted stop ([`splitqp.py:181`](splitqp.py#L181)); the explanatory trace
drives the same jitted step from a host loop and synchronizes explicitly
([`splitqp.py:230`](splitqp.py#L230)). JIT compiles a pure fixed-shape
program per input shape; it does not remove compilation cost.

## correctness

```bash
JAX_PLATFORMS=cpu uv run python test.py
uv run python test.py
```

Both commands print, with identical iteration counts:

```text
trajectory okay (3751 iterations, 30 audited against reference)
batch okay (5 QPs, iterations [722, 3041, 1886, 1189, 23410], 1 factorization)
OSQP okay (objectives and KKT residuals agree)
all okay
```

The primary oracle is trajectory parity: an exact-fraction hand audit of the
first step, then every named intermediate (`rhs`, `x_tilde`, `z_tilde`,
`z_bar`, `v`, `z`, `y`, residuals, tolerances) of the cached path against
`reference.py`, eager against `jit`, `jit(vmap(step))` against a Python stack
of scalar steps, and the compiled loop against the host trace loop. OSQP
checks only the final objective and KKT point. A solver that reaches a similar
`x` through compensating mistakes fails this ladder.

## research roots

- [Boyd et al., *Distributed Optimization and Statistical Learning via the
  Alternating Direction Method of Multipliers*](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf)
  — ADMM state, the optimality meaning of primal/dual residuals,
  absolute-plus-relative stopping, and warm-start paths;
- [Stellato et al., *OSQP: an Operator Splitting Solver for Quadratic
  Programs*](https://arxiv.org/abs/1711.08013) — the box form, proximal
  `sigma`, relaxed updates, cached factorization, and the final-point oracle;
- [Bishop et al., *ReLU-QP*](https://arxiv.org/abs/2311.18056) — the
  weight-tied fixed-point and batched interpretation of this exact iteration,
  not its explicit inverse or GPU implementation;
- [JAX documentation](https://docs.jax.dev/) and
  [JAXopt BoxOSQP at the inspected commit](https://github.com/google/jaxopt/tree/fd852822e8aed93c2e067a348ce5cf3f236d3ed2)
  — public `jit`/`vmap`/factor-solve patterns and factor-in-state, not
  JAXopt's framework, pytrees, implicit differentiation, adaptive rho, or
  certificates.

## this is not OSQP

Dense, well-scaled, feasible convex box QPs only, with a fixed scalar `rho`.
No sparse matrices, Ruiz scaling, adaptive rho, infeasibility certificates,
polishing, matrix updates, autodiff, custom GPU kernels, multi-device
execution, or production guarantees. An infeasible problem simply stops at
`max_iter` with `converged=False`; it is never reported as solved. The same
JAX program runs on CPU or one GPU (both commands above are exercised), but
the tiny default demo makes no GPU speed claim.

## license

No license file yet — choose one before publishing.
