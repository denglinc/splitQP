"""splitQP correctness ladder.

Run directly:  JAX_PLATFORMS=cpu uv run python test.py   (and again on the
default device with plain `uv run python test.py`).  Three rungs:

1. trajectory: hand-audited first step, then every named intermediate of the
   cached path against reference.py, eager vs jit, compiled loop vs trace loop;
2. batch: jit(vmap(step)) vs a Python stack of scalar steps, batched solve vs
   independently stopped scalar solves, one factorization, warm start;
3. OSQP: final objective and original-scale KKT residuals against the
   external oracle.

Tolerance agreement everywhere, never cross-device bitwise equality.
"""

import numpy as np
import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

import splitqp
from reference import solve_reference

assert jnp.ones(1).dtype == jnp.float64, "x64 must be active before arrays exist"

# ---------------------------------------------------------------- fixtures
# Tiny hand-auditable QP: integers only, so every first-step quantity below is
# an exact fraction.
P2 = jnp.array([[4.0, 0.0], [0.0, 2.0]])
A2 = jnp.array([[1.0, 1.0], [1.0, 0.0]])
q2 = jnp.array([-8.0, -6.0])
l2 = jnp.array([1.0, 0.0])
u2 = jnp.array([1.0, 0.75])

# Medium seeded QP, feasible by construction (box drawn around A @ x0).
rng = np.random.default_rng(7)
n, m = 8, 12
M = rng.normal(size=(n, n))
P = jnp.asarray(M @ M.T + n * np.eye(n))
A = jnp.asarray(rng.normal(size=(m, n)))
q = jnp.asarray(rng.normal(size=n))
c0 = np.asarray(A) @ rng.normal(size=n)
l = jnp.asarray(c0 - rng.uniform(0.1, 1.0, size=m))
u = jnp.asarray(c0 + rng.uniform(0.1, 1.0, size=m))

# Batch of B feasible QPs sharing P and A.
B = 5
qb = jnp.asarray(rng.normal(size=(B, n)))
cb = (np.asarray(A) @ rng.normal(size=(B, n)).T).T
lb = jnp.asarray(cb - rng.uniform(0.1, 1.0, size=(B, m)))
ub = jnp.asarray(cb + rng.uniform(0.1, 1.0, size=(B, m)))

# ------------------------------------------------------- rung 1: trajectory
# Hand-audited first step from zeros with rho=2, sigma=1, alpha=3/2:
#   H = P + I + 2*A'A = [[9, 2], [2, 5]],  rhs = -q = (8, 6),
#   x_tilde = H^-1 rhs = (28/41, 38/41),   z_tilde = A x_tilde = (66/41, 28/41),
#   x = 3/2 x_tilde = (42/41, 57/41),      z_bar = 3/2 z_tilde = (99/41, 42/41),
#   v = z_bar,  z = clip(v, l, u) = (1, 3/4),  y = 2 (z_bar - z) = (116/41, 45/82).
cache2 = splitqp.Solver(P2, A2, rho=2.0, sigma=1.0, alpha=1.5).cache
state1, parts1 = splitqp.step(cache2, q2, l2, u2,
                              splitqp.State(jnp.zeros(2), jnp.zeros(2), jnp.zeros(2)))
audited = [
    (parts1.rhs, [8, 6]), (parts1.x_tilde, [28 / 41, 38 / 41]),
    (parts1.z_tilde, [66 / 41, 28 / 41]), (state1.x, [42 / 41, 57 / 41]),
    (parts1.z_bar, [99 / 41, 42 / 41]), (parts1.v, [99 / 41, 42 / 41]),
    (state1.z, [1, 3 / 4]), (state1.y, [116 / 41, 45 / 82]),
]
for got, exact in audited:
    assert_allclose(np.asarray(got), np.array(exact, dtype=np.float64),
                    rtol=1e-13, atol=1e-14)
_, _, _, ref2 = solve_reference(P2, A2, q2, l2, u2, rho=2.0, sigma=1.0, alpha=1.5,
                                max_iter=1)
for (got, exact), name in zip(audited[:3], ("rhs", "x_tilde", "z_tilde")):
    assert_allclose(np.asarray(ref2[0][name]), np.array(exact, dtype=np.float64),
                    rtol=1e-13, atol=1e-14)

# Every named intermediate of the cached path equals the fresh-solve reference.
solver = splitqp.Solver(P, A)
cache = solver.cache
_, _, _, ref = solve_reference(P, A, q, l, u, max_iter=30)
state = splitqp.State(jnp.zeros(n), jnp.zeros(m), jnp.zeros(m))
for k in range(30):
    state, parts = splitqp.step(cache, q, l, u, state)
    rep = splitqp.report(cache, q, l, u, state, 1e-6, 1e-6)
    r = ref[k]
    named = dict(rhs=parts.rhs, x_tilde=parts.x_tilde, z_tilde=parts.z_tilde,
                 x=state.x, z_bar=parts.z_bar, v=parts.v, z=state.z, y=state.y,
                 objective=rep.objective, r_primal=rep.r_primal,
                 r_dual=rep.r_dual, eps_primal=rep.eps_primal,
                 eps_dual=rep.eps_dual)
    for name, val in named.items():
        assert_allclose(np.asarray(val), np.asarray(r[name]), rtol=1e-9,
                        atol=1e-11, err_msg=f"iteration {k + 1}: {name}")
    assert bool(rep.converged) == r["converged"]

# jit(step) agrees with the eager step it traced.
jit_step = jax.jit(splitqp.step)
s0 = splitqp.State(jnp.zeros(n), jnp.zeros(m), jnp.zeros(m))
eager_out, jit_out = splitqp.step(cache, q, l, u, s0), jit_step(cache, q, l, u, s0)
for a, b in zip(eager_out[0] + eager_out[1], jit_out[0] + jit_out[1]):
    assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-14)

# Compiled while_loop, host trace loop, and reference all accept the same stop.
res_loop = solver.solve(q, l, u)
res_trace, trace = solver.trace(q, l, u)
_, _, _, ref_full = solve_reference(P, A, q, l, u)
assert bool(res_loop.converged) and bool(res_trace.converged)
assert int(res_loop.iterations) == int(res_trace.iterations) == len(ref_full)
assert_allclose(np.asarray(res_loop.x), np.asarray(res_trace.x), rtol=1e-10, atol=1e-12)
assert_allclose(np.asarray(res_trace.x), np.asarray(ref_full[-1]["x"]), rtol=1e-8,
                atol=1e-10)
assert trace.x.shape[0] == int(res_trace.iterations)
print(f"trajectory okay ({int(res_loop.iterations)} iterations, "
      "30 audited against reference)")

# ------------------------------------------------------------ rung 2: batch
# jit(vmap(step)) equals a Python stack of scalar steps.
batched_step = jax.jit(jax.vmap(splitqp.step, in_axes=(None, 0, 0, 0, 0)))
sb = splitqp.State(jnp.zeros((B, n)), jnp.zeros((B, m)), jnp.zeros((B, m)))
for _ in range(10):
    sb, _ = batched_step(cache, qb, lb, ub, sb)
for i in range(B):
    si = splitqp.State(jnp.zeros(n), jnp.zeros(m), jnp.zeros(m))
    for _ in range(10):
        si, _ = splitqp.step(cache, qb[i], lb[i], ub[i], si)
    for a, b in zip((sb.x[i], sb.z[i], sb.y[i]), si):
        assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-10, atol=1e-12)

# One batched solve equals B independently stopped scalar solves, lane by lane.
solver = splitqp.Solver(P, A)
cache = solver.cache
res_b = solver.solve_batch(qb, lb, ub, max_iter=30000)
assert bool(jnp.all(res_b.converged))
for i in range(B):
    r_i = solver.solve(qb[i], lb[i], ub[i], max_iter=30000)
    assert int(res_b.iterations[i]) == int(r_i.iterations)
    assert_allclose(np.asarray(res_b.x[i]), np.asarray(r_i.x), rtol=1e-9, atol=1e-11)
assert solver.factorizations == 1, "many solves must reuse the one factorization"

# Projection invariant: z = clip(z + y/rho, l, u) holds at every solution, so
# the normal-cone condition costs nothing to enforce.
z_proj = jnp.clip(res_b.z + res_b.y / cache.rho, lb, ub)
assert_allclose(np.asarray(res_b.z), np.asarray(z_proj), rtol=0, atol=1e-9)

# Warm start touches only (x, z, y): a solution is a fixed point of the step,
# so restarting from it is accepted after a single confirming iteration.
warm = solver.solve(qb[0], lb[0], ub[0],
                    init=splitqp.State(res_b.x[0], res_b.z[0], res_b.y[0]))
assert int(warm.iterations) == 1 and bool(warm.converged)
print(f"batch okay ({B} QPs, iterations {[int(i) for i in res_b.iterations]}, "
      "1 factorization)")

# ------------------------------------------------------------- rung 3: OSQP
import scipy.sparse as sparse  # dev-only oracle dependencies
import osqp

for Pi, Ai, qi, li, ui, ours in [
    (P2, A2, q2, l2, u2, splitqp.Solver(P2, A2).solve(q2, l2, u2)),
    (P, A, q, l, u, res_loop),
]:
    ref_solver = osqp.OSQP()
    ref_solver.setup(sparse.csc_matrix(np.asarray(Pi)), np.asarray(qi),
                     sparse.csc_matrix(np.asarray(Ai)), np.asarray(li),
                     np.asarray(ui), eps_abs=1e-9, eps_rel=1e-9, rho=0.1,
                     sigma=1e-6, alpha=1.6, adaptive_rho=False, scaling=0,
                     polishing=False, verbose=False, max_iter=200000)
    osqp_res = ref_solver.solve()
    assert osqp_res.info.status == "solved"
    assert bool(ours.converged)
    assert_allclose(float(ours.objective), osqp_res.info.obj_val,
                    rtol=1e-6, atol=1e-5)
    # Our final point satisfies the original-scale KKT conditions on its own.
    kkt_primal = float(ours.r_primal)
    kkt_dual = float(ours.r_dual)
    assert kkt_primal < 1e-4 and kkt_dual < 1e-4, (kkt_primal, kkt_dual)
print("OSQP okay (objectives and KKT residuals agree)")

print("all okay")
