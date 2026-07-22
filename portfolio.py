"""splitQP demo: one Cholesky factorization prices a whole efficient frontier.

    uv run python portfolio.py --points 64

A fixed covariance matrix and constraint matrix define a family of Markowitz
QPs in which only the target-return bound changes.  Constructing one Solver
factors H once; a batch solve, 64 cold scalar solves, and a warm-started sweep
all reuse that factor.  Every printed claim is asserted before it is printed.
"""

import argparse

import numpy as np
import jax

import splitqp

parser = argparse.ArgumentParser()
parser.add_argument("--points", type=int, default=64, help="family size B")
B = parser.parse_args().points

# ------------------------------------------------------------- the QP family
# n assets with returns mu, exponentially correlated volatilities vol, and a
# position cap w_max.  Rows of A: budget (equality), target return (one-sided),
# then per-asset bounds; all data are explicit formulas, so the demo is
# deterministic.
n = 16
mu = np.linspace(0.02, 0.12, n)
vol = np.linspace(0.05, 0.30, n)
idx = np.arange(n)
Sigma = np.diag(vol) @ np.exp(-np.abs(idx[:, None] - idx[None, :]) / 8.0) @ np.diag(vol)
w_max = 0.30
A = np.vstack([np.ones(n), mu, np.eye(n)])

# Highest return reachable under the caps (greedy top holdings), and B targets.
r_hi = mu[::-1][:3] @ np.full(3, w_max) + mu[::-1][3] * (1 - 3 * w_max)
targets = np.linspace(mu.min(), 0.98 * r_hi, B)
q = np.zeros((B, n))
l = np.tile(np.concatenate([[1.0, 0.0], np.zeros(n)]), (B, 1))
u = np.tile(np.concatenate([[1.0, np.inf], np.full(n, w_max)]), (B, 1))
l[:, 1] = targets  # the only entry that varies across the family

# ------------------------------------------------- solve everything, one factor
solver = splitqp.Solver(Sigma, A)  # the single Cholesky factorization
MAX_IT = 20000

batch = solver.solve_batch(q, l, u, max_iter=MAX_IT)
jax.block_until_ready(batch)  # finish async device work before host reads

cold = [solver.solve(q[i], l[i], u[i], max_iter=MAX_IT) for i in range(B)]

warm = []
state = None
for i in range(B):  # ascending targets, each started from the previous solution
    r = solver.solve(q[i], l[i], u[i], init=state, max_iter=MAX_IT)
    warm.append(r)
    state = r.state

# ------------------------------------------- assert every claim, then print it
x = np.asarray(batch.x)  # one explicit host conversion for all demo checks
iters_batch = np.asarray(batch.iterations)
iters_cold = np.array([int(r.iterations) for r in cold])
iters_warm = np.array([int(r.iterations) for r in warm])

assert bool(np.all(np.asarray(batch.converged)))
assert all(bool(r.converged) for r in cold + warm)
assert solver.factorizations == 1, "every solve above must reuse one factor"

max_kkt = max(float(np.max(np.asarray(batch.r_primal))),
              float(np.max(np.asarray(batch.r_dual))))
assert max_kkt < 1e-5

batch_vs_scalar = max(float(np.max(np.abs(x[i] - np.asarray(cold[i].x))))
                      for i in range(B))
assert batch_vs_scalar < 1e-8
assert np.array_equal(iters_batch, iters_cold), "frozen lanes must match scalar stops"

assert iters_warm.sum() < iters_cold.sum(), "warm starts must shorten the sweep"
returns = x @ mu
risks = np.sqrt(np.einsum("bi,ij,bj->b", x, Sigma, x))
assert np.all(returns >= targets - 1e-5) and np.all(np.diff(risks) > -1e-9)

print(f"{B} QPs, {solver.factorizations} factorization")
print(f"max KKT residual: {max_kkt:.2e}")
print(f"batch/scalar difference: {batch_vs_scalar:.2e}")
print(f"cold iterations vs warm sequence: {iters_cold.sum()} vs {iters_warm.sum()}")
