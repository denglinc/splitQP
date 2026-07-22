"""splitQP demo: one Cholesky factorization prices a whole efficient frontier.

    uv run python portfolio.py --points 64

A fixed covariance matrix and constraint matrix define a family of Markowitz
QPs in which only the target-return bound changes.  setup() factors H once;
solve_batch, 64 cold scalar solves, a warm-started sweep, and a detailed traced
solve all reuse that factor.  Every printed claim is asserted before the
2x2 portfolio.png is written.
"""

import argparse
import os

import numpy as np
import jax

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: write a PNG, open no window
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

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
m = n + 2

# Highest return reachable under the caps (greedy top holdings), and B targets.
r_hi = mu[::-1][:3] @ np.full(3, w_max) + mu[::-1][3] * (1 - 3 * w_max)
targets = np.linspace(mu.min(), 0.98 * r_hi, B)
q = np.zeros((B, n))
l = np.tile(np.concatenate([[1.0, 0.0], np.zeros(n)]), (B, 1))
u = np.tile(np.concatenate([[1.0, np.inf], np.full(n, w_max)]), (B, 1))
l[:, 1] = targets  # the only entry that varies across the family

# ------------------------------------------------- solve everything, one factor
splitqp.factorizations = 0
cache = splitqp.setup(Sigma, A)  # the single Cholesky factorization
MAX_IT = 20000

batch = splitqp.solve_batch(cache, q, l, u, max_iter=MAX_IT)
jax.block_until_ready(batch)  # finish async device work before host reads

cold = [splitqp.solve(cache, q[i], l[i], u[i], max_iter=MAX_IT) for i in range(B)]

warm = []
state = None
for i in range(B):  # ascending targets, each started from the previous solution
    r = splitqp.solve(cache, q[i], l[i], u[i], init=state, max_iter=MAX_IT)
    warm.append(r)
    state = splitqp.State(r.x, r.z, r.y)

hi = (3 * B) // 4  # the family member shown in the detailed panels
detail = splitqp.solve(cache, q[hi], l[hi], u[hi], max_iter=MAX_IT, trace=True)

# ------------------------------------------- assert every claim, then print it
x = np.asarray(batch.x)  # one explicit host conversion for all plotting data
iters_batch = np.asarray(batch.iterations)
iters_cold = np.array([int(r.iterations) for r in cold])
iters_warm = np.array([int(r.iterations) for r in warm])
tr = detail.trace

assert bool(np.all(np.asarray(batch.converged)))
assert all(bool(r.converged) for r in cold + warm + [detail])
assert splitqp.factorizations == 1, "every solve above must reuse one factor"

max_kkt = max(float(np.max(np.asarray(batch.r_primal))),
              float(np.max(np.asarray(batch.r_dual))))
assert max_kkt < 1e-5

batch_vs_scalar = max(float(np.max(np.abs(x[i] - np.asarray(cold[i].x))))
                      for i in range(B))
assert batch_vs_scalar < 1e-8
assert np.array_equal(iters_batch, iters_cold), "frozen lanes must match scalar stops"

assert iters_warm.sum() < iters_cold.sum(), "warm starts must shorten the sweep"
assert int(detail.iterations) == int(iters_batch[hi])
returns = x @ mu
risks = np.sqrt(np.einsum("bi,ij,bj->b", x, Sigma, x))
assert np.all(returns >= targets - 1e-5) and np.all(np.diff(risks) > -1e-9)

print(f"{B} QPs, {splitqp.factorizations} factorization")
print(f"max KKT residual: {max_kkt:.2e}")
print(f"batch/scalar difference: {batch_vs_scalar:.2e}")
print(f"cold iterations vs warm sequence: {iters_cold.sum()} vs {iters_warm.sum()}")

# ------------------------------------------------------------------ the figure
BLUE, ORANGE, RED, YELLOW = "#2a78d6", "#eb6834", "#e34948", "#eda100"
fig, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(2, 2, figsize=(12.5, 9), dpi=120)
fig.suptitle("splitQP: one Cholesky factorization, one QP family", fontsize=14)

# A: the family has a visible result -- the efficient frontier.
ax_a.plot(100 * risks, 100 * returns, color=BLUE, lw=2, marker=".", ms=5)
ax_a.plot(100 * risks[hi], 100 * returns[hi], "o", ms=11, mfc="none", mec=RED, mew=2)
ax_a.annotate(f"member #{hi}", (100 * risks[hi], 100 * returns[hi]),
              textcoords="offset points", xytext=(10, -12), color=RED)
ax_a.set_xlabel("risk  (std dev, %)")
ax_a.set_ylabel("expected return  (%)")
ax_a.set_title(f"A. efficient frontier: {B} QPs | 1 Cholesky | "
               f"max KKT {max_kkt:.1e} | batch vs scalar {batch_vs_scalar:.1e}",
               fontsize=10)

# B: matrix work is reused by both curves; state is reused only by the warm one.
ax_b.plot(100 * targets, iters_cold, color=BLUE, lw=2, label="cold start")
ax_b.plot(100 * targets, iters_warm, color=ORANGE, lw=2, label="warm sequence")
ax_b.axvline(100 * targets[hi], color=RED, lw=1, ls=":")
ax_b.set_xlabel("target return  (%)")
ax_b.set_ylabel("ADMM iterations")
ax_b.set_title(f"B. iterations: cold {iters_cold.sum()} vs warm {iters_warm.sum()} "
               "(same factor)", fontsize=10)
ax_b.legend(frameon=False)

# C: stopping is visible -- residuals cross their moving tolerances.
its = tr.iteration
ax_c.semilogy(its, np.max(np.abs(tr.r_primal), axis=1), color=BLUE, lw=2,
              label="|r_primal|")
ax_c.semilogy(its, tr.eps_primal, color=BLUE, lw=1.2, ls="--", label="eps_primal")
ax_c.semilogy(its, np.max(np.abs(tr.r_dual), axis=1), color=ORANGE, lw=2,
              label="|r_dual|")
ax_c.semilogy(its, tr.eps_dual, color=ORANGE, lw=1.2, ls="--", label="eps_dual")
ax_c.axvline(int(detail.iterations), color="0.4", lw=1, ls=":")
ax_c.annotate(f"accepted at {int(detail.iterations)}",
              (int(detail.iterations), float(tr.eps_dual[-1])),
              textcoords="offset points", xytext=(-8, 10), ha="right", color="0.25")
ax_c.set_xlabel(f"iteration (member #{hi})")
ax_c.set_ylabel("infinity norm")
ax_c.set_title("C. residuals vs absolute-plus-relative tolerances", fontsize=10)
ax_c.legend(frameon=False, ncols=2, fontsize=9)

# D: clip is an observable gate -- where the pre-clip value v lands each step.
codes = np.ones(tr.v.shape, dtype=int)          # 1 = free
codes[tr.v < l[hi][None, :]] = 0                # lower-clipped
codes[tr.v > u[hi][None, :]] = 2                # upper-clipped
codes[:, l[hi] == u[hi]] = 3                    # equality row
gate_cmap = ListedColormap([BLUE, "#f0efec", RED, YELLOW])
ax_d.imshow(codes.T, aspect="auto", interpolation="nearest", origin="lower",
            cmap=gate_cmap, vmin=-0.5, vmax=3.5,
            extent=(0.5, codes.shape[0] + 0.5, -0.5, m - 0.5))
ax_d.set_yticks([0, 1, 2, m - 1], ["budget", "return", "cap 1", f"cap {n}"])
ax_d.set_xlabel(f"iteration (member #{hi})")
ax_d.set_title("D. projection gate: v = z_bar + y/rho against [l, u]", fontsize=10)
ax_d.legend(handles=[Patch(fc=BLUE, label="lower-clipped"),
                     Patch(fc="#f0efec", ec="0.7", label="free"),
                     Patch(fc=RED, label="upper-clipped"),
                     Patch(fc=YELLOW, label="equality")],
            frameon=False, fontsize=8, loc="center right")

for ax in (ax_a, ax_b, ax_c):
    ax.grid(alpha=0.25, lw=0.5)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig("portfolio.png")
assert os.path.getsize("portfolio.png") > 0
print("wrote portfolio.png")
