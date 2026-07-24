"""splitQP local timing experiment (optional, prints only).

    uv run --group bench python bench.py          # portfolio + synthetic_medium
    uv run --group bench python bench.py --full    # + synthetic_small and _large

Times a fixed-(P, A) family solved several ways -- splitQP's Python warm sequence
(``solve`` per member), compiled ``solve_sequence`` (one JAX program), and
``solve_batch`` (independent members), all reusing one construction-time
factorization -- alongside persistent OSQP and dense ProxQP driven with their
public vector-only update APIs. It writes nothing and downloads nothing, and runs
a few private sanity checks first. OSQP/ProxQP are persistent objects with vector
updates (their adaptive rho/mu may refactorize), not one-factorization solvers;
splitQP's batch is a solver-specific mode. A local experiment, not a ranking.
"""

import argparse
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import splitqp

TOL = 1e-6
ACCEPT = 1e-5
MAX_IT = 20000
_PROX_INF = 1e20  # proxsuite treats bounds beyond +/-1e20 as infinite; it segfaults on inf


# --------------------------------------------------------------------------- #
# Metrics + deterministic families                                            #
# --------------------------------------------------------------------------- #
def metrics(P, A, q, l, u, x, dual):
    x = np.asarray(x, float)
    Ax = A @ x
    lo = np.where(np.isfinite(l), l - Ax, -np.inf)
    hi = np.where(np.isfinite(u), Ax - u, -np.inf)
    primal = max(np.max(lo, initial=0.0), np.max(hi, initial=0.0), 0.0)
    obj = 0.5 * x @ P @ x + q @ x
    stat = np.max(np.abs(P @ x + q + A.T @ np.asarray(dual)), initial=0.0) if dual is not None else np.inf
    return float(obj), float(primal), float(stat)


def accept(P, A, q, l, u, x, dual):
    if x is None or not np.all(np.isfinite(np.asarray(x))):
        return False
    obj, primal, stat = metrics(P, A, q, l, u, x, dual)
    return np.isfinite(obj) and primal < ACCEPT and stat < ACCEPT


def _spd(rng, n, cond=None):
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    d = rng.uniform(0.5, 2.0, size=n) if cond is None else np.exp(np.linspace(0.0, np.log(cond), n))
    return (Q * d) @ Q.T


def portfolio(B=32):
    n = 16
    mu = np.linspace(0.02, 0.12, n)
    vol = np.linspace(0.05, 0.30, n)
    idx = np.arange(n)
    Sigma = np.diag(vol) @ np.exp(-np.abs(idx[:, None] - idx[None, :]) / 8.0) @ np.diag(vol)
    A = np.vstack([np.ones(n), mu, np.eye(n)])
    r_hi = mu[::-1][:3] @ np.full(3, 0.30) + mu[::-1][3] * (1 - 3 * 0.30)
    targets = np.linspace(mu.min(), 0.98 * r_hi, B)
    q = np.zeros((B, n))
    l = np.tile(np.concatenate([[1.0, 0.0], np.zeros(n)]), (B, 1))
    u = np.tile(np.concatenate([[1.0, np.inf], np.full(n, 0.30)]), (B, 1))
    l[:, 1] = targets
    return dict(name="portfolio", P=Sigma, A=A, q=q, l=l, u=u)


def synthetic(name, n, m, B, seed, cond=None):
    rng = np.random.default_rng(seed)
    P = 0.5 * (lambda M: M + M.T)(_spd(rng, n, cond))
    A = rng.normal(size=(m, n))
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    x0, dx = rng.normal(size=n), rng.normal(size=n)
    width = rng.uniform(0.3, 0.8, size=m)
    q = np.zeros((B, n)); l = np.zeros((B, m)); u = np.zeros((B, m))
    for i, t in enumerate(np.linspace(0.0, 1.0, B)):
        xc = x0 + t * dx
        c = A @ xc
        l[i], u[i] = c - width, c + width
        q[i] = -(P @ xc)
    return dict(name=name, P=P, A=A, q=q, l=l, u=u)


def families(full):
    med = synthetic("synthetic_medium", 64, 96, 64, 11)
    if not full:
        return [portfolio(), med]
    return [portfolio(), synthetic("synthetic_small", 32, 48, 64, 10), med,
            synthetic("synthetic_large", 128, 192, 64, 12)]


# --------------------------------------------------------------------------- #
# Timing helpers                                                              #
# --------------------------------------------------------------------------- #
def _median_ms(times_ns):
    return float(np.median(times_ns)) / 1e6


def _time_seq(build, run_seq, reps):
    """build() -> fresh persistent object (untimed); run_seq(obj) -> runs the family."""
    run_seq(build())  # warm-up (compiles JAX / primes the solver)
    ts = []
    for _ in range(reps):
        obj = build()
        t0 = time.perf_counter_ns()
        run_seq(obj)
        ts.append(time.perf_counter_ns() - t0)
    return _median_ms(ts)


def _time_call(fn, reps):
    """Single compiled call (solve_batch / solve_sequence)."""
    jax.block_until_ready(fn().x)  # warm-up
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        r = fn()
        jax.block_until_ready([r.x, r.y, r.z])
        ts.append(time.perf_counter_ns() - t0)
    return _median_ms(ts)


# --------------------------------------------------------------------------- #
# Per-solver family runs                                                      #
# --------------------------------------------------------------------------- #
def split_warm(solver, fam):
    qs, ls, us = fam["q"], fam["l"], fam["u"]
    out, state = [], None
    for i in range(qs.shape[0]):
        r = solver.solve(qs[i], ls[i], us[i], init=state, eps_abs=TOL, eps_rel=TOL, max_iter=MAX_IT)
        jax.block_until_ready([r.x])
        out.append(r); state = r.state
    return out


def osqp_seq(fam, reps):
    try:
        import osqp
        import scipy.sparse as sp
    except ImportError:
        return None, 0
    P, A, qs, ls, us = fam["P"], fam["A"], fam["q"], fam["l"], fam["u"]
    B, n, m = qs.shape[0], P.shape[0], A.shape[0]

    def build():
        o = osqp.OSQP()
        o.setup(sp.csc_matrix(P), qs[0], sp.csc_matrix(A), ls[0], us[0],
                eps_abs=TOL, eps_rel=TOL, verbose=False, max_iter=20000)
        return o

    def run(o):
        res = []
        for i in range(B):
            o.update(q=qs[i], l=ls[i], u=us[i])   # ordered warm sequence, vector-only updates
            r = o.solve(); res.append(r)
        return res
    solved = sum(accept(P, A, qs[i], ls[i], us[i], np.asarray(r.x), np.asarray(r.y))
                 and r.info.status == "solved"
                 for i, r in enumerate(run(build())))
    return _time_seq(build, run, reps) / B, solved


def proxqp_seq(fam, reps):
    try:
        from proxsuite import proxqp
    except ImportError:
        return None, 0
    P, A, qs, ls, us = fam["P"], fam["A"], fam["q"], fam["l"], fam["u"]
    B = qs.shape[0]
    fin = lambda v: np.clip(np.asarray(v, float), -_PROX_INF, _PROX_INF)

    def build():
        qp = proxqp.dense.QP(P.shape[0], 0, A.shape[0])
        qp.settings.eps_abs = TOL; qp.settings.eps_rel = TOL; qp.settings.verbose = False
        qp.settings.initial_guess = proxqp.InitialGuess.NO_INITIAL_GUESS
        qp.init(np.asarray(P), qs[0], None, None, np.asarray(A), fin(ls[0]), fin(us[0]))
        return qp

    def run(qp):
        res = []
        for i in range(B):   # member 0 cold; warm-start from the previous result after that
            qp.settings.initial_guess = (proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
                                         if i else proxqp.InitialGuess.NO_INITIAL_GUESS)
            qp.update(g=qs[i], l=fin(ls[i]), u=fin(us[i]))
            qp.solve()
            res.append((np.asarray(qp.results.x), np.asarray(qp.results.z),
                        str(qp.results.info.status)))
        return res
    solved = sum(accept(P, A, qs[i], ls[i], us[i], x, z)
                 and st == str(proxqp.QPSolverOutput.PROXQP_SOLVED)
                 for i, (x, z, st) in enumerate(run(build())))
    return _time_seq(build, run, reps) / B, solved


def sanity(fam):
    """Private checks before timing; raise clearly on any failure."""
    P, A, qs, ls, us = fam["P"], fam["A"], fam["q"], fam["l"], fam["u"]
    B = qs.shape[0]
    solver = splitqp.Solver(P, A)
    P0, A0 = np.array(solver.cache.P), np.array(solver.cache.A)
    scalar = solver.solve(qs[0], ls[0], us[0], max_iter=MAX_IT)
    batch = solver.solve_batch(qs, ls, us, max_iter=MAX_IT)
    seq = solver.solve_sequence(qs, ls, us, max_iter=MAX_IT)
    warm = split_warm(solver, fam)
    assert solver.factorizations == 1, "one construction-time factorization expected"
    assert np.array_equal(np.array(solver.cache.P), P0) and np.array_equal(np.array(solver.cache.A), A0), \
        "q/l/u must not mutate cached P, A"
    assert np.all(np.isfinite(np.asarray(batch.x))) and np.all(np.isfinite(np.asarray(seq.x)))
    assert float(np.max(np.abs(np.asarray(scalar.x) - np.asarray(batch.x[0])))) < 1e-8, "scalar == batch"
    diff = max(float(np.max(np.abs(np.asarray(seq.x[i]) - np.asarray(warm[i].x))))
               for i in range(B) if str(np.asarray(seq.status)[i]) == "solved")
    assert diff < 1e-4, f"python warm vs compiled sequence differ ({diff:.1e})"
    for i in range(B):
        assert accept(P, A, qs[i], ls[i], us[i], np.asarray(batch.x[i]), np.asarray(batch.y[i])), \
            f"splitQP result inaccurate at member {i}"
    return solver, batch, seq, warm, diff


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="splitQP local timing experiment")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    reps = 5

    print(f"splitQP local timing  |  amortized ms/QP unless noted  |  CPU float64  |  "
          f"fixed rho, tol {TOL}, accept < {ACCEPT}")
    print(f"{'family':>17} {'n':>4} {'m':>4} {'B':>4} | {'setup ms':>8} {'py-warm':>8} "
          f"{'compiled':>8} {'batch':>8} | {'osqp':>8} {'proxqp':>8}")

    for fam in families(args.full):
        P, A, qs, ls, us = fam["P"], fam["A"], fam["q"], fam["l"], fam["u"]
        B, n, m = qs.shape[0], P.shape[0], A.shape[0]
        sanity(fam)  # private correctness checks; raises clearly on any failure

        splitqp.Solver(P, A)  # warm-up the construction path, then time it (includes the factor)
        setup_reps = []
        for _ in range(reps):
            t0 = time.perf_counter_ns(); splitqp.Solver(P, A); setup_reps.append(time.perf_counter_ns() - t0)
        setup_ms = _median_ms(setup_reps)
        pywarm_ms = _time_seq(lambda: splitqp.Solver(P, A), lambda s: split_warm(s, fam), reps) / B
        seq_ms = _time_call(lambda: splitqp.Solver(P, A).solve_sequence(qs, ls, us, max_iter=MAX_IT), reps) / B
        batch_ms = _time_call(lambda: splitqp.Solver(P, A).solve_batch(qs, ls, us, max_iter=MAX_IT), reps) / B
        osqp_ms, osqp_ok = osqp_seq(fam, reps)
        prox_ms, prox_ok = proxqp_seq(fam, reps)

        def cell(v):
            return "not inst." if v is None else f"{v:.3f}"
        print(f"{fam['name']:>17} {n:>4} {m:>4} {B:>4} | {setup_ms:>8.3f} {pywarm_ms:>8.3f} "
              f"{seq_ms:>8.3f} {batch_ms:>8.3f} | {cell(osqp_ms):>8} {cell(prox_ms):>8}")

    # one fixed-rho scope-boundary case: an ill-conditioned family (random q, so
    # constraints are active) that the default rho does not converge within max_iter.
    rng = np.random.default_rng(3)
    n, m, B = 16, 12, 16
    Pb = _spd(rng, n, cond=1e6); Pb = 0.5 * (Pb + Pb.T)
    Ab = rng.normal(size=(m, n))
    qb = rng.normal(size=(B, n))
    cb = (Ab @ rng.normal(size=(B, n)).T).T
    wb = rng.uniform(0.2, 1.0, size=(B, m))
    s = splitqp.Solver(Pb, Ab)
    rb = s.solve_batch(qb, cb - wb, cb + wb, max_iter=MAX_IT)
    jax.block_until_ready(rb.x)
    conv = int(np.asarray(rb.converged).sum())
    print(f"\nscope boundary (fixed rho): illcond_P n=16 m=12 B=16 -> converged "
          f"{conv}/16 at the default rho; ill-conditioned families need a problem-matched rho.")
    print("py-warm = solve() per member (ordered, warm); compiled = one lax.scan; batch = "
          "independent members; all reuse one factorization. osqp/proxqp are persistent "
          "objects with vector-only updates. Local experiment, not a solver ranking.")


if __name__ == "__main__":
    main()
