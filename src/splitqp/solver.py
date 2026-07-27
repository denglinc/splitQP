"""A tiny JAX implementation of proximal ADMM for box-form QPs."""

from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)  # float64 for every array; set before any array exists

import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_factor, cho_solve


class Cache(NamedTuple):  # Fixed data shared by every vmap lane.
    P: jax.Array
    A: jax.Array
    factor: jax.Array  # lower-triangular Cholesky factor of H
    rho: jax.Array     # (m,) penalty diagonal; a scalar rho is stored as rho * 1
    sigma: jax.Array
    alpha: jax.Array


class State(NamedTuple):  # Warm-startable (x, z, y) pytree.
    x: jax.Array
    z: jax.Array
    y: jax.Array


class Report(NamedTuple):  # Residuals and stopping data for one state.
    r_primal: jax.Array
    r_dual: jax.Array
    eps_primal: jax.Array
    eps_dual: jax.Array
    objective: jax.Array
    converged: jax.Array


class Result(NamedTuple):  # Final state and per-QP diagnostics.
    x: jax.Array
    z: jax.Array
    y: jax.Array
    iterations: jax.Array
    converged: jax.Array
    objective: jax.Array
    r_primal: jax.Array  # infinity norm of Ax - z, original coordinates
    r_dual: jax.Array    # infinity norm of Px + q + A'y, original coordinates
    status: object       # solved | max_iter | numerical_error | invalid_problem
    #                      (str for one solve, ndarray of str per batch lane)

    @property
    def state(self):  # Repackage the result as a warm-start state.
        return State(self.x, self.z, self.y)


def _inf(w):  # Batched infinity norm over the final axis.
    # initial=0.0 gives the reduction a valid identity, so an empty constraint
    # block (m == 0) yields 0.0 instead of an invalid max over an empty axis.
    return jnp.max(jnp.abs(w), axis=-1, initial=0.0)


def step(cache, q, l, u, state):  # Pure scalar update transformed by jit and vmap.
    # rho is one (m,) vector, so every rho product below is elementwise and the
    # projection stays the coordinate-wise clip.  A full SPD penalty would couple
    # the coordinates and is deliberately out of scope.
    x, z, y = state
    rhs = cache.sigma * x - q + cache.A.T @ (cache.rho * z - y)
    x_tilde = cho_solve((cache.factor, True), rhs)  # reuse the factor; True is the literal static lower flag
    z_tilde = cache.A @ x_tilde
    x_new = cache.alpha * x_tilde + (1 - cache.alpha) * x
    z_bar = cache.alpha * z_tilde + (1 - cache.alpha) * z
    v = z_bar + y / cache.rho
    z_new = jnp.clip(v, l, u)
    y_new = y + cache.rho * (z_bar - z_new)
    return State(x_new, z_new, y_new)


def report(cache, q, l, u, state, eps_abs, eps_rel):  # Residuals and stop test.
    x, z, y = state
    Ax, Px, Aty = cache.A @ x, cache.P @ x, cache.A.T @ y
    r_primal = Ax - z
    r_dual = Px + q + Aty
    eps_primal = eps_abs + eps_rel * jnp.maximum(_inf(Ax), _inf(z))
    eps_dual = eps_abs + eps_rel * jnp.maximum(_inf(Px), jnp.maximum(_inf(Aty), _inf(q)))
    converged = (_inf(r_primal) <= eps_primal) & (_inf(r_dual) <= eps_dual)
    objective = 0.5 * x @ (cache.P @ x) + q @ x
    return Report(r_primal, r_dual, eps_primal, eps_dual, objective, converged)


# One scalar equation set, batched only by vmap: in_axes=None shares P, A, and
# the factor across lanes; 0 maps each QP's data and state on the leading axis.
_batch_step = jax.vmap(step, in_axes=(None, 0, 0, 0, 0))
_batch_report = jax.vmap(report, in_axes=(None, 0, 0, 0, 0, None, None))


def _solve_loop(cache, q, l, u, state, eps_abs, eps_rel, max_iter):  # Fixed-shape device loop.
    iters = jnp.zeros(q.shape[0], jnp.int64)
    done = jnp.zeros(q.shape[0], bool)

    def cond(carry):
        _, _, done, k = carry
        return (~jnp.all(done)) & (k < max_iter)

    def body(carry):
        state, iters, done, k = carry
        new_state = _batch_step(cache, q, l, u, state)
        keep = done[:, None]  # freeze lanes at their first accepted stop while others continue
        state = State(jnp.where(keep, state.x, new_state.x),
                      jnp.where(keep, state.z, new_state.z),
                      jnp.where(keep, state.y, new_state.y))
        iters = iters + jnp.where(done, 0, 1)
        done = done | _batch_report(cache, q, l, u, state, eps_abs, eps_rel).converged
        return state, iters, done, k + 1

    # Every carried array keeps one shape and dtype from first to last trip;
    # only the values iterate.  eps/max_iter arrive as traced values, so
    # changing tolerances never recompiles.
    state, iters, done, _ = jax.lax.while_loop(cond, body, (state, iters, done, jnp.int64(0)))
    return state, iters, done


_jit_solve_loop = jax.jit(_solve_loop)


def _scalar_solve_loop(cache, q, l, u, state, eps_abs, eps_rel, max_iter):  # One member's ADMM loop.
    # Same stop test and iteration accounting as one lane of _solve_loop, without vmap.
    def cond(carry):
        _, iters, done = carry
        return (~done) & (iters < max_iter)

    def body(carry):
        state, iters, _ = carry
        new_state = step(cache, q, l, u, state)
        conv = report(cache, q, l, u, new_state, eps_abs, eps_rel).converged
        return new_state, iters + 1, conv

    state, iters, done = jax.lax.while_loop(
        cond, body, (state, jnp.int64(0), jnp.asarray(False)))
    return state, iters, done


def _sequence_loop(cache, qs, ls, us, init_state, eps_abs, eps_rel, max_iter):  # Ordered warm family.
    def body(carry_state, member):
        q, l, u = member
        final, iters, done = _scalar_solve_loop(cache, q, l, u, carry_state,
                                                eps_abs, eps_rel, max_iter)
        rep = report(cache, q, l, u, final, eps_abs, eps_rel)
        finite = (jnp.all(jnp.isfinite(final.x)) & jnp.all(jnp.isfinite(final.z))
                  & jnp.all(jnp.isfinite(final.y)))
        # A non-finite member must not poison the next member's warm start.
        safe = State(jnp.where(finite, final.x, jnp.zeros_like(final.x)),
                     jnp.where(finite, final.z, jnp.zeros_like(final.z)),
                     jnp.where(finite, final.y, jnp.zeros_like(final.y)))
        out = (final.x, final.z, final.y, iters, done, rep.objective,
               _inf(rep.r_primal), _inf(rep.r_dual))
        return safe, out

    # scan threads the accepted state from one member to the next and stacks each
    # member's own outputs along the leading axis; P, A, and the factor stay fixed.
    _, outs = jax.lax.scan(body, init_state, (qs, ls, us))
    return outs


_jit_sequence_loop = jax.jit(_sequence_loop)


def _validate_family(P, A, rho, sigma, alpha):  # Fixed-data check at construction.
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        return "P must be a square matrix"
    n = P.shape[0]
    if A.ndim != 2 or A.shape[1] != n:
        return "A must be (m, n) sharing P's column count"
    if not (bool(jnp.all(jnp.isfinite(P))) and bool(jnp.all(jnp.isfinite(A)))):
        return "P and A must be finite"
    # Scale-aware symmetry: |P - P.T| <= atol + rtol*|P.T| tolerates float64
    # roundoff asymmetry while rejecting a materially non-symmetric matrix.
    if not bool(jnp.allclose(P, P.T, rtol=1e-8, atol=1e-12)):
        return "P must be symmetric"
    # Convexity precondition with a scale-aware floor: accept roundoff-level
    # negative eigenvalues consistent with a mathematically PSD matrix, reject a
    # materially indefinite one. The O(n^3) eigvalsh is part of construction and
    # is included in the benchmark's setup timing.
    scale = 1.0 + float(jnp.max(jnp.abs(P)))
    if float(jnp.min(jnp.linalg.eigvalsh(P))) < -1e-8 * scale:
        return "P must be positive semidefinite"
    # rho is a scalar or one entry per constraint row; the shape is checked before
    # the values so a mis-shaped penalty is named as such rather than as rho<=0.
    if rho.ndim > 1 or (rho.ndim == 1 and rho.shape != (A.shape[0],)):
        return "rho must be a scalar or a vector of shape (m,)"
    if not (bool(jnp.all(rho > 0)) and sigma > 0 and 0 < alpha < 2):
        return "require rho>0, sigma>0, 0<alpha<2"
    return None


def _check_family_data(q, l, u, n, m, max_iter):  # Per-solve (q, l, u) check.
    if q.ndim != 2 or q.shape[1] != n:
        return "q must be (B, n)"
    B = q.shape[0]
    if l.shape != (B, m) or u.shape != (B, m):
        return "l and u must be (B, m)"
    if not bool(jnp.all(jnp.isfinite(q))):
        return "q must be finite"
    if bool(jnp.any(jnp.isnan(l))) or bool(jnp.any(jnp.isnan(u))):
        return "l and u must not be NaN (infinities are allowed)"
    if not bool(jnp.all(l <= u)):
        return "require l <= u elementwise"
    if max_iter < 1:
        return "max_iter must be >= 1"
    return None


def _statuses(x, z, y, r_primal, r_dual, converged):  # Host per-lane status label.
    # A non-finite iterate or residual is a numerical failure, never a solve;
    # axis=-1 collapses each lane and also handles the single (unbatched) case.
    finite = (np.all(np.isfinite(np.asarray(x)), axis=-1)
              & np.all(np.isfinite(np.asarray(z)), axis=-1)
              & np.all(np.isfinite(np.asarray(y)), axis=-1)
              & np.isfinite(np.asarray(r_primal))
              & np.isfinite(np.asarray(r_dual)))
    conv = np.asarray(converged, dtype=bool)
    return np.where(~finite, "numerical_error",
                    np.where(conv, "solved", "max_iter"))


class Solver:  # Host API owning one factorized QP family.

    def __init__(self, P, A, *, rho=0.1, sigma=1e-6, alpha=1.6):  # Factor P, A once.
        # rho is a positive scalar or a positive (m,) diagonal penalty.
        P = jnp.asarray(P, jnp.float64)
        A = jnp.asarray(A, jnp.float64)
        rho = jnp.asarray(rho, jnp.float64)
        # Column count and constraint count, recorded even when data is invalid
        # so an explicit failure result still has the right shapes.
        self.n = int(P.shape[0]) if P.ndim == 2 else 0
        self.m = int(A.shape[0]) if A.ndim == 2 else 0
        # Invalid fixed data is surfaced as a returned status, never an exception.
        self._invalid_reason = _validate_family(P, A, rho, sigma, alpha)
        factor = None
        if self._invalid_reason is None:
            n = P.shape[0]
            # One (m,) penalty from here on: a scalar becomes rho * 1, so the jitted
            # iteration never branches on scalar versus diagonal.
            rho = jnp.broadcast_to(rho, (self.m,))
            H = P + sigma * jnp.eye(n) + A.T @ (rho[:, None] * A)
            # cho_factor returns (array, lower_flag); keep only the array. Carrying
            # the Python bool through jit/vmap would turn a static flag into a tracer.
            factor, _ = cho_factor(H, lower=True)
            if not bool(jnp.all(jnp.isfinite(factor))):
                self._invalid_reason = "H = P + sigma I + A^T diag(rho) A is not positive definite"
        if self._invalid_reason is not None:
            self._cache = None
            self._factorizations = 0
            return
        self._cache = Cache(P, A, factor, rho, jnp.float64(sigma),
                            jnp.float64(alpha))
        self._factorizations = 1  # the cho_factor call above is the only one in the module

    @property
    def cache(self):  # Expose the immutable shared setup data (None if invalid).
        return self._cache

    @property
    def factorizations(self):  # One after successful construction, else zero.
        return self._factorizations

    def _invalid_batch(self, qs):  # Explicit failure carrying no fake solution.
        q = np.asarray(qs)
        B = q.shape[0] if q.ndim >= 1 else 1
        n, m = self.n, self.m
        nan = jnp.nan
        return Result(jnp.full((B, n), nan), jnp.full((B, m), nan),
                      jnp.full((B, m), nan), jnp.zeros(B, jnp.int64),
                      jnp.zeros(B, bool), jnp.full(B, nan), jnp.full(B, nan),
                      jnp.full(B, nan), np.array(["invalid_problem"] * B))

    def solve(self, q, l, u, *, init=None, eps_abs=1e-6, eps_rel=1e-6,
              max_iter=4000):  # Wrap one QP as one batch lane.
        if self._invalid_reason is not None:
            n, m = self.n, self.m
            return Result(jnp.full(n, jnp.nan), jnp.full(m, jnp.nan),
                          jnp.full(m, jnp.nan), jnp.int64(0), jnp.asarray(False),
                          jnp.nan, jnp.nan, jnp.nan, "invalid_problem")
        m, n = self._cache.A.shape
        q, l, u = (jnp.asarray(w, jnp.float64) for w in (q, l, u))
        if init is None:
            init = State(jnp.zeros(n), jnp.zeros(m), jnp.zeros(m))
        # One QP is a one-lane batch of the same compiled loop: the lane axis
        # is added here and stripped from the returned leaves below.
        r = self.solve_batch(q[None], l[None], u[None],
                             init=State(init.x[None], init.z[None], init.y[None]),
                             eps_abs=eps_abs, eps_rel=eps_rel, max_iter=max_iter)
        return Result(r.x[0], r.z[0], r.y[0], r.iterations[0], r.converged[0],
                      r.objective[0], r.r_primal[0], r.r_dual[0], str(r.status[0]))

    def solve_batch(self, qs, ls, us, *, init=None, eps_abs=1e-6, eps_rel=1e-6,
                    max_iter=4000):  # Map the scalar step over a QP family.
        if self._invalid_reason is not None:
            return self._invalid_batch(qs)
        cache = self._cache
        m, n = cache.A.shape
        q, l, u = (jnp.asarray(w, jnp.float64) for w in (qs, ls, us))
        reason = _check_family_data(q, l, u, n, m, max_iter)
        if reason is not None:
            return self._invalid_batch(qs)
        B = q.shape[0]
        if init is None:
            init = State(jnp.zeros((B, n)), jnp.zeros((B, m)), jnp.zeros((B, m)))
        state, iters, done = _jit_solve_loop(cache, q, l, u, init, eps_abs,
                                             eps_rel, max_iter)
        rep = _batch_report(cache, q, l, u, state, eps_abs, eps_rel)
        r_primal, r_dual = _inf(rep.r_primal), _inf(rep.r_dual)
        status = _statuses(state.x, state.z, state.y, r_primal, r_dual, done)
        return Result(state.x, state.z, state.y, iters, done, rep.objective,
                      r_primal, r_dual, status)

    def solve_sequence(self, qs, ls, us, *, init=None, eps_abs=1e-6, eps_rel=1e-6,
                       max_iter=4000):  # One compiled program for an ordered warm family.
        # Each member is warm-started from the previous member's accepted state and
        # the whole ordered family runs in a single jax.lax.scan over the shared
        # factor -- no per-member Python dispatch or host synchronization. Returns
        # the same per-member Result shape as solve_batch.
        if self._invalid_reason is not None:
            return self._invalid_batch(qs)
        cache = self._cache
        m, n = cache.A.shape
        qs, ls, us = (jnp.asarray(w, jnp.float64) for w in (qs, ls, us))
        reason = _check_family_data(qs, ls, us, n, m, max_iter)
        if reason is not None:
            return self._invalid_batch(qs)
        init = State(jnp.zeros(n), jnp.zeros(m), jnp.zeros(m)) if init is None else init
        xs, zs, ys, iters, done, obj, r_primal, r_dual = _jit_sequence_loop(
            cache, qs, ls, us, init, eps_abs, eps_rel, max_iter)
        status = _statuses(xs, zs, ys, r_primal, r_dual, done)
        return Result(xs, zs, ys, iters, done, obj, r_primal, r_dual, status)
