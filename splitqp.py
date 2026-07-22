"""splitQP: one cached Cholesky, many box QPs.

    minimize    1/2 x'Px + q'x
    subject to  l <= Ax <= u

Solver(P, A) factors H = P + sigma*I + rho*A'A exactly once, at construction.
After that, every ADMM iteration is one triangular solve, a box projection,
and a dual update -- for one QP or a whole batch that shares P and A and
varies only q, l, u.  The mathematical core is the module-level pure
functions below; Solver is a thin host-side facade that owns the factored
family and never enters jit.
"""

from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)  # float64 for every array; set before any array exists

import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_factor, cho_solve


class Cache(NamedTuple):
    """Setup product shared by the whole QP family; vmap never maps it."""
    P: jax.Array
    A: jax.Array
    factor: jax.Array  # lower-triangular Cholesky factor of H
    rho: jax.Array
    sigma: jax.Array
    alpha: jax.Array


class State(NamedTuple):
    """The warm-startable iterate. JAX arrays are immutable, so a step returns
    a fresh State instead of mutating this one."""
    x: jax.Array
    z: jax.Array
    y: jax.Array


class Parts(NamedTuple):
    """Named intermediates of one step: the causal boundary between the reused
    linear solve and the piecewise-linear projection."""
    rhs: jax.Array
    x_tilde: jax.Array
    z_tilde: jax.Array
    z_bar: jax.Array
    v: jax.Array  # pre-clip value; which side of [l, u] it lands on is the active set


class Report(NamedTuple):
    """Residuals, tolerances, objective, and the stop decision for one state."""
    r_primal: jax.Array
    r_dual: jax.Array
    eps_primal: jax.Array
    eps_dual: jax.Array
    objective: jax.Array
    converged: jax.Array


class Trace(NamedTuple):
    """Per-iteration snapshots (leading axis = iteration) from Solver.trace."""
    iteration: np.ndarray
    rhs: np.ndarray
    x_tilde: np.ndarray
    z_tilde: np.ndarray
    x: np.ndarray
    z_bar: np.ndarray
    v: np.ndarray
    z: np.ndarray
    y: np.ndarray
    objective: np.ndarray
    r_primal: np.ndarray
    r_dual: np.ndarray
    eps_primal: np.ndarray
    eps_dual: np.ndarray
    converged: np.ndarray


class Result(NamedTuple):
    """Final iterate plus per-QP diagnostics. converged=False means max_iter."""
    x: jax.Array
    z: jax.Array
    y: jax.Array
    iterations: jax.Array
    converged: jax.Array
    objective: jax.Array
    r_primal: jax.Array  # infinity norm of Ax - z
    r_dual: jax.Array    # infinity norm of Px + q + A'y

    @property
    def state(self):
        """The warm-startable (x, z, y); pass as init= to a nearby next solve."""
        return State(self.x, self.z, self.y)


def _inf(w):
    return jnp.max(jnp.abs(w), axis=-1)


def step(cache, q, l, u, state):
    """One scalar ADMM update.  Pure: new state out, nothing mutated, so jit
    and vmap can trace it.  The compiled loop discards Parts (dead code)."""
    x, z, y = state
    rhs = cache.sigma * x - q + cache.A.T @ (cache.rho * z - y)
    x_tilde = cho_solve((cache.factor, True), rhs)  # reuse the factor; True is the literal static lower flag
    z_tilde = cache.A @ x_tilde
    x_new = cache.alpha * x_tilde + (1 - cache.alpha) * x
    z_bar = cache.alpha * z_tilde + (1 - cache.alpha) * z
    v = z_bar + y / cache.rho
    z_new = jnp.clip(v, l, u)
    y_new = y + cache.rho * (z_bar - z_new)
    return State(x_new, z_new, y_new), Parts(rhs, x_tilde, z_tilde, z_bar, v)


def report(cache, q, l, u, state, eps_abs, eps_rel):
    """OSQP residuals and absolute-plus-relative infinity-norm stop test."""
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
_jit_step = jax.jit(step)
_jit_report = jax.jit(report)


def _solve_loop(cache, q, l, u, state, eps_abs, eps_rel, max_iter):
    """Batched fixed-shape iteration, entirely on device.

    Convergence is only checked after a step: from the first iteration onward
    z is a box projection, so z is in [l, u] by construction and the residuals
    carry their optimality meaning.  A raw initial state has no such guarantee.
    """
    iters = jnp.zeros(q.shape[0], jnp.int64)
    done = jnp.zeros(q.shape[0], bool)

    def cond(carry):
        _, _, done, k = carry
        return (~jnp.all(done)) & (k < max_iter)

    def body(carry):
        state, iters, done, k = carry
        new_state, _ = _batch_step(cache, q, l, u, state)
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


def _trace_solve(cache, q, l, u, state, eps_abs, eps_rel, max_iter):
    """Same jitted step and report, driven from Python; snapshots are appended
    only after each compiled call has returned to the host."""
    converged, k, snaps = False, 0, []
    while not converged and k < max_iter:
        state, parts = _jit_step(cache, q, l, u, state)
        rep = _jit_report(cache, q, l, u, state, eps_abs, eps_rel)
        k += 1
        converged = bool(rep.converged)  # device-to-host sync: wait for and read one traced bool
        snaps.append(Trace(k, parts.rhs, parts.x_tilde, parts.z_tilde, state.x,
                           parts.z_bar, parts.v, state.z, state.y, rep.objective,
                           rep.r_primal, rep.r_dual, rep.eps_primal, rep.eps_dual,
                           rep.converged))
    trace = Trace(*(np.stack([np.asarray(getattr(s, f)) for s in snaps])
                    for f in Trace._fields))
    result = Result(state.x, state.z, state.y, jnp.int64(k), rep.converged,
                    rep.objective, _inf(rep.r_primal), _inf(rep.r_dual))
    return result, trace


class Solver:
    """Host-side owner of one fixed QP family: P and A set once, factored once.

    Construction converts and validates P and A, forms H, and calls Cholesky
    exactly once.  Every solve, batch, trace, and warm start reuses the
    resulting immutable Cache; no method mutates the solver or enters jit.
    """

    def __init__(self, P, A, *, rho=0.1, sigma=1e-6, alpha=1.6):
        P = jnp.asarray(P, jnp.float64)
        A = jnp.asarray(A, jnp.float64)
        n = P.shape[0]
        assert P.shape == (n, n) and A.ndim == 2 and A.shape[1] == n
        assert bool(jnp.allclose(P, P.T)), "P must be symmetric"
        assert rho > 0 and sigma > 0 and 0 < alpha < 2
        H = P + sigma * jnp.eye(n) + rho * A.T @ A
        # cho_factor returns (array, lower_flag); keep only the array.  Carrying
        # the Python bool through jit/vmap would turn a static flag into a tracer.
        factor, _ = cho_factor(H, lower=True)
        self._cache = Cache(P, A, factor, jnp.float64(rho), jnp.float64(sigma),
                            jnp.float64(alpha))
        self._factorizations = 1  # the cho_factor call above is the only one in the module

    @property
    def cache(self):
        """The immutable setup product that every method below reuses."""
        return self._cache

    @property
    def factorizations(self):
        """Construction is the single setup event; this never increments."""
        return self._factorizations

    def solve(self, q, l, u, *, init=None, eps_abs=1e-6, eps_rel=1e-6,
              max_iter=4000):
        """Solve one QP; init=State(x, z, y) warm-starts from a prior result."""
        m, n = self._cache.A.shape
        q, l, u = (jnp.asarray(w, jnp.float64) for w in (q, l, u))
        assert q.shape == (n,) and l.shape == (m,) and u.shape == (m,)
        if init is None:
            init = State(jnp.zeros(n), jnp.zeros(m), jnp.zeros(m))
        # One QP is a one-lane batch of the same compiled loop: the lane axis
        # is added here and stripped from the returned leaves below.
        r = self.solve_batch(q[None], l[None], u[None],
                             init=State(init.x[None], init.z[None], init.y[None]),
                             eps_abs=eps_abs, eps_rel=eps_rel, max_iter=max_iter)
        return Result(r.x[0], r.z[0], r.y[0], r.iterations[0], r.converged[0],
                      r.objective[0], r.r_primal[0], r.r_dual[0])

    def solve_batch(self, qs, ls, us, *, init=None, eps_abs=1e-6, eps_rel=1e-6,
                    max_iter=4000):
        """Solve B QPs sharing the cache; qs is (B, n) and ls, us are (B, m)."""
        cache = self._cache
        m, n = cache.A.shape
        q, l, u = (jnp.asarray(w, jnp.float64) for w in (qs, ls, us))
        B = q.shape[0]
        assert q.shape == (B, n) and l.shape == (B, m) and u.shape == (B, m)
        assert bool(jnp.all(l <= u)) and max_iter >= 1
        if init is None:
            init = State(jnp.zeros((B, n)), jnp.zeros((B, m)), jnp.zeros((B, m)))
        state, iters, done = _jit_solve_loop(cache, q, l, u, init, eps_abs,
                                             eps_rel, max_iter)
        rep = _batch_report(cache, q, l, u, state, eps_abs, eps_rel)
        return Result(state.x, state.z, state.y, iters, done, rep.objective,
                      _inf(rep.r_primal), _inf(rep.r_dual))

    def trace(self, q, l, u, *, init=None, eps_abs=1e-6, eps_rel=1e-6,
              max_iter=4000):
        """Solve one QP from a host loop around the same jitted step, recording
        every named intermediate.  Returns (Result, Trace) separately."""
        cache = self._cache
        m, n = cache.A.shape
        q, l, u = (jnp.asarray(w, jnp.float64) for w in (q, l, u))
        assert q.shape == (n,) and l.shape == (m,) and u.shape == (m,)
        assert bool(jnp.all(l <= u)) and max_iter >= 1
        if init is None:
            init = State(jnp.zeros(n), jnp.zeros(m), jnp.zeros(m))
        return _trace_solve(cache, q, l, u, init, eps_abs, eps_rel, max_iter)
