"""A tiny JAX implementation of proximal ADMM for box-form QPs."""

from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)  # float64 for every array; set before any array exists

import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_factor, cho_solve


class Cache(NamedTuple):
    """Fixed matrices and their shared Cholesky factor."""
    P: jax.Array
    A: jax.Array
    factor: jax.Array  # lower-triangular Cholesky factor of H
    rho: jax.Array
    sigma: jax.Array
    alpha: jax.Array


class State(NamedTuple):
    """Warm-startable ADMM iterate."""
    x: jax.Array
    z: jax.Array
    y: jax.Array


class Parts(NamedTuple):
    """Named intermediates from one ADMM step."""
    rhs: jax.Array
    x_tilde: jax.Array
    z_tilde: jax.Array
    z_bar: jax.Array
    v: jax.Array  # pre-clip value; which side of [l, u] it lands on is the active set


class Report(NamedTuple):
    """Residuals and stopping data for one state."""
    r_primal: jax.Array
    r_dual: jax.Array
    eps_primal: jax.Array
    eps_dual: jax.Array
    objective: jax.Array
    converged: jax.Array


class Trace(NamedTuple):
    """Iteration records stacked along the leading axis."""
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
    """Solution and stopping diagnostics."""
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
        """Return the warm-startable iterate."""
        return State(self.x, self.z, self.y)


def _inf(w):
    return jnp.max(jnp.abs(w), axis=-1)


def step(cache, q, l, u, state):
    """One pure scalar ADMM iteration."""
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
    """Compute OSQP residuals and stopping thresholds."""
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
    """Run a batched ADMM loop on device."""
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
    """Run the same jitted step from Python and record it."""
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
    """A fixed QP family sharing one factorization."""

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
        """Return the shared setup data."""
        return self._cache

    @property
    def factorizations(self):
        """Return the number of setup factorizations."""
        return self._factorizations

    def solve(self, q, l, u, *, init=None, eps_abs=1e-6, eps_rel=1e-6,
              max_iter=4000):
        """Solve one QP, optionally from a warm start."""
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
                      r.objective[0], r.r_primal[0], r.r_dual[0])

    def solve_batch(self, qs, ls, us, *, init=None, eps_abs=1e-6, eps_rel=1e-6,
                    max_iter=4000):
        """Solve a batch sharing P, A, and the factor."""
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
        """Solve one QP and retain every iteration."""
        cache = self._cache
        m, n = cache.A.shape
        q, l, u = (jnp.asarray(w, jnp.float64) for w in (q, l, u))
        assert q.shape == (n,) and l.shape == (m,) and u.shape == (m,)
        assert bool(jnp.all(l <= u)) and max_iter >= 1
        if init is None:
            init = State(jnp.zeros(n), jnp.zeros(m), jnp.zeros(m))
        return _trace_solve(cache, q, l, u, init, eps_abs, eps_rel, max_iter)
