"""Eager scalar ADMM reference for splitQP.

Solves  minimize 1/2 x'Px + q'x  subject to  l <= Ax <= u  with the six
OSQP-style updates written out one equation per line, re-solving the linear
system from scratch every iteration.  This file is the equation-for-equation
oracle: no jit, no vmap, no cached factorization.  splitqp.py runs the same
mathematics with one cached Cholesky; test.py holds the two paths together.
"""

import jax

jax.config.update("jax_enable_x64", True)  # float64 for every array; set before any array exists

import jax.numpy as jnp


def _inf(w):
    return jnp.max(jnp.abs(w))


def solve_reference(P, A, q, l, u, *, rho=0.1, sigma=1e-6, alpha=1.6,
                    eps_abs=1e-6, eps_rel=1e-6, max_iter=4000, init=None):
    """Run the scalar ADMM iteration eagerly and keep every intermediate.

    Returns (x, z, y, trace) where trace is a list of one dict per iteration
    holding the named intermediates, residuals, and stopping data, so the
    factorized path can be compared quantity by quantity.
    """
    n, m = q.shape[0], l.shape[0]
    # Same H as the cached path, but solved fresh below: slow and transparent.
    H = P + sigma * jnp.eye(n) + rho * A.T @ A
    x, z, y = (jnp.zeros(n), jnp.zeros(m), jnp.zeros(m)) if init is None else init
    trace = []
    for k in range(1, max_iter + 1):
        rhs = sigma * x - q + A.T @ (rho * z - y)
        x_tilde = jnp.linalg.solve(H, rhs)
        z_tilde = A @ x_tilde
        x = alpha * x_tilde + (1 - alpha) * x
        z_bar = alpha * z_tilde + (1 - alpha) * z
        v = z_bar + y / rho
        z = jnp.clip(v, l, u)
        y = y + rho * (z_bar - z)
        # OSQP residuals: r_primal is constraint violation, r_dual is the
        # gradient of the Lagrangian; both must pass an absolute-plus-relative
        # infinity-norm tolerance for the iterate to count as solved.
        Ax, Px, Aty = A @ x, P @ x, A.T @ y
        r_primal = Ax - z
        r_dual = Px + q + Aty
        eps_primal = eps_abs + eps_rel * jnp.maximum(_inf(Ax), _inf(z))
        eps_dual = eps_abs + eps_rel * jnp.maximum(_inf(Px),
                                                   jnp.maximum(_inf(Aty), _inf(q)))
        converged = bool((_inf(r_primal) <= eps_primal) & (_inf(r_dual) <= eps_dual))
        trace.append(dict(iteration=k, rhs=rhs, x_tilde=x_tilde, z_tilde=z_tilde,
                          x=x, z_bar=z_bar, v=v, z=z, y=y,
                          objective=0.5 * x @ (P @ x) + q @ x,
                          r_primal=r_primal, r_dual=r_dual,
                          eps_primal=eps_primal, eps_dual=eps_dual,
                          converged=converged))
        if converged:
            break
    return x, z, y, trace
