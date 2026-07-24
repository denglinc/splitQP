"""splitQP: a tiny JAX proximal-ADMM solver for box-QP families.

The public entry point is :class:`Solver`, which factors ``(P, A)`` once and then
serves ``solve``, ``solve_batch``, and ``solve_sequence``; each returns a
:class:`Result` carrying the solution(s), status, residuals, and counters.
"""

from .solver import Solver, Result

__all__ = ["Solver", "Result"]
