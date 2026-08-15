"""The numerical core: solve a system of nonlinear constraint equations, and
report the sketch's degrees of freedom.

A constrained sketch is a set of parameters q (the coordinates of every point,
the radius of every circle) and a set of residual equations r(q) = 0 (one per
scalar constraint). Solving the sketch means driving every residual to zero at
once. The equations are nonlinear — a distance constraint is quadratic, an angle
constraint trigonometric — so this is a root-find, done with Gauss–Newton and
Levenberg–Marquardt damping.

The second output matters as much as the first. The rank of the constraint
Jacobian at the solution tells you the sketch's *degrees of freedom*: how many
ways the geometry can still move without breaking a constraint (0 = fully
defined) and how many constraints are redundant. That is exactly the "fully
constrained / under-constrained / over-constrained" verdict a CAD sketcher shows
you, computed here from the linear algebra rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SolveResult:
    q: np.ndarray                # the solved parameter vector
    converged: bool
    iterations: int
    residual_norm: float         # max |residual| at the solution
    # degrees-of-freedom analysis (computed at the solution)
    n_params: int                # total free parameters (unpinned)
    n_equations: int             # number of scalar constraint equations
    rank: int                    # numerical rank of the Jacobian
    dof: int                     # remaining degrees of freedom = n_params - rank
    redundant: int               # dependent constraints = n_equations - rank

    @property
    def status(self) -> str:
        if self.dof > 0:
            return "under-constrained"
        if self.redundant > 0:
            return "over-constrained"
        return "fully-constrained"

    def describe(self) -> str:
        return (
            f"{self.status}: {self.n_params} free params, {self.n_equations} equations, "
            f"rank {self.rank} → {self.dof} DOF remaining, {self.redundant} redundant; "
            f"residual {self.residual_norm:.2e} in {self.iterations} iters"
        )


def solve(
    residual_fn,
    jacobian_fn,
    q0: np.ndarray,
    free_mask: np.ndarray,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> SolveResult:
    """Levenberg–Marquardt least-squares solve of residual_fn(q) = 0.

    `free_mask` is a boolean over q: False entries are pinned (anchored points)
    and never move. `tol` is on the max-norm of the residual. The damped normal
    equations (JᵀJ + λI) δ = −Jᵀr are solved for the step; λ shrinks on an
    accepted step and grows on a rejected one, which lets the method take
    Newton-sized steps near the solution and cautious ones far from it.
    """
    q = np.array(q0, dtype=np.float64)
    free = np.flatnonzero(free_mask)
    lam = 1e-3

    R = residual_fn(q)
    if R.size == 0:
        return _finish(q, True, 0, 0.0, residual_fn, jacobian_fn, free)

    def cost(x):  # the least-squares objective the step actually minimises
        r = residual_fn(x)
        return float(r @ r)

    converged = False
    iteration = 0
    for iteration in range(1, max_iter + 1):
        R = residual_fn(q)
        if float(np.max(np.abs(R))) < tol:
            converged = True
            break
        J = jacobian_fn(q)[:, free]
        c0 = R @ R

        # Try a full Gauss–Newton (least-squares) step first. Near the solution
        # this converges quadratically; `lstsq` also copes with a rank-deficient
        # Jacobian (an under-constrained sketch) by taking the minimum-norm step.
        step, *_ = np.linalg.lstsq(J, -R, rcond=None)
        trial = q.copy()
        trial[free] += step
        if cost(trial) < c0:
            q = trial
            lam = max(lam * 0.3, 1e-12)
            continue

        # Gauss–Newton overshot: fall back to Levenberg–Marquardt, growing the
        # damping until a step decreases the objective.
        g = J.T @ R
        H = J.T @ J
        eye = np.eye(H.shape[0])
        for _ in range(30):
            damped = np.linalg.solve(H + lam * eye, -g)
            trial = q.copy()
            trial[free] += damped
            if cost(trial) < c0:
                q = trial
                lam = min(lam * 2.0, 1e12)
                break
            lam = min(lam * 4.0, 1e12)
        else:
            # No step decreased the objective: singular / inconsistent. Stop and
            # let the DOF analysis explain why.
            break

    final_norm = float(np.max(np.abs(residual_fn(q)))) if residual_fn(q).size else 0.0
    converged = converged or final_norm < tol
    return _finish(q, converged, iteration, final_norm, residual_fn, jacobian_fn, free)


def _finish(q, converged, iterations, final_norm, residual_fn, jacobian_fn, free) -> SolveResult:
    R = residual_fn(q)
    m = int(R.size)
    n = int(free.size)
    if m == 0 or n == 0:
        rank = 0
    else:
        J = jacobian_fn(q)[:, free]
        rank = int(np.linalg.matrix_rank(J, tol=_rank_tol(J)))
    return SolveResult(
        q=q,
        converged=bool(converged),
        iterations=int(iterations),
        residual_norm=float(final_norm),
        n_params=n,
        n_equations=m,
        rank=rank,
        dof=n - rank,
        redundant=m - rank,
    )


def _rank_tol(J: np.ndarray) -> float:
    """A scale-aware singular-value threshold for the numerical rank."""
    if J.size == 0:
        return 0.0
    s = np.linalg.svd(J, compute_uv=False)
    return s[0] * max(J.shape) * np.finfo(float).eps if s.size else 0.0
