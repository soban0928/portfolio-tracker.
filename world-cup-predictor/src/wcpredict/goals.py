"""Dixon-Coles-style goals model with Elo/form-driven scoring rates.

Classic Dixon-Coles fits an attack and defence strength per team, which is
unstable for national teams that play rarely. Here each side's expected goals
come from a Poisson regression on match features::

    log λ_team = β0 + β1·(Elo_team − Elo_opp)/100 + β2·venue_team
               + β3·log(0.5 + gf10_team) + β4·log(0.5 + ga10_opp)
               + β5·is_friendly + β6·is_tournament

fitted on both sides of every match (mirrored rows) with exponential
time-decay weights. The Dixon-Coles correction τ(ρ) then adjusts the
probabilities of the low scores 0-0, 1-0, 0-1 and 1-1, with ρ fitted by
maximum likelihood given the λs.

The approximate covariance of β (inverse Fisher information) is kept so
simulations can sample parameters and propagate model uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln

MAX_GOALS = 10
TERMS = ["const", "elo", "venue", "attack", "defence", "friendly", "tournament"]


def design(X: pd.DataFrame, side: str) -> np.ndarray:
    """Design matrix for the goals scored by ``side`` ("home" or "away")."""
    sign = 1.0 if side == "home" else -1.0
    opp = "away" if side == "home" else "home"
    return np.column_stack(
        [
            np.ones(len(X)),
            sign * X["elo_diff"].to_numpy() / 100.0,
            sign * X["venue"].to_numpy(),
            np.log(0.5 + X[f"gf10_{side}"].to_numpy()),
            np.log(0.5 + X[f"ga10_{opp}"].to_numpy()),
            X["is_friendly"].to_numpy(),
            X["is_tournament"].to_numpy(),
        ]
    )


def tau(h: np.ndarray, a: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    """Dixon-Coles low-score adjustment factor."""
    out = np.ones(np.broadcast(h, a, lam).shape)
    out = np.where((h == 0) & (a == 0), 1 - lam * mu * rho, out)
    out = np.where((h == 0) & (a == 1), 1 + lam * rho, out)
    out = np.where((h == 1) & (a == 0), 1 + mu * rho, out)
    return np.where((h == 1) & (a == 1), 1 - rho, out)


def score_matrix(lam: np.ndarray, mu: np.ndarray, rho: float, max_goals: int = MAX_GOALS):
    """Scoreline probabilities, shape (n, max_goals+1, max_goals+1), rows = home goals.

    Probabilities are renormalised to sum to 1 after truncation at ``max_goals``.
    """
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    g = np.arange(max_goals + 1)
    log_fact = gammaln(g + 1)
    ph = np.exp(g[None, :] * np.log(lam[:, None]) - lam[:, None] - log_fact[None, :])
    pa = np.exp(g[None, :] * np.log(mu[:, None]) - mu[:, None] - log_fact[None, :])
    m = ph[:, :, None] * pa[:, None, :]
    hh, aa = np.meshgrid(g, g, indexing="ij")
    m = m * tau(hh[None], aa[None], lam[:, None, None], mu[:, None, None], rho)
    m = np.clip(m, 0, None)
    return m / m.sum(axis=(1, 2), keepdims=True)


def outcome_probs(matrix: np.ndarray) -> np.ndarray:
    """(n, k, k) score matrices -> (n, 3) probabilities of H, D, A."""
    home = np.tril(np.ones(matrix.shape[1:]), -1)  # i > j
    p_h = (matrix * home).sum(axis=(1, 2))
    p_d = np.trace(matrix, axis1=1, axis2=2)
    return np.column_stack([p_h, p_d, 1 - p_h - p_d])


@dataclass
class GoalsModel:
    """Poisson goals model with Dixon-Coles correction.

    Attributes:
        half_life_years: Time-decay half-life for match weights.
        ridge: L2 penalty on the non-intercept coefficients (stabilises fits).
    """

    half_life_years: float = 8.0
    ridge: float = 1.0
    name: str = "Dixon-Coles (goals)"
    features: list[str] = field(
        default_factory=lambda: [
            "elo_diff",
            "venue",
            "gf10_home",
            "ga10_home",
            "gf10_away",
            "ga10_away",
            "is_friendly",
            "is_tournament",
        ]  # fmt: skip
    )

    def _weights(self, dates: pd.Series) -> np.ndarray:
        age = (dates.max() - dates).dt.days.to_numpy() / 365.25
        return 0.5 ** (age / self.half_life_years)

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> GoalsModel:
        """Fit β by weighted Poisson MLE, then ρ given β.

        Args:
            X: Feature rows that also contain ``home_score``, ``away_score`` and ``date``.
            y: Ignored (outcomes are derived from the scores); kept for the shared interface.
        """
        w = self._weights(X["date"])
        D = np.vstack([design(X, "home"), design(X, "away")])
        goals = np.concatenate([X["home_score"].to_numpy(), X["away_score"].to_numpy()])
        ww = np.concatenate([w, w])
        penalty = np.full(D.shape[1], self.ridge)
        penalty[0] = 0.0

        def nll(beta: np.ndarray) -> tuple[float, np.ndarray]:
            eta = D @ beta
            lam = np.exp(eta)
            f = -(ww * (goals * eta - lam)).sum() + 0.5 * (penalty * beta**2).sum()
            grad = -D.T @ (ww * (goals - lam)) + penalty * beta
            return f, grad

        beta0 = np.zeros(D.shape[1])
        beta0[0] = np.log(goals.mean())
        res = minimize(nll, beta0, jac=True, method="L-BFGS-B")
        self.beta_ = res.x
        lam = np.exp(D @ self.beta_)
        fisher = (D * (ww * lam)[:, None]).T @ D + np.diag(penalty)
        self.cov_ = np.linalg.inv(fisher)

        n = len(X)
        lam_h, lam_a = lam[:n], lam[n:]
        hs, as_ = X["home_score"].to_numpy(), X["away_score"].to_numpy()
        low = (hs <= 1) & (as_ <= 1)

        def rho_nll(rho: float) -> float:
            t = tau(hs[low], as_[low], lam_h[low], lam_a[low], rho)
            if (t <= 0).any():
                return np.inf
            return -(w[low] * np.log(t)).sum()

        self.rho_ = float(minimize_scalar(rho_nll, bounds=(-0.2, 0.2), method="bounded").x)
        return self

    def rates(self, X: pd.DataFrame, beta: np.ndarray | None = None):
        """Expected goals (λ_home, λ_away) for each row."""
        b = self.beta_ if beta is None else beta
        return np.exp(design(X, "home") @ b), np.exp(design(X, "away") @ b)

    def score_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Scoreline probability matrices for each row."""
        lam, mu = self.rates(X)
        return score_matrix(lam, mu, self.rho_)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """W/D/L probabilities implied by the scoreline distribution."""
        return outcome_probs(self.score_matrix(X))

    def sample_betas(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw coefficient vectors from the approximate posterior N(β, Σ)."""
        return rng.multivariate_normal(self.beta_, self.cov_, size=n)

    def coefficients(self) -> pd.DataFrame:
        """Fitted coefficients with standard errors."""
        return pd.DataFrame(
            {"coef": self.beta_, "std_err": np.sqrt(np.diag(self.cov_))}, index=TERMS
        )


def top_scorelines(matrix: np.ndarray, k: int = 5) -> list[tuple[int, int, float]]:
    """The ``k`` most likely (home_goals, away_goals, probability) for one matrix."""
    flat = matrix.ravel()
    idx = np.argsort(flat)[::-1][:k]
    size = matrix.shape[1]
    return [(int(i // size), int(i % size), float(flat[i])) for i in idx]
