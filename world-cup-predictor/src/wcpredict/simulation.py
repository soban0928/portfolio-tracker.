"""Vectorised Monte Carlo simulation of a World Cup.

Per batch of simulations (default 1,000):

1. **Parameter uncertainty** - draw one coefficient vector from the goals
   model's approximate posterior N(β, Σ) and build the matrix of expected
   goals for every ordered pair of teams.
2. **Group stage** - sample full scorelines for every group fixture from the
   Dixon-Coles score distribution (inverse-CDF sampling on the 11x11 grid),
   then rank groups with the tournament's tiebreak rules.
3. **Best thirds** (48-team format) - rank third-placed teams; map the set of
   qualifying groups to bracket slots through a precomputed lookup.
4. **Knockouts** - a team's chance of beating its opponent is
   ``P(win 90') + P(draw 90') * [P(win ET) + P(draw ET) * 0.5]`` where extra
   time uses independent Poisson goals at one third of the 90-minute rates and
   penalties are a coin flip.

Batches share a single ``numpy.random.Generator`` seeded once, so results are
reproducible. Monte Carlo standard errors are reported two ways: the naive
binomial ``sqrt(p(1-p)/N)`` and the batch-means estimate (std of per-batch
proportions / sqrt(batches)), which is the correct one when parameters are
re-sampled per batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import poisson

from wcpredict.config import SEED
from wcpredict.goals import MAX_GOALS, GoalsModel, score_matrix
from wcpredict.tournament import GROUP_PAIRS, Structure, rank_groups, rank_thirds

ET_FRACTION = 1.0 / 3.0


def knockout_win_matrix(lam: np.ndarray, rho: float) -> tuple[np.ndarray, np.ndarray]:
    """P(row team beats column team) in a knockout tie, and P(tie goes to penalties).

    Args:
        lam: (T, T) expected 90-minute goals of the row team against the column team.
        rho: Dixon-Coles correlation parameter.
    """
    T = lam.shape[0]
    m = score_matrix(lam.ravel(), lam.T.ravel(), rho)  # row team = "home"
    win = np.tril(np.ones((MAX_GOALS + 1,) * 2), -1)
    p_win = (m * win).sum(axis=(1, 2)).reshape(T, T)
    p_draw = np.trace(m, axis1=1, axis2=2).reshape(T, T)
    g = np.arange(MAX_GOALS + 1)
    ph = poisson.pmf(g[None, :], (lam * ET_FRACTION).ravel()[:, None])
    pa = poisson.pmf(g[None, :], (lam.T * ET_FRACTION).ravel()[:, None])
    et = ph[:, :, None] * pa[:, None, :]
    et /= et.sum(axis=(1, 2), keepdims=True)
    et_win = (et * win).sum(axis=(1, 2)).reshape(T, T)
    et_draw = np.trace(et, axis1=1, axis2=2).reshape(T, T)
    beat = p_win + p_draw * (et_win + 0.5 * et_draw)
    np.fill_diagonal(beat, 0.5)
    return beat, p_draw * et_draw


@dataclass
class SimulationInputs:
    """Everything the simulator needs for one tournament.

    Attributes:
        structure: Groups and bracket.
        goals_model: Fitted goals model (rates + parameter covariance).
        pair_features: Feature rows for every ordered pair (i, j), i != j, with
            columns ``i`` and ``j`` (team ids) plus the goals-model features
            (group-stage venues).
        elo: (T,) pre-tournament Elo, used as the last tiebreaker.
        ko_pair_features: Same for knockout venues (defaults to ``pair_features``).
    """

    structure: Structure
    goals_model: GoalsModel
    pair_features: pd.DataFrame
    elo: np.ndarray
    ko_pair_features: pd.DataFrame | None = None

    def rate_matrix(self, beta: np.ndarray | None = None, knockout: bool = False) -> np.ndarray:
        """(T, T) expected goals of team i against team j."""
        pf = (
            self.ko_pair_features
            if knockout and self.ko_pair_features is not None
            else (self.pair_features)
        )
        T = len(self.structure.teams)
        lam_h, _ = self.goals_model.rates(pf, beta)
        lam = np.ones((T, T))
        lam[pf["i"].to_numpy(), pf["j"].to_numpy()] = lam_h
        return lam


@dataclass
class SimulationResult:
    """Aggregated simulation output.

    Attributes:
        stages: Per-team probabilities of reaching each stage, with MCSE columns.
        positions: Per-team probabilities of each group finishing position.
        slot_counts: For each knockout round, (matches, 2, T) counts of which team
            occupied each side of each match.
        winner_counts: For each knockout round, (matches, T) counts of match winners.
        meta: Run metadata (sims, batches, seed, penalty share, ...).
    """

    stages: pd.DataFrame
    positions: pd.DataFrame
    slot_counts: list[np.ndarray]
    winner_counts: list[np.ndarray]
    meta: dict[str, float] = field(default_factory=dict)


def _sample_scores(
    cdf: np.ndarray, rng: np.random.Generator, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sample scorelines; ``cdf`` is (F, 121) cumulative probabilities for F fixtures."""
    u = rng.random((n, cdf.shape[0], 1))
    idx = (cdf[None, :, :] < u).sum(axis=-1)
    idx = np.minimum(idx, cdf.shape[1] - 1)
    return idx // (MAX_GOALS + 1), idx % (MAX_GOALS + 1)


@dataclass
class _Precomputed:
    """Batch-independent arrays derived from the structure."""

    groups: np.ndarray  # (G, 4) team ids
    gi: np.ndarray  # (G, 6) first team of each group fixture
    gj: np.ndarray  # (G, 6) second team
    elo_g: np.ndarray  # (G, 4)
    lookup: np.ndarray | None  # (2**G, slots) third-place assignment by bitmask


def _precompute(inputs: SimulationInputs) -> _Precomputed:
    s = inputs.structure
    G = len(s.groups)
    gi = np.array([[s.groups[g][i] for i, _ in GROUP_PAIRS] for g in range(G)])
    gj = np.array([[s.groups[g][j] for _, j in GROUP_PAIRS] for g in range(G)])
    lookup = None
    if s.n_best_thirds:
        lookup = np.full((1 << G, s.n_best_thirds), -1, dtype=int)
        for combo, assign in s.third_lookup.items():
            lookup[sum(1 << g for g in combo)] = assign
    return _Precomputed(s.groups, gi, gj, inputs.elo[s.groups], lookup)


def _run_batch(
    inputs: SimulationInputs,
    pre: _Precomputed,
    lam: np.ndarray,
    beat: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Simulate ``n`` tournaments with fixed rates.

    Returns:
        ``placed`` (n, G, 4) team ids by finishing position, and for each knockout
        round a tuple (side_a, side_b, winners), each (n, matches).
    """
    s = inputs.structure
    G = len(pre.groups)
    m = score_matrix(lam[pre.gi.ravel(), pre.gj.ravel()], lam[pre.gj.ravel(), pre.gi.ravel()],
                     inputs.goals_model.rho_)  # fmt: skip
    hg, ag = _sample_scores(m.reshape(len(m), -1).cumsum(axis=1), rng, n)
    tables = rank_groups(hg.reshape(n, G, 6), ag.reshape(n, G, 6),
                         np.broadcast_to(pre.elo_g, (n, G, 4)), s.rules)  # fmt: skip
    placed = np.take_along_axis(np.broadcast_to(pre.groups, (n, G, 4)), tables.order, axis=-1)

    assign = None
    if pre.lookup is not None:
        best = rank_thirds(
            tables.stat_of_position(tables.points, 3),
            tables.stat_of_position(tables.goal_diff, 3),
            tables.stat_of_position(tables.goals_for, 3),
            inputs.elo[placed[:, :, 2]],
        )[:, : s.n_best_thirds]
        assign = pre.lookup[(1 << best).sum(axis=1)]  # (n, slots)

    def resolve(slot: tuple) -> np.ndarray:
        if slot[0] == "pos":
            return placed[:, slot[1], slot[2] - 1]
        return placed[np.arange(n), assign[:, slot[1]], 2]  # type: ignore[index]

    a = np.column_stack([resolve(x) for x, _ in s.first_round])
    c = np.column_stack([resolve(y) for _, y in s.first_round])
    rounds = []
    for r in range(len(s.later_rounds) + 1):
        if r > 0:
            prev = rounds[-1][2]
            pairs = s.later_rounds[r - 1]
            a = prev[:, [x for x, _ in pairs]]
            c = prev[:, [y for _, y in pairs]]
        winners = np.where(rng.random(a.shape) < beat[a, c], a, c)
        rounds.append((a, c, winners))
    return placed, rounds


def simulate(
    inputs: SimulationInputs,
    n_sims: int = 100_000,
    batch_size: int = 1_000,
    seed: int = SEED,
    sample_params: bool = True,
) -> SimulationResult:
    """Simulate the tournament ``n_sims`` times.

    Args:
        inputs: Structure, model and pair features.
        n_sims: Number of simulated tournaments.
        batch_size: Simulations per parameter draw.
        seed: Seed for the single random generator.
        sample_params: Draw goals-model coefficients per batch (model uncertainty).

    Returns:
        A :class:`SimulationResult`.
    """
    s = inputs.structure
    rng = np.random.default_rng(seed)
    T = len(s.teams)
    n_batches = int(np.ceil(n_sims / batch_size))
    sizes = np.array([min(batch_size, n_sims - b * batch_size) for b in range(n_batches)])
    all_stages = ["R32", "R16", "QF", "SF", "FINAL", "WINNER"]
    stage_names = all_stages[all_stages.index(s.round_names[0]) :]
    reach = np.zeros((n_batches, T, len(stage_names)))
    pos_counts = np.zeros((T, 4))
    n_rounds = len(s.later_rounds) + 1
    match_counts = [len(s.first_round)] + [len(r) for r in s.later_rounds]
    slot_counts = [np.zeros((k, 2, T)) for k in match_counts]
    winner_counts = [np.zeros((k, T)) for k in match_counts]
    pen_share = []

    pre = _precompute(inputs)
    model = inputs.goals_model
    if sample_params:
        betas = model.sample_betas(n_batches, rng)
    else:
        betas = np.tile(model.beta_, (n_batches, 1))
    for b in range(n_batches):
        n = int(sizes[b])
        lam = inputs.rate_matrix(betas[b])
        beat, pens = knockout_win_matrix(inputs.rate_matrix(betas[b], knockout=True), model.rho_)
        pen_share.append(float(pens.mean()))
        placed, rounds = _run_batch(inputs, pre, lam, beat, n, rng)
        for p in range(4):
            pos_counts[:, p] += np.bincount(placed[:, :, p].ravel(), minlength=T)
        for r, (a, c, winners) in enumerate(rounds):
            for k in range(a.shape[1]):
                slot_counts[r][k, 0] += np.bincount(a[:, k], minlength=T)
                slot_counts[r][k, 1] += np.bincount(c[:, k], minlength=T)
                winner_counts[r][k] += np.bincount(winners[:, k], minlength=T)
            reach[b, :, r] = np.bincount(np.concatenate([a, c], axis=1).ravel(), minlength=T) / n
        reach[b, :, n_rounds] = np.bincount(rounds[-1][2].ravel(), minlength=T) / n

    weights = sizes / sizes.sum()
    p = np.tensordot(weights, reach, axes=1)  # (T, stages)
    if n_batches > 1:
        var = ((reach - p[None]) ** 2 * weights[:, None, None]).sum(0) * n_batches / (n_batches - 1)
        batch_se = np.sqrt(var / n_batches)
    else:
        batch_se = np.full_like(p, np.nan)
    binom_se = np.sqrt(p * (1 - p) / sizes.sum())

    stages = pd.DataFrame(p, index=s.teams, columns=[f"p_{x}" for x in stage_names])
    for k, x in enumerate(stage_names):
        stages[f"se_{x}"] = batch_se[:, k]
        stages[f"se_binom_{x}"] = binom_se[:, k]
    stages.insert(0, "elo", inputs.elo)
    stages = stages.sort_values("p_WINNER", ascending=False)
    positions = pd.DataFrame(pos_counts / sizes.sum(), index=s.teams,
                             columns=["p_1st", "p_2nd", "p_3rd", "p_4th"])  # fmt: skip
    meta = {
        "n_sims": int(sizes.sum()),
        "n_batches": n_batches,
        "batch_size": batch_size,
        "seed": seed,
        "sample_params": sample_params,
        "mean_penalty_share": float(np.mean(pen_share)),
    }
    return SimulationResult(stages, positions, slot_counts, winner_counts, meta)
