"""Strategy backtesting on top of the portfolio engine.

How a backtest runs:

1. A :class:`Strategy` turns close prices into a DataFrame of **target
   weights** (date x ticker, each row summing to <= 1). Signals only use
   rolling/lagged calculations, so the weight on day *t* depends only on
   data up to and including the close of day *t*.
2. The runner executes the weights decided at the close of day *t* at the
   **open of day t+1** with market orders through a real
   :class:`~simulator.portfolio.Portfolio`, so commission, slippage, FX
   conversion and order validation are identical to manual trading.
3. Positions are only rebalanced when the target weights change (not for
   daily drift), which keeps commission costs realistic.
4. The benchmark is run through the same engine as buy-and-hold SPY, so it
   pays the same fees.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from simulator.analytics import performance_summary
from simulator.config import BASE_CURRENCY, STARTING_CASH, FeeConfig
from simulator.data import DateLike, PriceStore, to_timestamp
from simulator.errors import NoPriceDataError
from simulator.orders import make_order, max_affordable_quantity
from simulator.portfolio import Portfolio

# ---------------------------------------------------------------- strategies


class Strategy(ABC):
    """Base class: maps a close-price panel to target portfolio weights."""

    name: str = "Strategy"

    @abstractmethod
    def target_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        """Target weights per date and ticker, using only data up to each date.

        Args:
            closes: Close prices, indexed by date, one column per ticker.

        Returns:
            DataFrame aligned with ``closes`` holding weights in [0, 1].
        """

    def __repr__(self) -> str:
        return self.name


class BuyAndHold(Strategy):
    """Equal-weight every ticker on day one and never trade again."""

    name = "Buy & Hold"

    def target_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        """Constant equal weights."""
        n = closes.shape[1]
        return pd.DataFrame(1.0 / n, index=closes.index, columns=closes.columns)


@dataclass
class MovingAverageCrossover(Strategy):
    """Hold a ticker (at 1/N weight) while its short MA is above its long MA.

    Attributes:
        short_window: Fast moving-average length in days.
        long_window: Slow moving-average length in days.
    """

    short_window: int = 50
    long_window: int = 200

    def __post_init__(self) -> None:
        if not 0 < self.short_window < self.long_window:
            raise ValueError("need 0 < short_window < long_window")

    @property
    def name(self) -> str:  # type: ignore[override]
        """Display name including parameters."""
        return f"MA Crossover ({self.short_window}/{self.long_window})"

    def target_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        """1/N weight when short MA > long MA, else 0 (NaN warm-up -> 0)."""
        short = closes.rolling(self.short_window).mean()
        long = closes.rolling(self.long_window).mean()
        signal = (short > long) & long.notna()
        return signal.astype(float) / closes.shape[1]


@dataclass
class Momentum(Strategy):
    """Periodically hold the top-N tickers by trailing return.

    Attributes:
        lookback: Days of trailing return used for ranking.
        top_n: Number of tickers to hold (each at weight 1/top_n).
        rebalance_every: Re-rank every this many trading days.
        absolute: If True, only hold tickers with a positive trailing return
            (the rest of the allocation stays in cash).
    """

    lookback: int = 126
    top_n: int = 2
    rebalance_every: int = 21
    absolute: bool = True

    def __post_init__(self) -> None:
        if self.lookback < 1 or self.top_n < 1 or self.rebalance_every < 1:
            raise ValueError("lookback, top_n and rebalance_every must be >= 1")

    @property
    def name(self) -> str:  # type: ignore[override]
        """Display name including parameters."""
        return f"Momentum ({self.lookback}d, top {self.top_n})"

    def target_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        """Weights set on rebalance dates and carried forward in between."""
        mom = closes.pct_change(self.lookback, fill_method=None)
        weights = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
        for day in closes.index[:: self.rebalance_every]:
            row = mom.loc[day].dropna()
            if self.absolute:
                row = row[row > 0]
            weights.loc[day] = 0.0
            weights.loc[day, row.nlargest(self.top_n).index] = 1.0 / self.top_n
        return weights.ffill().fillna(0.0)


# ---------------------------------------------------------------- runner


@dataclass
class BacktestResult:
    """Outcome of one backtest run."""

    strategy: str
    values: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, float]
    portfolio: Portfolio = field(repr=False)


def _close_panel(store: PriceStore, tickers: list[str], end: pd.Timestamp) -> pd.DataFrame:
    """Native close prices up to ``end`` (full history for indicator warm-up)."""
    panel = pd.concat([store.closes(t, end=end) for t in tickers], axis=1)
    panel.columns = tickers
    return panel


def _simulate(
    store: PriceStore,
    strategy: Strategy,
    tickers: list[str],
    start: DateLike,
    end: DateLike,
    initial_cash: float,
    fees: FeeConfig,
) -> tuple[pd.Series, Portfolio]:
    """Run the day-by-day simulation; returns daily GBP values and the portfolio."""
    if not tickers:
        raise ValueError("At least one ticker is required")
    tickers = [t.upper() for t in tickers]
    start_ts, end_ts = to_timestamp(start), to_timestamp(end)
    days = store.trading_days(start_ts, end_ts)
    if len(days) < 2:
        raise ValueError("Backtest needs at least two trading days in the range")

    closes = _close_panel(store, tickers, end_ts)
    weights = strategy.target_weights(closes).reindex(days).fillna(0.0)

    pf = Portfolio(name=strategy.name, starting_cash=initial_cash, fees=fees)
    current: np.ndarray | None = None
    for prev_day, day in zip(days[:-1], days[1:], strict=True):
        target = weights.loc[prev_day].to_numpy()
        if current is not None and np.allclose(target, current):
            continue
        if _rebalance(pf, store, tickers, target, prev_day, day):
            current = target

    values = pf.value_history(store, days[0], days[-1])["total"].rename(strategy.name)
    return values, pf


def run_backtest(
    store: PriceStore,
    strategy: Strategy,
    tickers: list[str],
    start: DateLike,
    end: DateLike,
    initial_cash: float = STARTING_CASH,
    fees: FeeConfig | None = None,
    risk_free: float = 0.0,
    benchmark: pd.Series | None = None,
) -> BacktestResult:
    """Simulate ``strategy`` on ``tickers`` between ``start`` and ``end``.

    Args:
        store: Price data.
        strategy: Strategy producing target weights.
        tickers: Universe to trade.
        start: First trading day (no position is held at its close; the first
            possible trade is on the following open).
        end: Last trading day.
        initial_cash: Starting GBP cash.
        fees: Execution costs (same model as manual trading).
        risk_free: Annual risk-free rate for the Sharpe ratio / alpha.
        benchmark: Pre-computed benchmark values (from :func:`benchmark_run`);
            computed if omitted.

    Returns:
        A :class:`BacktestResult` with daily values, trades and metrics.
    """
    fees = fees or FeeConfig()
    values, pf = _simulate(store, strategy, tickers, start, end, initial_cash, fees)
    if benchmark is None:
        benchmark = benchmark_run(store, start, end, initial_cash, fees)
    metrics = performance_summary(values, benchmark, risk_free)
    return BacktestResult(strategy.name, values, pf.trades_frame(), metrics, pf)


def _rebalance(
    pf: Portfolio,
    store: PriceStore,
    tickers: list[str],
    target: np.ndarray,
    decision_day: pd.Timestamp,
    exec_day: pd.Timestamp,
) -> bool:
    """Trade towards ``target`` weights at ``exec_day``'s open.

    Sizing uses the portfolio value and prices at ``decision_day``'s close (the
    information available when the decision is made). Buys are then clamped to
    what cash can afford at the actual open. Sells execute before buys.

    Returns:
        True if every ticker had a bar on ``exec_day`` (rebalance complete).
    """
    tradable = [t for t in tickers if store.has_bar(t, exec_day)]
    value = pf.total_value(store, decision_day)
    desired: dict[str, int] = {}
    for ticker, weight in zip(tickers, target, strict=True):
        if ticker not in tradable:
            continue
        try:
            px = store.to_base(
                store.close_on_or_before(ticker, decision_day),
                store.currency(ticker),
                decision_day,
            )
        except NoPriceDataError:  # no history yet: can't size, hold 0
            px = math.nan
        desired[ticker] = int(weight * value // px) if weight > 0 and px > 0 else 0

    for ticker, qty in desired.items():  # sells first to free cash
        delta = qty - pf.shares_held(ticker)
        if delta < 0:
            pf.place_order(make_order(ticker, "sell", -delta, exec_day), store)
    for ticker, qty in desired.items():
        delta = qty - pf.shares_held(ticker)
        if delta <= 0:
            continue
        bar = store.get_bar(ticker, exec_day)
        ccy = store.currency(ticker)
        fx = store.fx_rate(exec_day) if ccy != BASE_CURRENCY else 1.0
        qty_ok = min(delta, max_affordable_quantity(pf.cash, bar.open, fx, pf.fees))
        if qty_ok > 0:
            pf.place_order(make_order(ticker, "buy", qty_ok, exec_day), store)
    return len(tradable) == len(tickers)


def benchmark_run(
    store: PriceStore,
    start: DateLike,
    end: DateLike,
    initial_cash: float = STARTING_CASH,
    fees: FeeConfig | None = None,
) -> pd.Series:
    """Daily GBP value of buy-and-hold in the benchmark, with the same fees."""
    values, _ = _simulate(
        store, BuyAndHold(), [store.benchmark], start, end, initial_cash, fees or FeeConfig()
    )
    return values.rename(f"Benchmark ({store.benchmark})")


def compare_strategies(
    store: PriceStore,
    strategies: list[Strategy],
    tickers: list[str],
    start: DateLike,
    end: DateLike,
    initial_cash: float = STARTING_CASH,
    fees: FeeConfig | None = None,
    risk_free: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run several strategies plus the benchmark over the same period.

    Returns:
        ``(values, metrics)``: daily values (one column per strategy plus the
        benchmark) and a metrics table (one row per strategy plus the benchmark).
    """
    bench = benchmark_run(store, start, end, initial_cash, fees)
    series = []
    rows = {}
    for strat in strategies:
        res = run_backtest(
            store, strat, tickers, start, end, initial_cash, fees, risk_free, benchmark=bench
        )
        series.append(res.values)
        rows[res.strategy] = {**res.metrics, "trades": len(res.trades)}
    rows[str(bench.name)] = {
        **performance_summary(bench, bench, risk_free),
        "trades": 1,
    }
    values = pd.concat([*series, bench], axis=1)
    return values, pd.DataFrame(rows).T
