"""Performance analytics on daily value or price series.

Conventions:

* Returns are simple daily returns (``pct_change``).
* Annualisation uses :data:`~simulator.config.TRADING_DAYS` (252).
* Volatility uses the sample standard deviation (``ddof=1``).
* The annual risk-free rate is converted to a daily rate as ``rf / 252``.
* Sharpe = mean(daily excess return) / std(daily excess return) * sqrt(252).
* Beta = cov(r_p, r_b) / var(r_b) on aligned daily returns.
* Alpha is annualised Jensen's alpha:
  ``(mean(r_p - rf) - beta * mean(r_b - rf)) * 252``.
* Max drawdown is the worst peak-to-trough fall, reported as a negative fraction.

Functions return ``nan`` when there isn't enough data rather than raising,
so dashboards can render partially-filled portfolios.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from simulator.config import TRADING_DAYS
from simulator.data import DateLike, PriceStore

NAN = float("nan")


def daily_returns(values: pd.Series) -> pd.Series:
    """Simple daily returns of a value/price series (first NaN dropped)."""
    return values.astype(float).pct_change().dropna()


def total_return(values: pd.Series) -> float:
    """Return from first to last value as a fraction."""
    v = values.dropna()
    if len(v) < 2 or v.iloc[0] == 0:
        return NAN
    return float(v.iloc[-1] / v.iloc[0] - 1)


def annualised_return(values: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Compound annual growth rate, assuming ``periods`` observations per year."""
    v = values.dropna()
    n = len(v) - 1
    tr = total_return(v)
    if n < 1 or math.isnan(tr) or tr <= -1:
        return NAN
    return float((1 + tr) ** (periods / n) - 1)


def annualised_volatility(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Sample standard deviation of daily returns, scaled by sqrt(periods)."""
    r = returns.dropna()
    if len(r) < 2:
        return NAN
    return float(r.std(ddof=1) * math.sqrt(periods))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe ratio of daily returns.

    Args:
        returns: Daily simple returns.
        risk_free: Annual risk-free rate (e.g. 0.04 for 4%).
        periods: Observations per year.
    """
    r = returns.dropna()
    if len(r) < 2:
        return NAN
    excess = r - risk_free / periods
    sd = excess.std(ddof=1)
    if sd == 0 or math.isnan(sd):
        return NAN
    return float(excess.mean() / sd * math.sqrt(periods))


def drawdown_series(values: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak at each date (<= 0)."""
    v = values.astype(float)
    return v / v.cummax() - 1


def max_drawdown(values: pd.Series) -> float:
    """Largest peak-to-trough decline as a negative fraction (0 if none)."""
    v = values.dropna()
    if len(v) < 2:
        return NAN
    return float(drawdown_series(v).min())


def _align(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    df = pd.concat([a, b], axis=1, join="inner").dropna()
    return df.iloc[:, 0], df.iloc[:, 1]


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Sensitivity of returns to benchmark returns: cov(r, b) / var(b)."""
    r, b = _align(returns, benchmark_returns)
    if len(r) < 2:
        return NAN
    var_b = b.var(ddof=1)
    if var_b == 0 or math.isnan(var_b):
        return NAN
    return float(np.cov(r, b, ddof=1)[0, 1] / var_b)


def alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free: float = 0.0,
    periods: int = TRADING_DAYS,
) -> float:
    """Annualised Jensen's alpha versus the benchmark."""
    r, b = _align(returns, benchmark_returns)
    bt = beta(r, b)
    if math.isnan(bt):
        return NAN
    rf = risk_free / periods
    return float(((r - rf).mean() - bt * (b - rf).mean()) * periods)


def normalise(values: pd.Series, base: float = 100.0) -> pd.Series:
    """Rebase a series so its first value equals ``base`` (for comparison charts)."""
    v = values.dropna()
    if v.empty or v.iloc[0] == 0:
        return v
    return v / v.iloc[0] * base


def benchmark_values(
    store: PriceStore,
    start: DateLike,
    end: DateLike,
    initial: float,
    ticker: str | None = None,
) -> pd.Series:
    """Value of ``initial`` GBP invested in the benchmark at ``start`` (no fees).

    The benchmark is priced in GBP so it's comparable with a GBP portfolio.
    """
    prices = store.closes_in_base(ticker or store.benchmark, start, end)
    if prices.empty:
        return prices
    return (prices / prices.iloc[0] * initial).rename("benchmark")


def performance_summary(
    values: pd.Series,
    benchmark: pd.Series | None = None,
    risk_free: float = 0.0,
    periods: int = TRADING_DAYS,
) -> dict[str, float]:
    """All headline metrics for a value series, optionally vs a benchmark.

    Returns:
        Dict with total_return, annualised_return, annualised_volatility,
        sharpe_ratio, max_drawdown and, if a benchmark is given, beta, alpha
        and benchmark_return.
    """
    r = daily_returns(values)
    out = {
        "total_return": total_return(values),
        "annualised_return": annualised_return(values, periods),
        "annualised_volatility": annualised_volatility(r, periods),
        "sharpe_ratio": sharpe_ratio(r, risk_free, periods),
        "max_drawdown": max_drawdown(values),
    }
    if benchmark is not None:
        rb = daily_returns(benchmark)
        out["beta"] = beta(r, rb)
        out["alpha"] = alpha(r, rb, risk_free, periods)
        out["benchmark_return"] = total_return(benchmark)
    return out
