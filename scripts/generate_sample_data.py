"""Generate the bundled SYNTHETIC sample price data in ``data/sample/``.

The sandbox this project was built in could not reach Yahoo Finance, so the
offline samples are simulated with a seeded one-factor model:

* a market factor (``SPY``) follows geometric Brownian motion;
* each stock's daily return is ``beta * market + idiosyncratic noise``;
* open/high/low are derived from the close path with small random gaps/ranges;
* GBP/USD follows a low-volatility random walk around 1.28.

The output is deterministic (fixed seed) and only meant to make the app and
tests work offline. Run ``scripts/download_data.py`` for real prices.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "sample"
START, END = "2019-01-02", "2024-12-31"
SEED = 31

# ticker: (start price, beta, annual drift, idiosyncratic annual vol)
STOCKS: dict[str, tuple[float, float, float, float]] = {
    "AAPL": (39.0, 1.2, 0.18, 0.20),
    "MSFT": (98.0, 1.1, 0.16, 0.18),
    "GOOGL": (52.0, 1.05, 0.12, 0.22),
    "AMZN": (76.0, 1.25, 0.10, 0.26),
    "NVDA": (3.4, 1.7, 0.35, 0.38),
    "VOD.L": (1.55, 0.7, -0.08, 0.20),  # London listing, priced in GBP
}
MARKET = ("SPY", 250.0, 0.11, 0.18)  # ticker, start, drift, vol


def make_ohlc(close: np.ndarray, rng: np.random.Generator, intraday_vol: float) -> pd.DataFrame:
    """Build plausible OHLCV bars around a close-price path."""
    prev_close = np.concatenate([[close[0]], close[:-1]])
    open_ = prev_close * (1 + rng.normal(0, intraday_vol * 0.4, len(close)))
    top = np.maximum(open_, close)
    bottom = np.minimum(open_, close)
    high = top * (1 + np.abs(rng.normal(0, intraday_vol * 0.6, len(close))))
    low = bottom * (1 - np.abs(rng.normal(0, intraday_vol * 0.6, len(close))))
    volume = rng.integers(1_000_000, 50_000_000, len(close))
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume})


def main() -> None:
    """Write synthetic CSVs for every configured ticker plus GBPUSD."""
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range(START, END)
    n = len(dates)
    dt = 1 / 252
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    m_ticker, m_start, m_mu, m_sigma = MARKET
    market_ret = rng.normal((m_mu - 0.5 * m_sigma**2) * dt, m_sigma * np.sqrt(dt), n)
    market_ret[0] = 0.0
    series = {m_ticker: (m_start * np.exp(np.cumsum(market_ret)), m_sigma / np.sqrt(252))}

    for ticker, (start, beta, mu, idio) in STOCKS.items():
        eps = rng.normal((mu - beta * m_mu - 0.5 * idio**2) * dt, idio * np.sqrt(dt), n)
        ret = beta * market_ret + eps
        ret[0] = 0.0
        vol = np.sqrt((beta * m_sigma) ** 2 + idio**2) / np.sqrt(252)
        series[ticker] = (start * np.exp(np.cumsum(ret)), vol)

    for ticker, (close, daily_vol) in series.items():
        df = make_ohlc(close, rng, daily_vol)
        df.insert(0, "Date", dates.strftime("%Y-%m-%d"))
        df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].round(4)
        df.to_csv(OUT_DIR / f"{ticker}.csv", index=False)

    fx_ret = rng.normal(0, 0.08 * np.sqrt(dt), n)
    fx_close = 1.28 * np.exp(np.cumsum(fx_ret))
    fx = make_ohlc(fx_close, rng, 0.08 / np.sqrt(252))
    fx["Volume"] = 0
    fx.insert(0, "Date", dates.strftime("%Y-%m-%d"))
    fx[["Open", "High", "Low", "Close"]] = fx[["Open", "High", "Low", "Close"]].round(5)
    fx.to_csv(OUT_DIR / "GBPUSD.csv", index=False)
    print(f"Wrote {len(series) + 1} synthetic files ({n} rows each) to {OUT_DIR}")


if __name__ == "__main__":
    main()
