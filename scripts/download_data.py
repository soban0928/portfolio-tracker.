"""Download daily price history from Yahoo Finance into ``data/downloaded/``.

Usage::

    pip install yfinance
    python scripts/download_data.py                       # default tickers, 2019-2024
    python scripts/download_data.py --tickers AAPL TSLA --start 2020-01-01

Prices are split/dividend adjusted (``auto_adjust=True``). The GBP/USD FX
series is always downloaded too (Yahoo symbol ``GBPUSD=X``) and saved as
``GBPUSD.csv``. Files in ``data/downloaded/`` take precedence over the bundled
synthetic samples in ``data/sample/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "downloaded"
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "SPY", "VOD.L"]
FX_SYMBOL = "GBPUSD=X"


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance output to Date, Open, High, Low, Close, Volume."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna()


def download(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Download one symbol's daily bars."""
    import yfinance as yf  # imported lazily: optional dependency

    raw = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"No data returned for {symbol}")
    return normalise(raw)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    failures = 0
    for symbol in [*args.tickers, FX_SYMBOL]:
        name = "GBPUSD" if symbol == FX_SYMBOL else symbol.upper()
        try:
            df = download(symbol, args.start, args.end)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[fail] {symbol}: {exc}", file=sys.stderr)
            failures += 1
            continue
        path = args.out / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"[ok]   {symbol} -> {path.relative_to(ROOT)} ({len(df)} rows)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
