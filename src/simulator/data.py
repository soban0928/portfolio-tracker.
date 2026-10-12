"""Market data access: daily OHLCV bars and GBP/USD FX rates from CSV files.

CSV format (one file per ticker, file name ``<TICKER>.csv``)::

    Date,Open,High,Low,Close,Volume
    2019-01-02,39.48,40.12,39.20,39.93,148158800

The FX file ``GBPUSD.csv`` uses the same layout; its ``Close`` is the number of
US dollars per one pound sterling.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from simulator.config import (
    BASE_CURRENCY,
    BENCHMARK_TICKER,
    DOWNLOADED_DATA_DIR,
    FX_TICKER,
    SAMPLE_DATA_DIR,
)
from simulator.errors import NoPriceDataError, UnknownTickerError

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

DateLike = date | str | pd.Timestamp


def to_timestamp(when: DateLike) -> pd.Timestamp:
    """Normalise a date-like value to a midnight, timezone-naive ``Timestamp``."""
    ts = pd.Timestamp(when)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def currency_for(ticker: str) -> str:
    """Infer the trading currency of a ticker from its exchange suffix.

    London listings (``.L`` suffix, Yahoo convention) are treated as GBP;
    everything else is assumed to be a US listing priced in USD.
    """
    return "GBP" if ticker.upper().endswith(".L") else "USD"


@dataclass(frozen=True)
class Bar:
    """A single daily OHLCV price bar in the instrument's own currency."""

    date: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


def load_price_csv(path: Path) -> pd.DataFrame:
    """Load and validate one price CSV into a date-indexed DataFrame.

    Args:
        path: Path to a CSV with at least Date, Open, High, Low and Close columns.

    Returns:
        DataFrame indexed by ``Date`` (sorted, de-duplicated) with ``PRICE_COLUMNS``.

    Raises:
        ValueError: If required columns are missing.
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip().title() for c in df.columns]
    missing = {"Date", "Open", "High", "Low", "Close"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df.drop_duplicates("Date", keep="last").set_index("Date").sort_index()
    return df[PRICE_COLUMNS].astype(float)


class PriceStore:
    """In-memory store of daily prices for many tickers plus GBP/USD FX.

    All prices are kept in the instrument's native currency; use
    :meth:`to_base` / :meth:`fx_rate` to convert into GBP.
    """

    def __init__(self, frames: dict[str, pd.DataFrame], fx: pd.Series | None = None) -> None:
        """Create a store from pre-loaded frames.

        Args:
            frames: Mapping of ticker -> OHLCV DataFrame indexed by date.
            fx: Series of USD per 1 GBP indexed by date. Required if any ticker
                is priced in a currency other than GBP.
        """
        self._frames = {t.upper(): f.sort_index() for t, f in frames.items()}
        self._fx = fx.sort_index() if fx is not None else None
        needs_fx = any(currency_for(t) != BASE_CURRENCY for t in self._frames)
        if needs_fx and (self._fx is None or self._fx.empty):
            raise ValueError("FX series (GBPUSD) is required for non-GBP tickers")

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_directories(cls, *directories: Path) -> PriceStore:
        """Load every ``*.csv`` from the given directories.

        Files in later directories override same-named files in earlier ones,
        so ``from_directories(SAMPLE_DATA_DIR, DOWNLOADED_DATA_DIR)`` prefers
        freshly downloaded data. Missing directories are skipped.
        """
        frames: dict[str, pd.DataFrame] = {}
        for directory in directories:
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.csv")):
                frames[path.stem.upper()] = load_price_csv(path)
        fx_frame = frames.pop(FX_TICKER, None)
        fx = fx_frame["Close"] if fx_frame is not None else None
        if not frames:
            raise FileNotFoundError(f"No price CSVs found in {[str(d) for d in directories]}")
        return cls(frames, fx)

    @classmethod
    def default(cls) -> PriceStore:
        """Load bundled sample data, overridden by any downloaded data."""
        return cls.from_directories(SAMPLE_DATA_DIR, DOWNLOADED_DATA_DIR)

    # ------------------------------------------------------------------ queries
    @property
    def tickers(self) -> list[str]:
        """All tradeable tickers (sorted), including the benchmark."""
        return sorted(self._frames)

    @property
    def benchmark(self) -> str:
        """Benchmark ticker (S&P 500 ETF)."""
        return BENCHMARK_TICKER

    def frame(self, ticker: str) -> pd.DataFrame:
        """Return the full OHLCV frame for ``ticker``.

        Raises:
            UnknownTickerError: If the ticker is not loaded.
        """
        try:
            return self._frames[ticker.upper()]
        except KeyError:
            raise UnknownTickerError(f"Unknown ticker: {ticker}") from None

    def date_range(self, ticker: str | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
        """First and last available dates for a ticker, or across all tickers."""
        if ticker is not None:
            idx = self.frame(ticker).index
            return idx[0], idx[-1]
        starts = [f.index[0] for f in self._frames.values()]
        ends = [f.index[-1] for f in self._frames.values()]
        return min(starts), max(ends)

    def trading_days(
        self, start: DateLike, end: DateLike, ticker: str | None = None
    ) -> pd.DatetimeIndex:
        """Trading days between ``start`` and ``end`` inclusive.

        Uses the given ticker's calendar, or the benchmark's if available,
        otherwise the union of all tickers' dates.
        """
        s, e = to_timestamp(start), to_timestamp(end)
        if ticker is None and BENCHMARK_TICKER in self._frames:
            ticker = BENCHMARK_TICKER
        if ticker is not None:
            idx = self.frame(ticker).index
        else:
            idx = pd.DatetimeIndex(sorted(set().union(*(f.index for f in self._frames.values()))))
        return idx[(idx >= s) & (idx <= e)]

    def get_bar(self, ticker: str, when: DateLike) -> Bar:
        """Return the bar for ``ticker`` on exactly ``when``.

        Raises:
            UnknownTickerError: If the ticker is not loaded.
            NoPriceDataError: If there is no bar on that date (weekend, holiday,
                or outside the data range).
        """
        df = self.frame(ticker)
        ts = to_timestamp(when)
        if ts not in df.index:
            raise NoPriceDataError(f"No price data for {ticker.upper()} on {ts.date()}")
        row = df.loc[ts]
        return Bar(ts, row["Open"], row["High"], row["Low"], row["Close"], row["Volume"])

    def has_bar(self, ticker: str, when: DateLike) -> bool:
        """True if ``ticker`` has a bar on exactly ``when``."""
        try:
            self.get_bar(ticker, when)
        except NoPriceDataError:
            return False
        return True

    def close_on_or_before(self, ticker: str, when: DateLike) -> float:
        """Last close on or before ``when`` (used for marking positions to market).

        Raises:
            NoPriceDataError: If ``when`` is before the first available bar.
        """
        closes = self.frame(ticker)["Close"]
        ts = to_timestamp(when)
        pos = closes.index.searchsorted(ts, side="right") - 1
        if pos < 0:
            raise NoPriceDataError(f"No price data for {ticker.upper()} on or before {ts.date()}")
        return float(closes.iloc[pos])

    def closes(
        self, ticker: str, start: DateLike | None = None, end: DateLike | None = None
    ) -> pd.Series:
        """Close prices for ``ticker`` in its native currency, optionally sliced."""
        s = self.frame(ticker)["Close"]
        if start is not None:
            s = s[s.index >= to_timestamp(start)]
        if end is not None:
            s = s[s.index <= to_timestamp(end)]
        return s.rename(ticker.upper())

    def closes_in_base(
        self, ticker: str, start: DateLike | None = None, end: DateLike | None = None
    ) -> pd.Series:
        """Close prices converted to GBP using each day's FX rate."""
        s = self.closes(ticker, start, end)
        ccy = currency_for(ticker)
        if ccy == BASE_CURRENCY:
            return s
        rates = self.fx_series().reindex(s.index, method="ffill")
        if rates.isna().any():
            raise NoPriceDataError(f"No FX data covering {ticker.upper()} prices")
        return s / rates

    # ---------------------------------------------------------------------- FX
    def fx_series(self) -> pd.Series:
        """GBP/USD series (USD per 1 GBP)."""
        if self._fx is None:
            raise NoPriceDataError("No FX data loaded")
        return self._fx

    def fx_rate(self, when: DateLike) -> float:
        """USD per 1 GBP on ``when``, using the last available rate on or before it.

        Raises:
            NoPriceDataError: If no FX rate exists on or before ``when``.
        """
        fx = self.fx_series()
        ts = to_timestamp(when)
        pos = fx.index.searchsorted(ts, side="right") - 1
        if pos < 0:
            raise NoPriceDataError(f"No GBP/USD rate on or before {ts.date()}")
        return float(fx.iloc[pos])

    def to_base(self, amount: float, currency: str, when: DateLike) -> float:
        """Convert ``amount`` in ``currency`` into GBP at the rate on ``when``."""
        if currency == BASE_CURRENCY:
            return amount
        if currency != "USD":
            raise ValueError(f"Unsupported currency: {currency}")
        return amount / self.fx_rate(when)

    def from_base(self, amount: float, currency: str, when: DateLike) -> float:
        """Convert a GBP ``amount`` into ``currency`` at the rate on ``when``."""
        if currency == BASE_CURRENCY:
            return amount
        if currency != "USD":
            raise ValueError(f"Unsupported currency: {currency}")
        return amount * self.fx_rate(when)

    def currency(self, ticker: str) -> str:
        """Trading currency of a loaded ticker."""
        self.frame(ticker)  # validate
        return currency_for(ticker)


def load_store(directories: Iterable[Path] | None = None) -> PriceStore:
    """Convenience wrapper: load a :class:`PriceStore` from directories or defaults."""
    if directories is None:
        return PriceStore.default()
    return PriceStore.from_directories(*directories)
