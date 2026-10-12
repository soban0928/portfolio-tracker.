"""Global configuration and defaults for the simulator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Base (reporting) currency of every portfolio.
BASE_CURRENCY = "GBP"

#: Cash every new portfolio starts with, in the base currency.
STARTING_CASH = 10_000.0

#: Trading days per year, used to annualise returns and volatility.
TRADING_DAYS = 252

#: Project root (two levels above this file: src/simulator/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Bundled sample data that ships with the repo.
SAMPLE_DATA_DIR = PROJECT_ROOT / "data" / "sample"

#: Where ``scripts/download_data.py`` writes fresh data (git-ignored).
DOWNLOADED_DATA_DIR = PROJECT_ROOT / "data" / "downloaded"

#: Default SQLite database path.
DEFAULT_DB_PATH = PROJECT_ROOT / "simulator.db"

#: Ticker used as the market benchmark.
BENCHMARK_TICKER = "SPY"

#: Ticker of the FX series giving USD per 1 GBP.
FX_TICKER = "GBPUSD"


@dataclass(frozen=True)
class FeeConfig:
    """Execution costs applied to every trade.

    Attributes:
        commission: Flat commission per trade, in the base currency (GBP).
        slippage_bps: Slippage/half-spread in basis points. Buys fill this much
            above the reference price and sells this much below it.
    """

    commission: float = 5.0
    slippage_bps: float = 10.0

    def __post_init__(self) -> None:
        if self.commission < 0:
            raise ValueError("commission must be non-negative")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")

    @property
    def slippage_rate(self) -> float:
        """Slippage as a fraction (e.g. 10 bps -> 0.001)."""
        return self.slippage_bps / 10_000.0
