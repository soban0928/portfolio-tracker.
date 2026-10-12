"""Shared fixtures: a tiny, hand-checkable price store."""

from __future__ import annotations

import pandas as pd
import pytest

from simulator.data import PriceStore


def make_frame(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    """Rows of (date, open, high, low, close) -> OHLCV frame."""
    df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close"])
    df["Volume"] = 1000.0
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")


@pytest.fixture
def store() -> PriceStore:
    """Three trading days. ABC is USD-priced, XYZ.L is GBP-priced.

    GBPUSD is 1.25 then 1.20 then 1.25 (so $125 = £100 on day 1).
    """
    abc = make_frame(
        [
            ("2024-01-02", 100.0, 105.0, 95.0, 102.0),
            ("2024-01-03", 102.0, 110.0, 101.0, 108.0),
            ("2024-01-04", 108.0, 109.0, 90.0, 92.0),
        ]
    )
    xyz = make_frame(
        [
            ("2024-01-02", 10.0, 10.5, 9.5, 10.0),
            ("2024-01-03", 10.0, 11.0, 9.8, 11.0),
            ("2024-01-04", 11.0, 12.0, 10.5, 12.0),
        ]
    )
    spy = make_frame(
        [
            ("2024-01-02", 400.0, 402.0, 398.0, 400.0),
            ("2024-01-03", 400.0, 410.0, 399.0, 404.0),
            ("2024-01-04", 404.0, 405.0, 395.0, 398.0),
        ]
    )
    fx = pd.Series(
        [1.25, 1.20, 1.25],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    return PriceStore({"ABC": abc, "XYZ.L": xyz, "SPY": spy}, fx)
