# Bundled sample data (SYNTHETIC)

> **These prices are synthetic, not real market data.** Yahoo Finance was
> unreachable from the build environment, so these files were made by
> `scripts/generate_sample_data.py` (seeded one-factor GBM model). They exist so
> the app and test suite work fully offline.

| File | Meaning | Currency |
|---|---|---|
| `AAPL.csv`, `MSFT.csv`, `GOOGL.csv`, `AMZN.csv`, `NVDA.csv` | US stocks (synthetic) | USD |
| `SPY.csv` | S&P 500 ETF benchmark (synthetic) | USD |
| `VOD.L.csv` | London-listed stock (synthetic) | GBP |
| `GBPUSD.csv` | US dollars per 1 GBP (synthetic) | – |

Date range: 2019-01-02 to 2024-12-31, business days only.

To use **real** data, run `python scripts/download_data.py`. It writes to
`data/downloaded/`, which takes precedence over this folder.
