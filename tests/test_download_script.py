import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_normalise_flattens_multiindex():
    mod = load_script("download_data")
    idx = pd.DatetimeIndex(["2024-01-02"], name="Date")
    cols = pd.MultiIndex.from_product([["Close", "High", "Low", "Open", "Volume"], ["AAPL"]])
    raw = pd.DataFrame([[1.0, 2.0, 0.5, 1.1, 100]], index=idx, columns=cols)
    out = mod.normalise(raw)
    assert list(out.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
    assert out.iloc[0]["Date"] == "2024-01-02"


def test_main_reports_failures(tmp_path, monkeypatch):
    mod = load_script("download_data")

    def fake_download(symbol, start, end):
        if symbol == "BAD":
            raise RuntimeError("boom")
        return pd.DataFrame(
            {
                "Date": ["2024-01-02"],
                "Open": [1],
                "High": [1],
                "Low": [1],
                "Close": [1],
                "Volume": [0],
            }
        )

    monkeypatch.setattr(mod, "download", fake_download)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    assert mod.main(["--tickers", "AAA", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "AAA.csv").exists() and (tmp_path / "GBPUSD.csv").exists()
    assert mod.main(["--tickers", "BAD", "--out", str(tmp_path)]) == 1
