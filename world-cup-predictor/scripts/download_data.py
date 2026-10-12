"""Download the raw datasets into data/raw/ and record their checksums.

Sources (see data/SOURCES.md):
* martj42/international_results (CC0): results.csv, shootouts.csv, former_names.csv
* Dato-Futbol/fifa-ranking: ranking_fifa_historical.csv (FIFA men's ranking 1992-2024)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MARTJ42 = "https://raw.githubusercontent.com/martj42/international_results/master/"
SOURCES = {
    "results.csv": MARTJ42 + "results.csv",
    "shootouts.csv": MARTJ42 + "shootouts.csv",
    "former_names.csv": MARTJ42 + "former_names.csv",
    "fifa_ranking.csv": (
        "https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/master/"
        "ranking_fifa_historical.csv"
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download raw football data.")
    parser.add_argument("--out", type=Path, default=RAW)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    lines = [f"# Downloaded {datetime.now(UTC):%Y-%m-%d %H:%M} UTC", ""]
    failed = 0
    for name, url in SOURCES.items():
        path = args.out / name
        try:
            urllib.request.urlretrieve(url, path)  # noqa: S310 - fixed https URLs
        except OSError as exc:
            print(f"[fail] {name}: {exc}", file=sys.stderr)
            failed += 1
            continue
        lines.append(f"{name}  sha256={sha256(path)}  {url}")
        print(f"[ok]   {name} ({path.stat().st_size:,} bytes)")
    (args.out / "CHECKSUMS.txt").write_text("\n".join(lines) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
