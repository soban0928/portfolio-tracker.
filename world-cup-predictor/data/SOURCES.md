# Data sources

| File | Source | Licence | Notes |
|---|---|---|---|
| `raw/results.csv` | [martj42/international_results](https://github.com/martj42/international_results) (also on Kaggle as *International football results from 1872 to 2026*) | CC0 1.0 | Men's full internationals. Scores include extra time, exclude shoot-outs. |
| `raw/shootouts.csv` | same | CC0 1.0 | Penalty shoot-out winners. |
| `raw/former_names.csv` | same | CC0 1.0 | Historical team names (none currently appear in `results.csv`). |
| `raw/fifa_ranking.csv` | [Dato-Futbol/fifa-ranking](https://github.com/Dato-Futbol/fifa-ranking) | see repo | FIFA men's ranking, Dec 1992 – Sep 2024 (scraped from fifa.com). Ends ~21 months before the 2026 World Cup. |

`raw/CHECKSUMS.txt` records the SHA-256 of each file and when it was downloaded.
The snapshot is committed so results are reproducible; refresh it with
`make data` (results will change as the upstream data is updated or corrected).

No results were typed in by hand: every match, score and shoot-out comes from
these files.
