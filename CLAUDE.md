# ETF Duel Foundation Model Project (Claude repo)

## Where to save artifacts (Google Drive)
All datasets, cached data, and outputs must be saved under:

- `/content/drive/MyDrive/ETF Duel Foundation Model Project/claude_build/`

Suggested subfolders (create if missing):
- `data/raw/`
- `data/processed/`
- `runs/`
- `models/`

Do not store large artifacts only on Colab local disk.

## Secrets (Colab)
Never commit secrets. Never print secrets.

Alpaca keys are stored in Colab Secrets:
- `PAPER_KEY`
- `PAPER_SEC`

Load ONLY in notebooks:
- `from google.colab import userdata`
- `key = userdata.get("PAPER_KEY")`
- `sec = userdata.get("PAPER_SEC")`

## Data source
Use `alpaca-py` (Alpaca Market Data API) for historical bars.
- 30-minute bars
- split-adjusted prices (`adjustment="split"`)
- regular trading hours (US/Eastern 09:30–16:00)

## Current scope (Experiment 0 only)
Implement only:
1) Download + cache bars
2) Compute ATR + triple-barrier labels (project rules)
3) Minimal baseline features
4) Sanity checks + leakage smoke test
5) Save dataset + summary JSON to Drive

## Triple-barrier label rules (must match exactly)
- Features at time `t` use bars up to and including bar `t` only.
- Entry = open of next bar (`t+1`).
- ATR at time `t` is computed causally using only bars ≤ `t`.
- Long trade:
  - TP = entry + k_up * ATR_t
  - SL = entry - k_dn * ATR_t
- Scan bars `t+1 .. t+N` inclusive.
- If TP and SL hit in the same bar, assume SL hits first.
- Label: +1 TP first, -1 SL first, 0 timeout.

## Colab notebooks
Notebooks in `notebooks/` must:
- mount Drive
- create the output folder
- run top-to-bottom without edits
- cache intermediate results to Drive (Parquet)

## Repo hygiene
Keep code under `src/`, use type hints + docstrings, and keep PRs focused.
