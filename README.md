# ETF Dual Foundation Model Project

A systematic framework for evaluating time-series forecasting features in tabular classification models for ETF trading signals.

## Overview

This project implements a series of experiments to evaluate whether time-series foundation models (Chronos) can improve triple-barrier classification for ETF trading signals. The experiments progressively add feature groups to measure their incremental value.

## Experiments

| Experiment | Description | Features |
|------------|-------------|----------|
| Exp 0 | Pipeline sanity check | Data fetching, labeling, basic validation |
| Exp 1 | Tabular baseline | Causal features only (returns, volatility, ATR) |
| Exp 2 | + Forecast features | Add Chronos forecast-derived features |
| Exp 3 | + Context features | Add market context forecasts (SPY, QQQ) |
| Exp 4 | + Error features | Add forecast error monitoring features |
| Exp 5 | Ablations | Systematic comparison of configurations |

## Project Structure

```
ETF-Dual-Foundation-Project-CC-Version/
├── notebooks/
│   ├── 00_setup_colab.ipynb       # Environment setup
│   ├── 01_experiment0_pipeline.ipynb
│   ├── 02_experiment1_tabular_baseline.ipynb
│   ├── 03_experiment2_ts_features.ipynb
│   ├── 04_experiment3_context_features.ipynb
│   ├── 05_experiment4_error_features.ipynb
│   ├── 06_experiment5_ablations.ipynb
│   └── 07_experiment_summary.ipynb
├── src/etf_pipeline/
│   ├── data/
│   │   └── alpaca.py              # Alpaca API data fetching
│   ├── features/
│   │   ├── baseline.py            # Causal baseline features
│   │   ├── forecast_features.py   # TS forecast-derived features
│   │   ├── context_features.py    # Market context features
│   │   ├── forecast_error_features.py  # Error monitoring
│   │   └── registry.py            # Feature set management
│   ├── labels/
│   │   └── triple_barrier.py      # Triple-barrier labeling
│   ├── splits/
│   │   └── purged_walkforward.py  # Time-series splits with purging
│   ├── timeseries/
│   │   ├── dataset.py             # TS data preparation
│   │   ├── train.py               # TimeSeriesPredictor training
│   │   └── rolling_predict.py     # Causal rolling forecasts
│   ├── models/
│   │   └── tabular_baseline.py    # AutoGluon TabularPredictor
│   ├── metrics/
│   │   └── classification.py      # Evaluation metrics
│   ├── experiments/
│   │   ├── runner.py              # Experiment infrastructure
│   │   └── results.py             # Results aggregation
│   └── utils/
│       └── paths.py               # Path management
├── configs/
│   └── exp5_ablation.yaml         # Ablation configurations
├── tests/
│   ├── test_purged_walkforward.py
│   ├── test_features.py
│   └── test_experiments.py
└── CLAUDE.md                       # Project instructions
```

## Key Design Decisions

### Triple-Barrier Labels
- **Entry**: Open price of next bar (t+1)
- **ATR**: Computed causally using only bars ≤ t
- **TP/SL**: entry ± k × ATR (k_up=2.0, k_dn=1.0)
- **Vertical barrier**: N=26 bars (1 trading day of 30-min bars)
- **Label**: +1 (TP first), -1 (SL first), 0 (timeout)

### Time-Series Splits
- **Purging**: Remove training samples within N bars of test start
- **Embargo**: Additional gap after purge (N bars)
- **No leakage**: Validated by unit tests

### Rolling Forecasts
- For each decision timestamp t, forecasts use only history ≤ t
- Forecasts are generated incrementally to prevent lookahead bias
- Cached to Google Drive for reproducibility

## Feature Sets

### Small (default)
- Horizons: [1, 26]
- Features per horizon: mu, unc, trend, pos_in_interval
- Total: 8 forecast features

### Medium
- Horizons: [1, 4, 13, 26]
- Features per horizon: mu, unc, trend, pos_in_interval
- Total: 16 forecast features

### Large
- Horizons: [1, 2, 4, 8, 13, 20, 26]
- Features per horizon: mu, sigma, unc, trend, pos_in_interval, skew
- Total: 42 forecast features

## Running Experiments

### Prerequisites
1. Google Colab with A100 GPU (recommended)
2. Alpaca API credentials stored in Colab Secrets
3. Google Drive mounted

### Quick Start
1. Open `notebooks/00_setup_colab.ipynb` and run to install dependencies
2. Run notebooks in order: 01, 02, 03, 04, 05, 06
3. Run `07_experiment_summary.ipynb` to compile results

### Individual Experiment
```python
# In any notebook after setup
from etf_pipeline.experiments.runner import ExperimentConfig, ExperimentRunner

config = ExperimentConfig(
    experiment_name="exp2",
    symbols=["SPY"],
    feature_set="small",
    time_limit_sec=1200,
)

runner = ExperimentRunner(config)
runner.ensure_directories()
```

## Artifact Storage

All artifacts are saved to Google Drive:
```
/content/drive/MyDrive/ETF Duel Foundation Model Project/claude_build/
├── data/raw/           # Raw bar data
├── data/processed/     # Labeled datasets, forecasts
├── models/             # Trained models
│   ├── ts/             # TimeSeriesPredictor models
│   ├── exp1/           # Experiment 1 TabularPredictor models
│   ├── exp2/           # ...
│   └── ...
└── runs/               # Experiment runs (predictions, metrics)
    ├── exp1_SPY_20240101_120000/
    │   ├── config.yaml
    │   ├── predictions_SPY.parquet
    │   └── metrics_SPY.json
    └── ...
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_features.py -v

# Run with coverage
pytest tests/ --cov=src/etf_pipeline
```

## Configuration

### Environment Variables (Colab Secrets)
- `PAPER_KEY`: Alpaca API key
- `PAPER_SEC`: Alpaca API secret

### Key Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_rows_per_symbol` | 6500 | Maximum rows per symbol |
| `vertical_barrier_bars` | 26 | Label horizon (N bars) |
| `embargo_bars` | 26 | Embargo gap after purge |
| `time_limit_sec` | 1200 | AutoGluon training time |
| `ts_presets` | chronos_small | Chronos model size |

## Results Comparison

After running experiments, use the summary notebook to:
1. Compare balanced accuracy across experiments
2. Compute feature ablation impact
3. Generate publication-ready tables
4. Export results to CSV/JSON

## Citation

If you use this code, please cite:
```
@software{etf_dual_foundation,
  title = {ETF Dual Foundation Model Project},
  year = {2024},
  url = {https://github.com/mh122333/ETF-Dual-Foundation-Project-CC-Version}
}
```

## License

MIT License - see LICENSE file for details.
