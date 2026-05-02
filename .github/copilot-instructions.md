# Copilot instructions for USA-IVS-CNN

## Build, test, and lint commands

```bash
# environment + package setup
pip install -r requirements.txt
pip install -e .

# data extraction (requires .env with WRDS_USERNAME)
python download_ivs.py

# single experiment run
python train.py --exp_group <group> --exp_name <name> --ivs_transform <raw|log|clip|zscore|minmax|rgb>

# batch experiments defined in run.sh
bash run.sh
```

Current repository state:
- There is no committed automated test suite yet (`tests/` and CI workflow are absent).
- `pytest` is included in dependencies; when tests are added, use:
  - Full suite: `pytest`
  - Single test: `pytest path/to/test_file.py::test_name`
- There is no repo-defined lint command/config yet.

## High-level architecture

### 1. Data ingestion and storage
- `download_ivs.py` is the entry point for WRDS pulls.
- `src/wrds_client.py` builds yearly SQL against OptionMetrics/CRSP, writes yearly parquet files (`option_ivs_crsp_<year>.parquet`), and also exports:
  - `DB/OptionDB/SPY_Benchmark/spy_benchmark_monthly.parquet`
  - `DB/OptionDB/RF_Rate/fama_french_rf_monthly.parquet`
- `src/path.py` centralizes all data/result paths and creates directories on import.

### 2. Dataset shaping and IVS transforms
- `src/data/dataset.py` builds model samples from yearly parquet files:
  - keeps only complete IVS grids (`days × delta` fully populated),
  - keeps month-end observations per `(permno, month)`,
  - aligns labels as next-month return (`future_return`),
  - returns `(x, y, date, permno)` where `x` is CNN-ready tensor.
- `src/data/transforms.py` contains transform factory (`raw`, `log`, `clip`, `zscore`, `minmax`, `rgb`).

### 3. Model training and ensembling
- `train.py` orchestrates training from `BaselineConfig` (`configs/default.py`) and CLI overrides.
- Default path is `training_strategy == "standard"` (expanding strategy is not wired yet in `train.py`).
- The training loop builds an ensemble (`num_ensembles=40` by default), trains `CNN4` models, and averages predictions.
- `src/trainers/trainer.py` handles epoch loop, eval, early stopping, and prediction output collection.
- `src/models/cnn.py` defines `CNN1`, `CNN4`, `CNN5` built from shared `CNNBlock`.

### 4. Result logging and backtest
- `src/utils/experiment.py` writes timestamped experiment folders under:
  - `DB/Results/CNN/<exp_group>/<exp_name>_<timestamp>/`
- Main artifacts are `config.json`, `loss_history_all.csv`, `ensemble_predictions.csv`, plus backtest outputs.
- `src/backtester/backtest.py` runs monthly long-short simulation (top/bottom decile by prediction), applies transaction + shorting costs, and saves:
  - `holdings_detail.csv`
  - `backtest_metrics.txt`
  - `backtest_timeseries.csv`
  - `cumulative_performance.png`

## Key repository conventions

- **Canonical paths come from `src/path.py`** (`OptionPath`, `ResultsPath`); avoid hardcoded DB paths in new code.
- **Input geometry convention**: IVS tensors are treated as image-like grids (`[batch, channels, 10, 18]` for default single-channel setup).
- **Transform statistics are train-split derived** in `train.py` (`zscore`/`minmax`/`rgb` params) and then reused for validation.
- **Backtest assumes external market metadata file**: `DB/OptionDB/Stock_Info/market_metadata.parquet` must exist with `Date`, `Permno`, and microcap-related fields.
- **CLI flag caveat**: in current `train.py`, passing `--maxpool` flips `config.max_pool` to `False`.
- **Output naming is stable and used downstream** (especially `ensemble_predictions.csv` and `backtest_timeseries.csv`), so preserve these filenames unless intentionally migrating the pipeline.
- Relevant domain docs are in:
  - `docs/backtest.md`
  - `docs/sql_query_explanation.md`
  - `docs/ivs_data_status.md`
  - `docs/paper.md`
