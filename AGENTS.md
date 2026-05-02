# Agent Instructions

## Overview
This repository contains a PyTorch-based project (USA-IVS-CNN) for predicting stock returns based on Implied Volatility Surfaces (IVS) using Convolutional Neural Networks (CNNs). The primary task is regression (predicting next-month stock returns from 10×18 IVS grids).

## Run & Build
- **Setup**: `pip install -r requirements.txt && pip install -e .`
- **Download Data**: `python download_ivs.py` (requires `.env` with `WRDS_USERNAME`). See [README](README.md).
- **Run Training**: `python train.py --exp_group <group> --exp_name <name> --ivs_transform <type>`
- **Batch Experiments**: Use `bash run.sh` to execute batch experiments.

## Code Structure
- **Models**: [src/models/](src/models/) - Contains CNN architectures (`CNN1`, `CNN4`, `CNN5`) predicting from inputs of shape `[Batch, 1, 10, 18]`.
- **Data & Transforms**: [src/data/](src/data/) - `IVSDataset` loads parquet files. Transforms include `raw`, `log`, `clip`, `zscore`, `minmax`, `rgb`.
- **Training**: [src/trainers/](src/trainers/) - Handles train/eval loops, early stopping, and ensembles.
- **Backtesting**: [src/backtester/](src/backtester/) - Evaluates predictions on long-short portfolios (top/bottom 10%) including transaction costs and short fees.
- **Configurations**: [configs/default.py](configs/default.py) contains training and model hyperparameters.
- **Interactive Workflows**: Check the [notebooks/](notebooks/) directory for data visualization, model testing, and backtest evaluations.

## Documentation Links
Before modifying features, refer to the relevant documentation:
- [Backtesting Logic & Costs](docs/backtest.md)
- [SQL Data Extraction Queries](docs/sql_query_explanation.md)
- [IVS Data Status](docs/ivs_data_status.md)
- [Reference Paper](docs/paper.md)

## Conventions
- **Outputs**: All results are saved in `DB/Results/CNN/<exp_group>/<exp_name>/`. Output artifacts typically include `config.json`, `loss_history_all.csv`, `ensemble_predictions.csv`, and `backtest_timeseries.csv`.
- **Ensemble Learning**: By default, standard training executes an ensemble block averaging 40 model predictions.
