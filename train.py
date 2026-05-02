import os
import gc
import argparse
import typing
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.default import BaselineConfig
from src.path import OptionPath
from src.utils.experiment import ExperimentLogger
from src.models.cnn import CNN1, CNN4, CNN5
from src.trainers.trainer import Trainer, EarlyStopping
from src.data.dataset import IVSDataset
from src.data.transforms import get_ivs_transform
from src.backtester.backtest import BacktestEngine
from src.utils.seed import set_seed

def parse_args(config: BaselineConfig) -> BaselineConfig:
    parser = argparse.ArgumentParser(description="IVS CNN Training")
    parser.add_argument("--exp_group", type=str, default=None, help="實驗主資料夾名稱")
    parser.add_argument("--exp_name", type=str, default=None, help="實驗名稱 (子資料夾)")
    parser.add_argument("--model_type", type=str, default=None, help="模型類型 (CNN1/CNN4/CNN5)")
    parser.add_argument("--ivs_transform", type=str, default=None, help="IVS 特徵轉換方式 (raw/log/clip...)")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num_ensembles", type=int, default=None)
    parser.add_argument("--maxpool", type=bool, default=None)
    parser.add_argument("--training_strategy", type=str, default=None, help="訓練機制 (standard/expanding/rolling_finetune)")
    parser.add_argument("--step_months", type=int, default=None, help="時間窗推進的步長 (月數)")

    args = parser.parse_args()

    if args.exp_group: config.exp_group = args.exp_group
    if args.exp_name: config.exp_name = args.exp_name
    if args.model_type: config.model_type = args.model_type
    if args.ivs_transform: config.ivs_transform = args.ivs_transform
    if args.learning_rate: config.learning_rate = args.learning_rate
    if args.epochs: config.epochs = args.epochs
    if args.num_ensembles: config.num_ensembles = args.num_ensembles
    if args.maxpool: config.max_pool = False # default 是 True，若有傳入參數則改為 False
    if args.training_strategy: config.training_strategy = args.training_strategy
    if args.step_months: config.step_months = args.step_months
    return config

def prepare_datasets(config: BaselineConfig, train_start: int, train_end: int, val_start: int, val_end: int):
    """根據指定的年份範圍，建立 Dataset 並處理好 Transform"""
    train_dataset = IVSDataset(config.data_dir, start_year=train_start, end_year=train_end)
    X_flat = train_dataset.X.reshape(-1)
    indices = torch.randperm(X_flat.numel())[:1000000]
    sample_X = X_flat[indices]

    transform_kwargs = {}
    if config.ivs_transform == 'zscore':
        transform_kwargs['mean'] = train_dataset.X.mean().item()
        transform_kwargs['std'] = train_dataset.X.std().item()
    elif config.ivs_transform == 'minmax':
        transform_kwargs['min_val'] = train_dataset.X.min().item()
        transform_kwargs['max_val'] = torch.quantile(sample_X, 0.95).item()
    elif config.ivs_transform == 'clip':
        transform_kwargs['max_val'] = config.ivs_clip_max
    elif config.ivs_transform == 'rgb':
        transform_kwargs['min_val'] = train_dataset.X.min().item()
        transform_kwargs['max_val'] = torch.quantile(sample_X, 0.95).item()
        transform_kwargs['cmap_name'] = 'viridis'

    ivs_transform_func = get_ivs_transform(config.ivs_transform, **transform_kwargs)
    train_dataset.transform = ivs_transform_func
    val_dataset = IVSDataset(config.data_dir, start_year=val_start, end_year=val_end, transform=ivs_transform_func)

    return train_dataset, val_dataset

def get_transform_func(config: BaselineConfig, X_tensor: torch.Tensor):
    """根據輸入張量的統計資訊，構建 IVS Transform 函數。"""
    X_flat = X_tensor.reshape(-1)
    indices = torch.randperm(X_flat.numel())[:1000000]
    sample_X = X_flat[indices]

    transform_kwargs = {}
    if config.ivs_transform == 'zscore':
        transform_kwargs['mean'] = X_tensor.mean().item()
        transform_kwargs['std'] = X_tensor.std().item()
    elif config.ivs_transform == 'minmax':
        transform_kwargs['min_val'] = X_tensor.min().item()
        transform_kwargs['max_val'] = torch.quantile(sample_X, 0.95).item()
    elif config.ivs_transform == 'clip':
        transform_kwargs['max_val'] = config.ivs_clip_max
    elif config.ivs_transform == 'rgb':
        transform_kwargs['min_val'] = X_tensor.min().item()
        transform_kwargs['max_val'] = torch.quantile(sample_X, 0.95).item()
        transform_kwargs['cmap_name'] = 'viridis'

    return get_ivs_transform(config.ivs_transform, **transform_kwargs)

def get_model(model_name: str, input_channels: int, max_pool: bool):
    if model_name == "CNN1":
        return CNN1(input_channels, max_pool)
    elif model_name == "CNN4":
        return CNN4(input_channels, max_pool)
    elif model_name == "CNN5":
        return CNN5(input_channels, max_pool)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

def main():
    config = BaselineConfig()
    config = parse_args(config)

    logger = ExperimentLogger(config)
    set_seed(config.random_seed)
    init_seed = config.random_seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Start Experiment: {config.exp_name} | Group: {config.exp_group} | Device: {device} ---")

    if config.training_strategy == 'standard':
        train_dataset, val_dataset = prepare_datasets(
            config,
            train_start=config.start_year,
            train_end=config.standard_train_end_year,
            val_start=config.standard_train_end_year + 1,
            val_end=config.val_end_year
        )

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        sample_x, _, _, _ = train_dataset[0]
        input_channels = sample_x.shape[0]

        all_predictions = []
        all_histories = []

        for i in range(config.num_ensembles):
            print(f"\n--- Training Ensemble Model {i+1}/{config.num_ensembles} ---")
            start_time = time.time()
            current_seed = init_seed + i
            set_seed(current_seed)

            model = get_model(config.model_type, input_channels, config.max_pool)
            criterion = nn.BCEWithLogitsLoss() if config.task_type == "classification" else nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

            trainer = Trainer(model, optimizer, criterion, config.task_type, device)

            early_stopping = None
            if config.use_early_stopping:
                early_stopping = EarlyStopping(
                    patience=config.es_patience,
                    min_delta=config.es_min_delta
                )

            history = trainer.fit(train_loader, val_loader, epochs=config.epochs, early_stopping=early_stopping)
            all_histories.append(history)

            preds, actuals, test_dates, test_permnos = trainer.predict(val_loader)
            all_predictions.append(preds)
            end_time = time.time()
            duration_minutes = (end_time - start_time) / 60
            print(f"Finish experiment. Time: {duration_minutes:.2f} minutes")

            trainer = None # type: ignore
            optimizer = None
            criterion = None
            model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.save_all_loss_histories(all_histories)
        ensemble_pred = np.mean(all_predictions, axis=0)

        df_preds = pd.DataFrame({
            'Date': test_dates,
            'Permno': test_permnos,
            'Pred': ensemble_pred,
            'Actual': actuals
        })

    elif config.training_strategy in ['expanding', 'rolling_finetune']:
        from src.data.time_window import TimeWindowDatasetManager

        print(f"\n--- Initializing TimeWindowDatasetManager ---")
        manager = TimeWindowDatasetManager(
            data_dir=config.data_dir,
            start_year=config.start_year,
            val_end_year=config.val_end_year
        )

        train_start_date = pd.Timestamp(f"{config.start_year}-01-01")
        global_end_date = pd.Timestamp(f"{config.val_end_year}-12-31")
        warm_end_date = train_start_date + pd.DateOffset(years=config.warm_up_years) - pd.Timedelta(days=1)

        print(f"Train start date: {train_start_date}")
        print(f"Warm-up end date: {warm_end_date}")

        # 第一階段：Warm-up 初始化
        warm_train_subset = manager.get_split(train_start_date, warm_end_date)
        if len(warm_train_subset) == 0:
            raise ValueError("Warm-up dataset is empty. Check your start_year and warm_up_years.")

        manager.transform = get_transform_func(config, warm_train_subset.X)
        warm_loader = DataLoader(warm_train_subset, batch_size=config.batch_size, shuffle=True)

        sample_x, _, _, _ = warm_train_subset[0]
        input_channels = sample_x.shape[0]

        print(f"\n--- Warm-up Phase: Training {config.num_ensembles} models for {config.warm_up_epochs} epochs ---")
        start_time = time.time()
        model_states: typing.List[typing.Dict[str, typing.Any]] = []
        for i in range(config.num_ensembles):
            print(f"Initializing Model {i+1}/{config.num_ensembles}")
            current_seed = init_seed + i
            set_seed(current_seed)
            model = get_model(config.model_type, input_channels, config.max_pool)
            criterion = nn.BCEWithLogitsLoss() if config.task_type == "classification" else nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            trainer = Trainer(model, optimizer, criterion, config.task_type, device)
            trainer.fit(warm_loader, warm_loader, epochs=config.warm_up_epochs)

            model_states.append({'model': model, 'optimizer': optimizer, 'trainer': trainer})

        # 第二階段：時間窗推進 (Time Forward)
        print(f"\n--- Starting rolling window loop (step: {config.step_months} months) ---")
        all_window_dfs = []
        current_end = warm_end_date

        while current_end < global_end_date:
            test_start = current_end + pd.Timedelta(days=1)
            test_end = current_end + pd.DateOffset(months=config.step_months)

            if test_end > global_end_date:
                test_end = global_end_date

            print(f"\n--- Inference & Finetune window: {test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')} ---")

            # 1. 對未來週期 (T+1) 進行「嚴格隔絕」的 Inference
            test_subset = manager.get_split(test_start, test_end)
            if len(test_subset) == 0:
                current_end = test_end
                continue

            test_loader = DataLoader(test_subset, batch_size=config.batch_size, shuffle=False)

            window_preds_list = []
            window_actuals = None
            window_dates = None
            window_permnos = None

            for state in model_states:
                trainer: Trainer = state['trainer']
                preds, actuals, dates, permnos = trainer.predict(test_loader)
                window_preds_list.append(preds)
                if window_actuals is None:
                    window_actuals = actuals
                    window_dates = dates
                    window_permnos = permnos

            ensemble_pred = np.mean(window_preds_list, axis=0)
            all_window_dfs.append(pd.DataFrame({
                'Date': window_dates,
                'Permno': window_permnos,
                'Pred': ensemble_pred,
                'Actual': window_actuals
            }))

            # 2. 獲取 T+1 的 Ground Truth 後，微調權重 (Fine-tune)
            # 確保不會接觸到 test_end 以後的資料
            finetune_start = train_start_date if config.training_strategy == 'expanding' else test_start
            finetune_subset = manager.get_split(finetune_start, test_end)
            finetune_loader = DataLoader(finetune_subset, batch_size=config.batch_size, shuffle=True)

            print(f"Fine-tuning {len(model_states)} models from {finetune_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}")
            for state in model_states:
                trainer: Trainer = state['trainer']
                trainer.fit(finetune_loader, finetune_loader, epochs=config.transfer_epochs)

            current_end = test_end

        end_time = time.time()
        duration_minutes = (end_time - start_time) / 60
        print(f"Finish experiment. Time: {duration_minutes:.2f} minutes")

        df_preds = pd.concat(all_window_dfs, ignore_index=True)

        for state in model_states:
            state['model'] = None
            state['optimizer'] = None
            state['trainer'] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.save_predictions(df_preds)

    print(f"Finish experiment. Save Result to {logger.exp_dir}")
    time.sleep(2)
    print("\n--- Start Backtesting ---")
    try:
        engine = BacktestEngine(df_preds, config.base_fee_bps)
        backtest_results = engine.run_simulation()
        engine.save_holdings_report(logger.exp_dir)
        engine.calculate_metrics(backtest_results, save=True, save_path=os.path.join(logger.exp_dir, "backtest_metrics.txt"), rf_path=os.path.join(OptionPath.RFrate, "fama_french_rf_monthly.parquet"))
        engine.save_and_plot_performance(backtest_results, os.path.join(OptionPath.Benchmark, 'spy_benchmark_monthly.parquet'), logger.exp_dir)

    except FileNotFoundError as e:
        print(f"File for backtesting not found: {e}")
    except Exception as e:
        print(f"Backtesting error: {e}")

    print("\n Finish all processes.")

if __name__ == "__main__":
    main()
