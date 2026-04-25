import os
import argparse
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.default import BaselineConfig
from src.path import OptionPath
from src.utils.experiment import ExperimentLogger
from src.models.cnn import CNN4
from src.trainers.trainer import Trainer, EarlyStopping
from src.data.dataset import IVSDataset
from src.data.transforms import get_ivs_transform
from src.backtester.backtest import BacktestEngine
from src.utils.seed import set_seed

def parse_args(config: BaselineConfig) -> BaselineConfig:
    parser = argparse.ArgumentParser(description="IVS CNN Training")
    parser.add_argument("--exp_group", type=str, default=None, help="實驗主資料夾名稱")
    parser.add_argument("--exp_name", type=str, default=None, help="實驗名稱 (子資料夾)")
    parser.add_argument("--ivs_transform", type=str, default=None, help="IVS 特徵轉換方式 (raw/log/clip)")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num_ensembles", type=int, default=None)

    args = parser.parse_args()

    if args.exp_group: config.exp_group = args.exp_group
    if args.exp_name: config.exp_name = args.exp_name
    if args.ivs_transform: config.ivs_transform = args.ivs_transform
    if args.learning_rate: config.learning_rate = args.learning_rate
    if args.epochs: config.epochs = args.epochs
    if args.num_ensembles: config.num_ensembles = args.num_ensembles

    return config

def prepare_datasets(config: BaselineConfig, train_start: int, train_end: int, val_start: int, val_end: int):
    """根據指定的年份範圍，建立 Dataset 並處理好 Transform"""
    train_dataset = IVSDataset(config.data_dir, start_year=train_start, end_year=train_end)

    transform_kwargs = {}
    if config.ivs_transform == 'zscore':
        transform_kwargs['mean'] = train_dataset.X.mean().item()
        transform_kwargs['std'] = train_dataset.X.std().item()
    elif config.ivs_transform == 'minmax':
        transform_kwargs['min_val'] = train_dataset.X.min().item()
        transform_kwargs['max_val'] = train_dataset.X.max().item()
    elif config.ivs_transform == 'clip':
        transform_kwargs['max_val'] = config.ivs_clip_max

    ivs_transform_func = get_ivs_transform(config.ivs_transform, **transform_kwargs)
    train_dataset.transform = ivs_transform_func
    val_dataset = IVSDataset(config.data_dir, start_year=val_start, end_year=val_end, transform=ivs_transform_func)

    return train_dataset, val_dataset

def main():
    config = BaselineConfig()
    config = parse_args(config)

    logger = ExperimentLogger(config)
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"開始實驗: {config.exp_name} | 群組: {config.exp_group} | Device: {device}")

    if config.training_strategy == 'standard':
        train_dataset, val_dataset = prepare_datasets(
            config,
            train_start=config.start_year,
            train_end=config.end_year,
            val_start=config.end_year + 1,
            val_end=2023
        )
    else:
        pass

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    all_predictions = []
    all_histories = []

    for i in range(config.num_ensembles):
        print(f"\n--- 訓練 Ensemble 模型 {i+1}/{config.num_ensembles} ---")
        start_time = time.time()

        model = CNN4()
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
        print(f"實驗完成，總耗時: {duration_minutes:.2f} 分鐘")

    logger.save_all_loss_histories(all_histories)
    ensemble_pred = np.mean(all_predictions, axis=0)

    df_preds = pd.DataFrame({
        'Date': test_dates,
        'Permno': test_permnos,
        'Pred': ensemble_pred,
        'Actual': actuals
    })
    logger.save_predictions(df_preds)

    print(f"實驗完成，結果已存入 {logger.exp_dir}")
    time.sleep(2)
    print("\n--- 開始執行回測模組 ---")
    try:
        engine = BacktestEngine(df_preds, config.base_fee_bps)
        backtest_results = engine.run_simulation()
        engine.calculate_metrics(backtest_results, save=True, save_path=os.path.join(logger.exp_dir, "backtest_metrics.txt"))
        engine.save_and_plot_performance(backtest_results, os.path.join(OptionPath.Benchmark, 'spy_benchmark_monthly.parquet'), logger.exp_dir)

    except FileNotFoundError as e:
        print(f"找不到回測所需的外部檔案，跳過回測。錯誤: {e}")
    except Exception as e:
        print(f"回測模組發生錯誤: {e}")

    print("\n 整個實驗流程全部完成！")

if __name__ == "__main__":
    main()