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
from src.data.transforms import get_ivs_transform, get_target_transform
from src.backtester.backtest import BacktestEngine
from src.utils.seed import set_seed

pd.set_option('future.no_silent_downcasting', True)

def parse_args(config: BaselineConfig) -> BaselineConfig:
    parser = argparse.ArgumentParser(description="IVS CNN Training")
    parser.add_argument("--dataset_type", type=str, default=None, help="資料集類型 (USA/USA_ALL)")
    parser.add_argument("--start_year", type=int, default=None, help="訓練資料的起始年份")
    parser.add_argument("--standard_train_end_year", type=int, default=None, help="standard 訓練策略中訓練資料的結束年份")
    parser.add_argument("--exp_group", type=str, default=None, help="實驗主資料夾名稱")
    parser.add_argument("--exp_name", type=str, default=None, help="實驗名稱 (子資料夾)")
    parser.add_argument("--task_type", type=str, default=None, help="任務類型 (regression/classification)")
    parser.add_argument("--jump_threshold", type=float, default=None, help="classification 任務中 jump 的定義閾值")
    parser.add_argument("--model_type", type=str, default=None, help="模型類型 (CNN1/CNN4/CNN5)")
    parser.add_argument("--padding", type=int, default=None, help="CNN padding 大小 (例如 1, 2)")
    parser.add_argument("--ivs_transform", type=str, default=None, help="IVS 特徵轉換方式 (raw/log/clip...)")
    parser.add_argument("--target_transform", type=str, default=None, help="Label (Target) 轉換方式 (raw/signed_log/arcsinh/winsorize)")
    parser.add_argument("--early_stopping", action="store_true", help="是否啟用 Early Stopping")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--dropout_rate", type=float, default=None)
    parser.add_argument("--l1_lambda", type=float, default=None)
    parser.add_argument("--l2_lambda", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--warm_up_epochs", type=int, default=None)
    parser.add_argument("--transfer_epochs", type=int, default=None)
    parser.add_argument("--num_ensembles", type=int, default=None)
    parser.add_argument("--training_strategy", type=str, default=None, help="訓練機制 (standard/expanding/rolling_finetune)")
    parser.add_argument("--new_optimizer", action="store_true", help="在擴展窗口訓練時中是否每次時間窗推進都重置優化器")
    parser.add_argument("--step_months", type=int, default=None, help="時間窗推進的步長 (月數)")
    parser.add_argument("--lookback_months", type=int, default=None, help="針對 rolling_finetune, 往回看的歷史資料長度 (月數), 0 表示僅使用新資料")
    parser.add_argument("--shrcd", type=int, nargs="+", default=None, help="篩選 Share Code (例如 --shrcd 10 11)")
    parser.add_argument("--exchcd", type=int, nargs="+", default=None, help="篩選 Exchange Code (例如 --exchcd 1 2 3)")
    parser.add_argument("--return_outlier_quantile", type=float, default=None, help="篩選極端報酬分位數 (對 future_return 做去除，如 0.01)")
    parser.add_argument("--prc_limit", type=float, default=None, help="篩選股票價格下限 (例如 --prc_limit 5.0)")
    parser.add_argument("--reverse_block", action="store_true", help="是否將 CNN 區塊內的卷積層順序反轉")

    args = parser.parse_args()

    if args.exp_group is not None: config.exp_group = args.exp_group
    if args.exp_name is not None: config.exp_name = args.exp_name
    if args.model_type is not None: config.model_type = args.model_type
    if args.padding is not None: config.padding = args.padding
    if args.ivs_transform is not None: config.ivs_transform = args.ivs_transform
    if args.target_transform is not None: config.target_transform = args.target_transform
    if args.learning_rate is not None: config.learning_rate = args.learning_rate
    if args.dropout_rate is not None: config.dropout_rate = args.dropout_rate
    if args.l1_lambda is not None: config.l1_lambda = args.l1_lambda
    if args.l2_lambda is not None: config.l2_lambda = args.l2_lambda
    if args.epochs is not None: config.epochs = args.epochs
    if args.num_ensembles is not None: config.num_ensembles = args.num_ensembles
    if args.training_strategy is not None: config.training_strategy = args.training_strategy
    if args.step_months is not None: config.step_months = args.step_months
    if args.lookback_months is not None: config.rolling_lookback_months = args.lookback_months
    if args.task_type is not None: config.task_type = args.task_type
    if args.jump_threshold is not None: config.jump_threshold = args.jump_threshold
    if args.start_year is not None: config.start_year = args.start_year
    if args.standard_train_end_year is not None: config.standard_train_end_year = args.standard_train_end_year
    if args.dataset_type is not None:
        config.dataset_type = args.dataset_type
        config.__post_init__() # re-initialize data_dir based on dataset_type
    if args.shrcd is not None: config.shrcd = args.shrcd
    if args.exchcd is not None: config.exchcd = args.exchcd
    if args.return_outlier_quantile is not None: config.return_outlier_quantile = args.return_outlier_quantile
    if args.early_stopping is not None: config.use_early_stopping = args.early_stopping
    if args.prc_limit is not None: config.prc_limit = args.prc_limit
    if args.new_optimizer is not None: config.new_optimizer = args.new_optimizer
    if args.reverse_block is not None: config.block_reverse = args.reverse_block
    if config.task_type == "classification":
        print(f"Task is classification. Forcing target_transform to 'raw' to avoid interfering with 0/1 labeling.")
        config.target_transform = 'raw'
    if args.warm_up_epochs is not None: config.warm_up_epochs = args.warm_up_epochs
    if args.transfer_epochs is not None: config.transfer_epochs = args.transfer_epochs

    return config

def truncate_dataset_before_month(dataset: IVSDataset, cutoff: pd.Timestamp):
    dates = pd.to_datetime(pd.Series(dataset.dates))
    mask = dates < cutoff
    idx = np.where(mask.to_numpy())[0]

    dataset.X = dataset.X[idx]
    dataset.y = dataset.y[idx]
    dataset.dates = dataset.dates[idx]
    dataset.permnos = dataset.permnos[idx]
    return dataset

def prepare_datasets(config: BaselineConfig, train_start: int, train_end: int, val_start: int, val_end: int):
    """根據指定的年份範圍，建立 Dataset 並處理好 Transform"""
    target_transform_func = get_target_transform(config.target_transform)
    train_dataset = IVSDataset(
        config.data_dir,
        start_year=train_start,
        end_year=train_end,
        target_transform=target_transform_func,
        shrcd=config.shrcd,
        exchcd=config.exchcd,
        return_outlier_quantile=config.return_outlier_quantile,
        prc_limit=config.prc_limit
    )

    train_cutoff = pd.Timestamp(f"{train_end}-12-01")
    train_dataset = truncate_dataset_before_month(train_dataset, train_cutoff)

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty after trimming the last month.")

    X_flat = train_dataset.X.reshape(-1)
    indices = torch.randperm(X_flat.numel())[:1000000]
    sample_X = X_flat[indices]

    transform_kwargs = {}
    if config.ivs_transform == 'zscore':
        transform_kwargs['mean'] = train_dataset.X.mean().item()
        transform_kwargs['std'] = train_dataset.X.std().item()
    elif config.ivs_transform == 'minmax':
        transform_kwargs['min_val'] = train_dataset.X.min().item()
        transform_kwargs['max_val'] = torch.quantile(sample_X, 0.99).item()
    elif config.ivs_transform == 'clip':
        transform_kwargs['max_val'] = config.ivs_clip_max
    elif config.ivs_transform == 'rgb':
        transform_kwargs['min_val'] = train_dataset.X.min().item()
        transform_kwargs['max_val'] = torch.quantile(sample_X, 0.99).item()
        transform_kwargs['cmap_name'] = 'viridis'
    elif config.ivs_transform == 'divide_by_spy':
        spy_by_month = load_spy_ivs_by_month(os.path.join(OptionPath.SPY_IVS, f'spy_ivs.parquet'))
        transform_kwargs['spy_by_month'] = spy_by_month

    ivs_transform_func = get_ivs_transform(config.ivs_transform, **transform_kwargs)
    train_dataset.set_transform(ivs_transform_func)
    val_dataset = IVSDataset(
        config.data_dir,
        start_year=val_start,
        end_year=val_end,
        target_transform=target_transform_func,
        shrcd=config.shrcd,
        exchcd=config.exchcd,
        return_outlier_quantile=config.return_outlier_quantile,
        prc_limit=config.prc_limit
    )
    val_dataset.set_transform(ivs_transform_func)

    return train_dataset, val_dataset

def load_spy_ivs_by_month(spy_path: str) -> typing.Dict[pd.Period, torch.Tensor]:
    spy_df = pd.read_parquet(spy_path).copy()
    if spy_df.empty:
        raise ValueError(f"SPY IVS file is empty: {spy_path}")

    spy_df["opt_date"] = pd.to_datetime(spy_df["opt_date"])
    spy_df["days"] = spy_df["days"].astype(int)
    spy_df["delta"] = spy_df["delta"].round().astype(int)
    spy_df["month"] = spy_df["opt_date"].dt.to_period("M")

    days_order = sorted(spy_df["days"].unique())
    all_deltas = list(spy_df["delta"].unique())
    put_deltas = sorted([d for d in all_deltas if d < 0 and d >= -50])
    call_deltas = sorted([d for d in all_deltas if d > 0 and d <= 50])
    col_order = call_deltas + put_deltas

    spy_by_month: typing.Dict[pd.Period, torch.Tensor] = {}
    last_valid_surface = None
    for month, month_df in spy_df.groupby("month", sort=True):
        month_surface = month_df.pivot_table(
            index="days",
            columns="delta",
            values="impl_volatility",
            aggfunc="last",
        ).reindex(index=days_order, columns=col_order)

        if month_surface.isna().all().all():

            if last_valid_surface is None:
                raise ValueError(f"No previous SPY IVS for {month}")

            month_surface = last_valid_surface.copy()

        elif month_surface.isna().any().any():
            missing = int(month_surface.isna().sum().sum())
            raise ValueError(f"SPY IVS month {month} has {missing} missing cells")

        last_valid_surface = month_surface
        spy_by_month[month] = torch.tensor(month_surface.to_numpy(dtype=np.float32), dtype=torch.float32).unsqueeze(0) # type: ignore

    return spy_by_month

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
        transform_kwargs['max_val'] = torch.quantile(sample_X, 0.99).item()
    elif config.ivs_transform == 'clip':
        transform_kwargs['max_val'] = config.ivs_clip_max
    elif config.ivs_transform == 'rgb':
        transform_kwargs['min_val'] = X_tensor.min().item()
        transform_kwargs['max_val'] = torch.quantile(sample_X, 0.99).item()
        transform_kwargs['cmap_name'] = 'viridis'
    elif config.ivs_transform == 'divide_by_spy':
        spy_by_month = load_spy_ivs_by_month(os.path.join(OptionPath.SPY_IVS, f'spy_ivs.parquet'))
        transform_kwargs['spy_by_month'] = spy_by_month

    return get_ivs_transform(config.ivs_transform, **transform_kwargs)

def get_model(model_name: str, input_channels: int, padding: int, dropout_rate: float, reverse_block: bool) -> nn.Module:
    if model_name == "CNN1":
        return CNN1(input_channels, dropout_rate, padding=padding, reverse_block=reverse_block)
    elif model_name == "CNN4":
        return CNN4(input_channels, dropout_rate, padding=padding, reverse_block=reverse_block)
    elif model_name == "CNN5":
        return CNN5(input_channels, dropout_rate, padding=padding, reverse_block=reverse_block)
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
        sample_x, _, _, _, _ = train_dataset[0]
        input_channels = sample_x.shape[0]

        all_predictions = []
        all_histories = []

        for i in range(config.num_ensembles):
            current_seed = init_seed + i
            set_seed(current_seed)
            print(f"\n--- Training Ensemble Model {i+1}/{config.num_ensembles}. Seed: {current_seed} ---")
            start_time = time.time()

            model = get_model(config.model_type, input_channels, config.padding, config.dropout_rate, reverse_block=config.block_reverse)
            if config.task_type == "classification":
                num_positives = (train_dataset.y == 1).sum().item()
                num_negatives = (train_dataset.y == 0).sum().item()
                pos_weight = torch.tensor([np.sqrt(num_negatives / num_positives)], dtype=torch.float32).to(device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2_lambda)

            trainer = Trainer(model, optimizer, criterion, config.task_type, device, config.jump_threshold, l1_lambda=config.l1_lambda)

            early_stopping = None
            if config.use_early_stopping:
                early_stopping = EarlyStopping(
                    patience=config.es_patience,
                    min_delta=config.es_min_delta
                )

            history = trainer.fit(train_loader, val_loader, epochs=config.epochs, early_stopping=early_stopping)
            all_histories.append(history)

            preds, actuals, test_dates, test_permnos, raw_actuals = trainer.predict(val_loader)
            all_predictions.append(preds)
            end_time = time.time()
            duration_minutes = (end_time - start_time) / 60
            print(f"Finish experiment. Time: {duration_minutes:.2f} minutes")

            trainer = None # type: ignore
            optimizer = None
            criterion = None
            model = None # type: ignore
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.save_all_loss_histories(all_histories)
        ensemble_pred = np.mean(all_predictions, axis=0)

        df_preds = pd.DataFrame({
            'Date': test_dates,
            'Permno': test_permnos,
            'Pred': ensemble_pred,
            'Actual': actuals,
            'ActualRaw': raw_actuals
        })

    elif config.training_strategy in ['expanding', 'rolling_finetune']:
        from src.data.time_window import TimeWindowDatasetManager

        print(f"\n--- Initializing TimeWindowDatasetManager ---")
        target_transform_func = get_target_transform(config.target_transform)
        manager = TimeWindowDatasetManager(
            data_dir=config.data_dir,
            start_year=config.start_year,
            val_end_year=config.val_end_year,
            target_transform=target_transform_func,
            shrcd=config.shrcd,
            exchcd=config.exchcd,
            return_outlier_quantile=config.return_outlier_quantile
        )

        train_start_date = pd.Timestamp(f"{config.start_year}-01-01")
        global_end_date = pd.Timestamp(f"{config.val_end_year}-12-31")
        warm_end_date = train_start_date + pd.DateOffset(years=config.warm_up_years) - pd.Timedelta(days=1)
        warm_train_end = warm_end_date - pd.offsets.MonthEnd(1)

        print(f"Train start date: {train_start_date}")
        print(f"Warm-up end date: {warm_end_date}")
        print(f"Warm-up train end date: {warm_train_end}")

        # Use the raw warm-up subset to fit the transform and cache it in the manager, then get the transformed warm-up subset for training
        raw_warm_up_subset = manager.get_split(train_start_date, warm_train_end)
        if len(raw_warm_up_subset) == 0:
            raise ValueError("Warm-up dataset is empty. Check your start_year and warm_up_years.")
        manager.transform = get_transform_func(config, raw_warm_up_subset.X)

        # Get the warm-up subset again with the transform applied
        warm_train_subset = manager.get_split(train_start_date, warm_train_end)
        warm_loader = DataLoader(warm_train_subset, batch_size=config.batch_size, shuffle=True)

        sample_x, _, _, _, _ = warm_train_subset[0]
        input_channels = sample_x.shape[0]

        print(f"\n--- Warm-up Phase: Training {config.num_ensembles} models for {config.warm_up_epochs} epochs ---")
        start_time = time.time()
        model_states: typing.List[typing.Dict[str, typing.Any]] = []
        for i in range(config.num_ensembles):
            print(f"Initializing Model {i+1}/{config.num_ensembles}")
            current_seed = init_seed + i
            set_seed(current_seed)
            model = get_model(config.model_type, input_channels, config.padding, config.dropout_rate, reverse_block=config.block_reverse)
            if config.task_type == "classification":
                num_positives = (warm_train_subset.y == 1).sum().item()
                num_negatives = (warm_train_subset.y == 0).sum().item()
                pos_weight = torch.tensor([np.sqrt(num_negatives / num_positives)], dtype=torch.float32).to(device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                criterion = nn.MSELoss()
            if config.training_strategy == 'rolling_finetune':
                optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.6, 0.999), weight_decay=config.l2_lambda)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2_lambda)
            trainer = Trainer(model, optimizer, criterion, config.task_type, device, config.jump_threshold, l1_lambda=config.l1_lambda)
            trainer.fit(warm_loader, warm_loader, epochs=config.warm_up_epochs) # We don't need to early stop during warm-up, so we can pass the same loader for train and val

            model_states.append({'model': model, 'optimizer': optimizer, 'trainer': trainer})

        # 第二階段：時間窗推進 (Time Forward)
        print(f"\n--- Starting rolling window loop (step: {config.step_months} months) ---")
        all_window_dfs = []
        current_end = warm_end_date

        while current_end < global_end_date:
            test_start = current_end + pd.Timedelta(days=1)
            test_end = current_end + pd.offsets.MonthEnd(config.step_months)

            if test_end > global_end_date:
                test_end = global_end_date

            print(f"\n--- Inference window: {test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')} ---")

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
                preds, actuals, dates, permnos, raw_actuals = trainer.predict(test_loader)
                window_preds_list.append(preds)
                if window_actuals is None:
                    window_actuals = actuals
                    window_raw_actuals = raw_actuals
                    window_dates = dates
                    window_permnos = permnos
                else:
                    assert window_dates is not None
                    assert window_permnos is not None
                    assert np.array_equal(actuals, window_actuals)
                    assert np.array_equal(dates, window_dates)
                    assert np.array_equal(permnos, window_permnos)

            ensemble_pred = np.mean(window_preds_list, axis=0)

            all_window_dfs.append(pd.DataFrame({
                'Date': window_dates,
                'Permno': window_permnos,
                'Pred': ensemble_pred,
                'Actual': window_actuals,
                'ActualRaw': window_raw_actuals
            }))

            # 微調權重 (Fine-tune)
            # 需要留緩衝，ex: 使用一月底資料預測二月報酬後，由於二月報酬在二月底才會知道
            # 因此一月底預測完不能馬上將該月資料加入訓練，必須等到二月底才將一月資料加入訓練
            safe_finetune_end = test_start - pd.Timedelta(days=1)

            if config.training_strategy == 'expanding':
                finetune_start = train_start_date # start from beginning for expanding window
            else: # rolling_finetune
                if config.rolling_lookback_months > 0: # fine tune with a recent lookback months
                    finetune_start = safe_finetune_end - pd.offsets.MonthEnd(config.rolling_lookback_months) + pd.Timedelta(days=1)
                    if finetune_start < train_start_date:
                        finetune_start = train_start_date
                else:
                    finetune_start = safe_finetune_end - pd.offsets.MonthEnd(config.step_months) + pd.Timedelta(days=1) # only fine tune with the new month data

            if safe_finetune_end <= warm_end_date:
                # 第一圈：尚未有新的可用歷史，跳過更新
                current_end = test_end
                continue

            finetune_subset = manager.get_split(finetune_start, safe_finetune_end)
            finetune_loader = DataLoader(finetune_subset, batch_size=config.batch_size, shuffle=True)

            print(f"Fine-tuning {len(model_states)} models from {finetune_start.strftime('%Y-%m')} to {safe_finetune_end.strftime('%Y-%m')}")
            for state in model_states:
                trainer: Trainer = state['trainer']
                model: nn.Module = state['model']
                if config.new_optimizer:
                    new_optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2_lambda)
                    trainer.optimizer = new_optimizer
                    state['optimizer'] = new_optimizer
                trainer.fit(finetune_loader, finetune_loader, epochs=config.transfer_epochs) # Don't need early stopping during fine-tuning too

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
        stock_info_dir = OptionPath.StockInfo_All if config.dataset_type == "USA_IVS_ALL" else OptionPath.StockInfo
        engine = BacktestEngine(df_preds, stock_info_dir, config.base_fee_bps, task_type=config.task_type, jump_threshold=config.jump_threshold)
        backtest_results = engine.run_simulation()
        engine.save_holdings_report(logger.exp_dir)
        engine.calculate_metrics(backtest_results, save=True, save_path=os.path.join(logger.exp_dir, "backtest_metrics.txt"), rf_path=os.path.join(OptionPath.RFrate, "fama_french_rf_monthly.parquet"))
        engine.save_and_plot_performance(backtest_results, os.path.join(OptionPath.Benchmark, 'spy_benchmark_monthly.parquet'), logger.exp_dir)
        engine.save_decile_analysis(df_preds, logger.exp_dir)

    except FileNotFoundError as e:
        print(f"File for backtesting not found: {e}")
    except Exception as e:
        print(f"Backtesting error: {e}")

    print("\n Finish all processes.")

if __name__ == "__main__":
    main()
