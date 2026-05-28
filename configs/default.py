import os
import typing
from src.path import OptionPath, ResultsPath
from dataclasses import dataclass, field

@dataclass
class BaselineConfig:
    project_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_type: str = "USA_ALL" # "USA" or "USA_ALL"
    data_dir: str = field(init=False)
    result_dir: str = os.path.join(ResultsPath.CNN)

    def __post_init__(self):
        if self.dataset_type == "USA":
            self.data_dir = OptionPath.IVS
        elif self.dataset_type == "USA_ALL":
            self.data_dir = OptionPath.IVS_ALL
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")

    exp_group: str = ''
    exp_name: str = "CNN1_Baseline"
    task_type: str = "regression" # or 'classification'

    # 若為 classification，在此設定跳躍的定義閾值 (如預測下個月回報 <= -15%)
    jump_threshold: float = -0.15

    # ---------------------------------------------------------
    # 資料集與前處理控制 (Dataset & Preprocessing)
    # ---------------------------------------------------------
    start_year: int = 1996
    standard_train_end_year: int = 2018
    val_end_year: int = 2023

    # IVS Data Filters
    shrcd: typing.Optional[typing.Tuple[int, ...]] = (10, 11, 12)
    exchcd: typing.Optional[typing.Tuple[int, ...]] = None
    return_outlier_quantile: typing.Optional[float] = 0.0
    prc_limit: typing.Optional[float] = None

    # 控制 IVS 特徵的轉換方式:
    # 'raw' (原樣), 'log' (取對數 log1p), 'clip' (截尾固定上限), 'zscore', 'minmax', 'rgb',
    # 'self_demean', 'self_zscore', 'cs_demean', 'cs_zscore'
    ivs_transform: str = 'raw'
    ivs_clip_max: float = 0.9
    # 控制 Label (Target / 報酬率) 的轉換方式:
    # 'raw' (原樣), 'signed_log' (sign(x) * log1p(abs(x))), 'arcsinh'
    # 使用當期 cross-sectional 轉換: 'winsorize', 'rank'
    target_transform: str = 'zscore'
    # ---------------------------------------------------------
    # Model types
    # ---------------------------------------------------------
    model_type: str = "CNN1"
    max_pool: bool = True

    # ---------------------------------------------------------
    # 優化器與訓練超參數 (Optimization & Training)
    # ---------------------------------------------------------
    random_seed: int = 42
    batch_size: int = 512
    learning_rate: float = 1e-3
    dropout_rate: float = 0.0
    l1_lambda: float = 0.0
    l2_lambda: float = 0.0
    epochs: int = 100

    # Ranking Loss 設定
    rank_loss: bool = False
    rank_lambda: float = 0.0

    # Early Stopping 設定
    use_early_stopping: bool = True
    es_patience: int = 15         # 容忍幾個 epoch 驗證集沒有改善
    es_min_delta: float = 0.0   # 被判定為有改善的最小變化量: 1e-4

    # ---------------------------------------------------------
    # 訓練機制設定 (Training Strategy: Standard vs Rolling/Expanding)
    # ---------------------------------------------------------
    # 'standard'  : 一次性切分 Train/Val/Test
    # 'expanding' : 論文中的 Expanding window，先以 warm_up_years 起步，逐年/月加入新資料微調
    # 'rolling_finetune' : 僅使用最新往前推一定時間的資料微調
    training_strategy: str = 'rolling_finetune'

    # 針對 Expanding / Rolling Strategy 的專屬設定
    warm_up_years: int = 7     # 起始熱身期使用的年份總數 (ex: 1998~2004)
    warm_up_epochs: int = 10   # 熱身期資料的初始訓練 epochs
    transfer_epochs: int = 5   # 每推移一個月/年，新資料加入後微調的 epochs
    step_months: int = 1       # 每次推進的步長 (1=逐月, 12=逐年)
    rolling_lookback_months: int = 0 # 針對 rolling_finetune，往回看的歷史資料長度，0 表示僅使用新資料

    # ---------------------------------------------------------
    # Ensemble 設定 (Ensembling)
    # ---------------------------------------------------------
    num_ensembles: int = 1   # 訓練幾個模型來平均預測

    # ---------------------------------------------------------
    # Backtesting 設定 (Backtesting)
    # ---------------------------------------------------------
    base_fee_bps: int = 10
