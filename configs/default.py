import os
from src.path import OptionPath, ResultsPath
from dataclasses import dataclass, field

@dataclass
class BaselineConfig:
    project_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir: str = os.path.join(OptionPath.IVS)
    result_dir: str = os.path.join(ResultsPath.CNN)

    exp_group: str = ''
    exp_name: str = "CNN4_Baseline"
    task_type: str = "regression" # 或 'classification'

    # 若為 classification，在此設定大跌的定義閾值 (如預測下個月回報 <= -15%)
    crash_threshold: float = -0.15

    # ---------------------------------------------------------
    # 資料集與前處理控制 (Dataset & Preprocessing)
    # ---------------------------------------------------------
    start_year: int = 1996
    end_year: int = 2018

    # 控制 IVS 特徵的轉換方式:
    # 'raw' (原樣), 'log' (取對數 log1p), 'clip' (截尾固定上限)
    ivs_transform: str = 'raw'
    ivs_clip_max: float = 0.8

    # ---------------------------------------------------------
    # Model types
    # ---------------------------------------------------------
    model_type: str = "CNN4"

    # ---------------------------------------------------------
    # 優化器與訓練超參數 (Optimization & Training)
    # ---------------------------------------------------------
    batch_size: int = 512
    learning_rate: float = 1e-3
    epochs: int = 100

    # Early Stopping 設定
    use_early_stopping: bool = True
    es_patience: int = 10         # 容忍幾個 epoch 驗證集沒有改善
    es_min_delta: float = 0.0   # 被判定為有改善的最小變化量: 1e-4

    # ---------------------------------------------------------
    # 訓練機制設定 (Training Strategy: Standard vs Rolling/Expanding)
    # ---------------------------------------------------------
    # 'standard'  : 一次性切分 Train/Val/Test
    # 'expanding' : 論文中的 Expanding window，先以 warm_up_years 起步，逐年加入新資料微調
    training_strategy: str = 'standard'

    # 針對 Expanding / Rolling Strategy 的專屬設定
    warm_up_years: int = 7     # 起始熱身期使用的年份總數 (ex: 1996~2002)
    warm_up_epochs: int = 10   # 熱身期資料的初始訓練 epochs
    transfer_epochs: int = 5   # 每推移一個月/年，新資料加入後微調的 epochs

    # ---------------------------------------------------------
    # Ensemble 設定 (Ensembling)
    # ---------------------------------------------------------
    num_ensembles: int = 10   # 訓練幾個模型來平均預測

