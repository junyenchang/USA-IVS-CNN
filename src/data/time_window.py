import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple, List
from .dataset import IVSDataset

class SubsetDataset(Dataset):
    """
    自 PyTorch 原始 Tensor 資料切出的輕量子集，避免重複拷貝。
    """
    def __init__(self, X: torch.Tensor, y: torch.Tensor, dates: np.ndarray, permnos: np.ndarray, transform: Optional[Callable] = None, y_raw: Optional[torch.Tensor] = None):
        self.X = X
        self.y = y
        self.dates = dates
        self.permnos = permnos
        self.transform = transform
        self.y_raw = y_raw if y_raw is not None else y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, int, float]:
        x = self.X[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y[idx], str(self.dates[idx]), self.permnos[idx], self.y_raw[idx].item()

class TimeWindowDatasetManager:
    """
    記憶體高效的時間窗口數據管理員 (Memory-Efficient Year-by-Year Loading)。

    設計策略：
    1. 首先輕量級地讀取所有年份的報酬數據（只讀 permno, crsp_date, crsp_monthly_return）
    2. 構建全局報酬池，包含隔年數據以確保年末資料有對應的 t+1 報酬（防止 Data Leakage）
    3. 逐年讀取 IVS 數據，傳入預構的報酬池給 IVSDataset
    4. 每年處理完後立即清空 Pandas DataFrame，保持記憶體低位
    5. 只在記憶體中保持 Tensor（遠小於原始 Parquet 檔案）
    """
    def __init__(
        self,
        data_dir: str,
        start_year: int,
        val_end_year: int,
        value_col: str = 'impl_volatility',
        target_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None
    ):
        self.data_dir = data_dir
        self.value_col = value_col
        self.target_transform = target_transform
        self.transform = transform
        self.start_year = start_year
        self.val_end_year = val_end_year

        # ===== 步驟 1：輕量級構建全局報酬池 =====
        print(f"Step 1: Building lightweight global returns pool from {start_year} to {val_end_year}...")
        self.global_returns = self._build_global_returns_pool(start_year, val_end_year)
        print(f"  ✓ Global returns pool built: {len(self.global_returns)} records")

        # ===== 步驟 2：逐年讀取並轉為 Tensors =====
        print(f"Step 2: Loading and converting year-by-year data to Tensors...")
        self.X_all, self.y_all, self.y_raw_all, self.dates_all, self.permnos_all = self._load_year_by_year(start_year, val_end_year)
        print(f" Dataset cached. Total samples: {len(self.y_all)}")

    def _build_global_returns_pool(self, start_year: int, end_year: int) -> pd.DataFrame:
        """
        輕量級讀取所有年份，只提取報酬相關欄位，構建全局報酬池。
        為了防止 Look-ahead Bias，包含 end_year 隔年的報酬數據。
        """
        all_returns = []

        # 讀取 start_year 到 end_year+1（多讀一年，確保年末 12 月有對應的隔年 1 月報酬）
        for year in range(start_year, end_year + 2):
            file_path = f"{self.data_dir}/option_ivs_crsp_{year}.parquet"

            if not os.path.exists(file_path):
                # 如果隔年檔案不存在，就停止
                break

            # 只讀取必要欄位，大幅降低記憶體用量
            df_year = pd.read_parquet(file_path, columns=['permno', 'crsp_date', 'crsp_monthly_return'])
            df_year['crsp_date'] = pd.to_datetime(df_year['crsp_date'])
            all_returns.append(df_year)

        # 合併所有年份的報酬數據
        returns_concat = pd.concat(all_returns, ignore_index=True)
        returns_concat = returns_concat.drop_duplicates(subset=['permno', 'crsp_date'])

        # 計算報酬對應的月份（t 月的報酬 → t-1 月的 target）
        returns_concat['target_for_month'] = returns_concat['crsp_date'].dt.to_period('M') - 1
        returns_concat = returns_concat[['permno', 'target_for_month', 'crsp_monthly_return']]
        returns_concat.rename(columns={'crsp_monthly_return': 'future_return'}, inplace=True)

        return returns_concat

    def _load_year_by_year(self, start_year: int, end_year: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        """
        逐年讀取 IVS 數據，轉為 Tensors，並逐年清空 Pandas DataFrame 釋放記憶體。
        """
        all_X = []
        all_y = []
        all_y_raw = []
        all_dates = []
        all_permnos = []

        for year in range(start_year, end_year + 1):
            print(f"  Loading year {year}...")

            # 讀取該年的 IVS 數據，傳入預構的報酬池
            year_dataset = IVSDataset(
                data_dir=self.data_dir,
                start_year=year,
                end_year=year,  # 逐年讀取
                value_col=self.value_col,
                target_transform=self.target_transform,
                transform=None,  # Transform 延後到 get_split() 時進行
                global_returns=self.global_returns  # 傳入全局報酬池，避免重複計算
            )

            # 收集該年的 Tensors
            all_X.append(year_dataset.X)
            all_y.append(year_dataset.y)
            all_y_raw.append(year_dataset.y_raw)
            all_dates.append(year_dataset.dates)
            all_permnos.extend(year_dataset.permnos)

            # 立即清空該年的 DataFrame 和列表，釋放記憶體
            if hasattr(year_dataset, 'df'):
                del year_dataset.df
            if hasattr(year_dataset, 'X_list'):
                del year_dataset.X_list
            if hasattr(year_dataset, 'y_list'):
                del year_dataset.y_list
            if hasattr(year_dataset, 'date_list'):
                del year_dataset.date_list
            if hasattr(year_dataset, 'permno_list'):
                del year_dataset.permno_list

            del year_dataset
            gc.collect()

        # 合併所有年份的 Tensors
        X_all = torch.cat(all_X, dim=0)
        y_all = torch.cat(all_y, dim=0)
        y_raw_all = torch.cat(all_y_raw, dim=0)
        dates_all = np.concatenate(all_dates, axis=0)
        permnos_all = np.array(all_permnos)

        return X_all, y_all, y_raw_all, dates_all, permnos_all

    def get_split(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> SubsetDataset:
        """
        取得從 start_date 到 end_date 的資料子集 (inclusive)。
        防止 Data Leakage：只返回該時間窗口內的真實資料。
        """
        mask = (self.dates_all >= start_date) & (self.dates_all <= end_date)
        idx = np.where(mask)[0]
        return SubsetDataset(
            self.X_all[idx],
            self.y_all[idx],
            self.dates_all[idx],
            self.permnos_all[idx],
            transform=self.transform,
            y_raw=self.y_raw_all[idx]
        )
