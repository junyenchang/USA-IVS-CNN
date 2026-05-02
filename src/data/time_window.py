import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Tuple, List
from .dataset import IVSDataset

class SubsetDataset(Dataset):
    """
    自 PyTorch 原始 Tensor 資料切出的輕量子集，避免重複拷貝。
    """
    def __init__(self, X: torch.Tensor, y: torch.Tensor, dates: np.ndarray, permnos: np.ndarray, transform: Optional[Callable] = None):
        self.X = X
        self.y = y
        self.dates = dates
        self.permnos = permnos
        self.transform = transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, int]:
        x = self.X[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y[idx], str(self.dates[idx]), self.permnos[idx]

class TimeWindowDatasetManager:
    """
    動態記憶體管理員。
    為了不把 25 年的 Pandas DataFrame 塞爆記憶體，我們按年將資料讀檔並轉為 Tensor 快取在記憶體中。
    當需要不同時段的 Train / Val 資料時，直接使用索引切片生出 `SubsetDataset`。
    因為跨年份標籤的關係 (如 12月的 return 在下一年1月)，內部實作以 "多載入前後一年" 取交集。
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

        # 把這段時間的資料一次轉好，但轉完就清掉 DataFrame
        print(f"Loading and pre-processing dataset into Tensors from {start_year} to {val_end_year}...")
        self.base_dataset = IVSDataset(
            data_dir=data_dir,
            start_year=start_year,
            end_year=val_end_year, # 一次性全吃到 val_end_year 以建立整個時間軸
            value_col=value_col,
            target_transform=target_transform,
            transform=None # 將 Transform 延後到取 Subset 時進行，避免記憶體裡存太多擴增資料
        )

        # 將轉換好的 Tensors 抓出來
        self.X_all = self.base_dataset.X
        self.y_all = self.base_dataset.y
        self.dates_all = pd.to_datetime(self.base_dataset.dates)
        self.permnos_all = self.base_dataset.permnos

        # 清除 base_dataset 的 DataFrame 相關東西釋放記憶體
        if hasattr(self.base_dataset, 'df'):
            del self.base_dataset.df
            del self.base_dataset.X_list
            del self.base_dataset.y_list
            del self.base_dataset.date_list
            del self.base_dataset.permno_list
        gc.collect()
        print(f"Dataset cached gracefully. Total samples: {len(self.y_all)}")

    def get_split(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> SubsetDataset:
        """
        取得從 start_date 到 end_date 的資料子集 (inclusive)。
        """
        mask = (self.dates_all >= start_date) & (self.dates_all <= end_date)
        idx = np.where(mask)[0]
        return SubsetDataset(
            self.X_all[idx],
            self.y_all[idx],
            self.base_dataset.dates[idx],
            self.permnos_all[idx],
            transform=self.transform
        )
