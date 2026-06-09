import torch
import typing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class ClipTransform:
    def __init__(self, max_val: float = 1.0):
        self.max_val = max_val

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, max=self.max_val)

class ZScoreNormalize:
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + 1e-8)

class MinMaxScale:
    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.min_val) / (self.max_val - self.min_val + 1e-8)

class RGBTransform:
    def __init__(self, cmap_name: str = 'viridis', min_val: float = 0.0, max_val: float = 1.0):
        self.cmap = plt.get_cmap(cmap_name)
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_squeeze = x.squeeze(0)

        # 將數值縮放到 0~1 之間，否則 Colormap 無法正確映射
        x_norm = (x_squeeze - self.min_val) / (self.max_val - self.min_val + 1e-8)
        x_norm = torch.clamp(x_norm, 0, 1).cpu().numpy()

        # 透過 matplotlib 將數值轉換為 (H, W, 4) 的產出，取前 3 個通道 (R, G, B)
        rgb_np = self.cmap(x_norm)[..., :3]

        # PyTorch 影像通道順序要求為 (C, H, W)，因此從 (H, W, 3) 轉換為 (3, H, W)
        return torch.tensor(rgb_np, dtype=torch.float32).permute(2, 0, 1)

class CSDemean:
    is_cross_sectional = True
    def __call__(self, x: torch.Tensor, dates: np.ndarray) -> torch.Tensor:
        x_out = x.clone()
        for date in np.unique(dates):
            mask = (dates == date)
            mean_val = x_out[mask].mean(dim=0, keepdim=True)
            x_out[mask] = x_out[mask] - mean_val
        return x_out

class CSZScore:
    is_cross_sectional = True
    def __call__(self, x: torch.Tensor, dates: np.ndarray) -> torch.Tensor:
        x_out = x.clone()
        for date in np.unique(dates):
            mask = (dates == date)
            mean_val = x_out[mask].mean(dim=0, keepdim=True)
            std_val = x_out[mask].std(dim=0, keepdim=True)
            x_out[mask] = (x_out[mask] - mean_val) / (std_val + 1e-8)
        return x_out

class SelfDemean:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x is (C, H, W)
        return x - x.mean()

class SelfZScore:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x is (C, H, W)
        return (x - x.mean()) / (x.std() + 1e-8)

# 工廠函數多開一個雙星號參數 kwargs，用來接收外部傳進來的統計值
def get_ivs_transform(transform_type: typing.Optional[str], **kwargs) -> typing.Optional[typing.Callable]:
    if transform_type is None or transform_type.lower() == 'raw':
        return None
    elif transform_type.lower() == 'log':
        return lambda x: torch.log1p(x)
    elif transform_type.lower() == 'clip':
        return ClipTransform(max_val=kwargs['max_val'])
    elif transform_type.lower() == 'zscore':
        return ZScoreNormalize(mean=kwargs['mean'], std=kwargs['std'])
    elif transform_type.lower() == 'self_demean':
        return SelfDemean()
    elif transform_type.lower() == 'self_zscore':
        return SelfZScore()
    elif transform_type.lower() == 'cs_demean':
        return CSDemean()
    elif transform_type.lower() == 'cs_zscore':
        return CSZScore()
    elif transform_type.lower() == 'minmax':
        return MinMaxScale(min_val=kwargs['min_val'], max_val=kwargs['max_val'])
    elif transform_type.lower() == 'rgb':
        return RGBTransform(
            cmap_name=kwargs.get('cmap_name', 'viridis'),
            min_val=kwargs.get('min_val', 0.0),
            max_val=kwargs.get('max_val', 1.0)
        )

    else:
        raise ValueError(f"尚未支援的 IVS 轉換方式: {transform_type}")

def get_target_transform(transform_type: typing.Optional[str], **kwargs) -> typing.Optional[typing.Callable]:
    if transform_type is None or transform_type.lower() == 'raw':
        return None
    elif transform_type.lower() == 'log':
        return lambda x: np.log1p(x)
    elif transform_type.lower() == 'log100':
        return lambda x: np.log1p(x) * 100
    elif transform_type.lower() == '100y':
        return lambda x: x * 100
    elif transform_type.lower() == 'win_100':
        def cs_winsorize(x, dates=None, **kw):
            if dates is None:
                raise ValueError("Cross-sectional winsorize requires 'dates' argument.")
            df = pd.DataFrame({'y': x * 100, 'date': dates})
            df['ym'] = pd.to_datetime(df['date']).dt.to_period('M')
            lower = df.groupby('ym')['y'].transform(lambda g: g.quantile(0.01))
            upper = df.groupby('ym')['y'].transform(lambda g: g.quantile(0.99))
            return np.clip(np.asarray(x), lower.to_numpy(), upper.to_numpy())
        return cs_winsorize
    elif transform_type.lower() == 'signed_log':
        return lambda x, **kw: np.sign(x) * np.log1p(np.abs(x))
    elif transform_type.lower() == 'arcsinh':
        return lambda x, **kw: np.arcsinh(x)
    elif transform_type.lower() == 'winsorize':
        def cs_winsorize(x, dates=None, **kw):
            if dates is None:
                raise ValueError("Cross-sectional winsorize requires 'dates' argument.")
            df = pd.DataFrame({'y': x, 'date': dates})
            df['ym'] = pd.to_datetime(df['date']).dt.to_period('M')
            lower = df.groupby('ym')['y'].transform(lambda g: g.quantile(0.01))
            upper = df.groupby('ym')['y'].transform(lambda g: g.quantile(0.99))
            return np.clip(np.asarray(x), lower.to_numpy(), upper.to_numpy())
        return cs_winsorize
    elif transform_type.lower() == 'rank':
        def cs_rank(x, dates=None, **kw):
            if dates is None:
                raise ValueError("Cross-sectional rank requires 'dates' argument.")
            df = pd.DataFrame({'y': x, 'date': dates})
            df['ym'] = pd.to_datetime(df['date']).dt.to_period('M')
            # 沿用與 winsorize 相同的設計思路：這會在每個月內部橫向排序。
            # pct=True 會讓排名變成分位數百分比 (落在 (0, 1] 之間)。報酬最高為 1，報酬最低逼近 0。
            ranks = df.groupby('ym')['y'].rank(pct=True, ascending=True)
            # 將 0~1 的分位數映射至 -1 到 1 的神經網路最適訓練區間
            return (ranks.to_numpy() * 2) - 1
        return cs_rank
    elif transform_type.lower() == 'zscore':
        def cs_zscore_winsor(x, dates=None, **kw):
            if dates is None:
                raise ValueError("Cross-sectional zscore requires 'dates' argument.")
            df = pd.DataFrame({'y': x, 'date': dates})
            df['ym'] = pd.to_datetime(df['date']).dt.to_period('M')
            lower = df.groupby('ym')['y'].transform(lambda g: g.quantile(0.01))
            upper = df.groupby('ym')['y'].transform(lambda g: g.quantile(0.99))
            y_clip = df['y'].clip(lower, upper)
            mean = y_clip.groupby(df['ym']).transform('mean')
            std = y_clip.groupby(df['ym']).transform('std')
            std = std.replace(0, np.nan)
            return ((y_clip - mean) / std).fillna(0).to_numpy()
        return cs_zscore_winsor
    elif transform_type.lower() == 'log_zscore':
        def cs_log_zscore_winsor(x, dates=None, **kw):
            if dates is None:
                raise ValueError("Cross-sectional log_zscore requires 'dates' argument.")
            df = pd.DataFrame({'y': np.log1p(x), 'date': dates})
            df['ym'] = pd.to_datetime(df['date']).dt.to_period('M')
            lower = df.groupby('ym')['y'].transform(lambda g: g.quantile(0.01))
            upper = df.groupby('ym')['y'].transform(lambda g: g.quantile(0.99))
            y_clip = df['y'].clip(lower, upper)
            mean = y_clip.groupby(df['ym']).transform('mean')
            std = y_clip.groupby(df['ym']).transform('std')
            std = std.replace(0, np.nan)
            return ((y_clip - mean) / std).fillna(0).to_numpy()
        return cs_log_zscore_winsor
    else:
        raise ValueError(f"尚未支援的 Target 轉換方式: {transform_type}")
