import torch
import typing
import matplotlib.pyplot as plt
import numpy as np

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
    elif transform_type.lower() == 'signed_log':
        return lambda x: np.sign(x) * np.log1p(np.abs(x))
    elif transform_type.lower() == 'arcsinh':
        return lambda x: np.arcsinh(x)
    elif transform_type.lower() == 'winsorize':
        lower = kwargs.get('lower', -0.5)
        upper = kwargs.get('upper', 0.5)
        return lambda x: np.clip(x, lower, upper)
    else:
        raise ValueError(f"尚未支援的 Target 轉換方式: {transform_type}")
