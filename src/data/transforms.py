import torch
import typing

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
        # 加上 1e-8 避免除以零
        return (x - self.mean) / (self.std + 1e-8)

class MinMaxScale:
    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.min_val) / (self.max_val - self.min_val + 1e-8)

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

    else:
        raise ValueError(f"尚未支援的 IVS 轉換方式: {transform_type}")
