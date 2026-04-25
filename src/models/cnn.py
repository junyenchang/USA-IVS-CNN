import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    """論文中定義的核心構建單元"""
    def __init__(self, in_channels, out_channels):
        super(CNNBlock, self).__init__()
        # 1. 卷積層 (3x3 filter)，設定 padding=1 以免空間維度縮減過快
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # 2. ReLU 啟動函數
        self.relu = nn.ReLU()
        # 3. 2x2 最大池化層
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)  # ceil_mode=True 以確保 odd-sized feature maps 也能被池化
        # 4. 批次歸一化層
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.bn(x)
        return x

class CNN4(nn.Module):
    """論文中的 CNN4 架構"""
    def __init__(self):
        super(CNN4, self).__init__()

        # 輸入為 1 個 Channel (IV 曲面)，依序經過 4 個區塊
        # 濾鏡數量遞增：16 -> 32 -> 64 -> 128
        self.block1 = CNNBlock(1, 16)
        self.block2 = CNNBlock(16, 32)
        self.block3 = CNNBlock(32, 64)
        self.block4 = CNNBlock(64, 128)

        # 全域平均池化 (GAP - Global Average Pooling)
        # AdaptiveAvgPool2d((1, 1)) 會將任何輸入大小的特徵圖平均縮放為 1x1
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # 單一層全連接節點，預測下個月回報
        self.fc = nn.Linear(128, 1)
        self.apply(init_weights_xavier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 輸入 x 的形狀預期為 [Batch, 1, 10, 18]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # 進行全域平均池化後形狀變為 [Batch, 128, 1, 1]
        x = self.gap(x)

        # 展平為一維向量 [Batch, 128] 以便輸入全連接層
        x = torch.flatten(x, 1)

        # 輸出預測值 [Batch, 1]
        out: torch.Tensor = self.fc(x)
        return out

def init_weights_xavier(m: nn.Module):
    """
    迭代模型中所有模組，並將 Conv2d 與 Linear 層的權重使用 Xavier 初始化。
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # 使用 Xavier uniform 初始化權重
        nn.init.xavier_uniform_(m.weight)

        # 若有 bias，將其初始化為 0
        if m.bias is not None:
            nn.init.zeros_(m.bias)
