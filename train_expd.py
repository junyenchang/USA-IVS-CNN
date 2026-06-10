import typing
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset import IVSDataset
from src.path import OptionPath
from src.models.cnn import CNN4
from src.utils.seed import set_seed

# ================= 1. 參數設定 =================
START_YEAR = 1998
END_YEAR = 2024
WARMUP_END_YEAR = START_YEAR + 6
NUM_ENSEMBLES = 5
BATCH_SIZE = 512
USE_ZSCORE = True
SEMI_ANNUAL_UPDATE = True  # 是否每12個月進行一次全樣本更新 (True) 或是每月都用全樣本更新 (False)
UPDATE_FREQ = 2
UPDATE_MONTHS = 12  # 如果 SEMI_ANNUAL_UPDATE=True，則每隔多少個月進行一次全樣本更新
dropout_rate = 0.0
shrcd = (10, 11, 12)
prc_limit = 5
# update_epochs = 5

# ================= 2. 一次性載入全部資料 =================
print(f"Loading data from {START_YEAR} to {END_YEAR}...")
dataset = IVSDataset(
    OptionPath.IVS_ALL,
    start_year=START_YEAR,
    end_year=END_YEAR,
    shrcd=shrcd,
    prc_limit=prc_limit,
    # target_transform=lambda y: np.log1p(y) * 100
    target_transform=lambda y: (y > 0).astype(np.float32)
)

# 為了方便切時間，將日期轉成月份刻度 (例: 2003-01)
dates_pd = pd.to_datetime(dataset.dates)
periods = dates_pd.to_period('M')

# ================= 3. 處理 Warm-up 資料與 Transform =================
warmup_mask = (dates_pd.year <= WARMUP_END_YEAR)
X_warmup = dataset.X[warmup_mask]
y_warmup = dataset.y[warmup_mask]

if USE_ZSCORE:
    print("Applying Z-score normalization to IVS...")
    # 計算 Z-score的平均/標準差 (為防資料洩漏，只能拿 Warm-up 指定範圍來對位)
    mean_val = X_warmup.mean().item()
    std_val = X_warmup.std().item()

    def transform_ivs(x_tensor):
        return (x_tensor - mean_val) / (std_val + 1e-7)

    X_all_tf = transform_ivs(dataset.X) # 對所有的 X 進行 Transform
    X_warmup_tf = X_all_tf[warmup_mask]
else:
    # 不轉換，直接使用原始特徵 (使用 .clone() 避免不小心改到原始資料)
    X_all_tf = dataset.X.clone()
    X_warmup_tf = X_warmup.clone()

warmup_loader = DataLoader(TensorDataset(X_warmup_tf, y_warmup), batch_size=BATCH_SIZE, shuffle=True)

# ================= 4. 初始化 Ensemble 進行 Warm-up =================
print(f"\nStart Warm-up training for {NUM_ENSEMBLES} models ({START_YEAR} - {WARMUP_END_YEAR})...")
models: typing.List[CNN4] = []
optimizers: typing.List[optim.Optimizer] = []
criterion = nn.MSELoss()

for i in range(NUM_ENSEMBLES):
    set_seed(42 + i) # 確保每個模型權重初始化及資料順序不同
    model = CNN4(in_channels=dataset.X.shape[1], dropout_rate=dropout_rate).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for ep in range(10):
        for bx, by in warmup_loader:
            bx, by = bx.cuda(), by.cuda()
            optimizer.zero_grad()
            loss = criterion(model(bx).view(-1), by)
            loss.backward()
            optimizer.step()

    models.append(model)
    optimizers.append(optimizer)
    print(f" [+] Model {i+1} Warm-up done.")


# ================= 5. Expanding 迴圈 (嚴格切窗) =================
unique_months = sorted(list(set(periods)))
test_months = [m for m in unique_months if m.year > WARMUP_END_YEAR]

all_preds_df = []

for step, test_month in enumerate(test_months):
    print(f"\n--- Testing Inference Month: {test_month} ---")

    # ----- A. 預測本月 (Inference) -----
    test_mask = (periods == test_month)
    X_test = X_all_tf[test_mask]
    y_test = dataset.y[test_mask]
    dates_test = dataset.dates[test_mask]     # numpy array
    permnos_test = dataset.permnos[test_mask] # numpy array

    if len(X_test) > 0:
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
        month_preds = []

        with torch.no_grad():
            for model in models:
                model.eval()
                model_preds = []
                for bx, _ in test_loader:
                    model_preds.append(model(bx.cuda()).view(-1).cpu().numpy())
                month_preds.append(np.concatenate(model_preds))

        ensemble_pred = np.mean(month_preds, axis=0)

        all_preds_df.append(pd.DataFrame({
            'Date': dates_test,
            'Permno': permnos_test,
            'Pred': ensemble_pred,
            'Actual': y_test.numpy()
        }))

    # ----- B. 模型更新 (Expanding Fine-Tuning) -----
    if (step + 1) % UPDATE_FREQ != 0:
        print(f"[累積中] 本月預測完畢，跳過更新，將於累積 {UPDATE_FREQ} 個月資料後一併更新。")
        continue
    # 【核心防漏邏輯】：假設現在 test_month 是 2003-01，我們剛做完了 2003-01 特徵的推論 (預測 2003-02 的報酬)
    # 接下來可以把什麼資料納入模型訓練？
    # 答案是 2002-12 的資料 (因為 2002-12 的 Label 是 2003-01 的當月報酬，2003-01月結束後就知道了！)
    finetune_end_month = test_month - 1
    finetune_start_month = test_month - UPDATE_FREQ

    # 斷言保護規則
    assert finetune_end_month < test_month, "Data Leakage: 微調樣本觸碰到甚至超過測試期！"

    is_full_update = False  # 預設為全樣本更新，下面根據 SEMI_ANNUAL_UPDATE 的設定調整
    if SEMI_ANNUAL_UPDATE:
        is_full_update = ((step + 1) % 12 == 0)
        if is_full_update:
            # 滿 12 個月：使用從頭到 finetune_end_month 的所有歷史資料
            finetune_mask = (periods <= finetune_end_month)
            print(f"[年度全樣本更新] Window: Up to {finetune_end_month} (Train samples: {finetune_mask.sum()})")
        else:
            # 平常月份：僅使用 finetune_end_month 當月的最新資料
            finetune_mask = (periods >= finetune_start_month) & (periods <= finetune_end_month)
            print(f"[最新 {UPDATE_FREQ} 個月資料更新] Window: {finetune_start_month} to {finetune_end_month} (Train samples: {finetune_mask.sum()})")
    else:
        # Expanding: 從第一天開始累積到 finetune_end_month 作為最新的 Train Set
        is_full_update = True
        finetune_mask = (periods <= finetune_end_month)
        print(f" Expanding Finetune Window: Up to {finetune_end_month} (Total train samples: {finetune_mask.sum()})")
    X_finetune = X_all_tf[finetune_mask]
    y_finetune = dataset.y[finetune_mask]

    finetune_loader = DataLoader(TensorDataset(X_finetune, y_finetune), batch_size=BATCH_SIZE, shuffle=True)

    for model, optimizer in zip(models, optimizers):
        model.train()
        if SEMI_ANNUAL_UPDATE:
            if is_full_update:
                update_epochs = 10
                for param_group in optimizer.param_groups:
                    param_group['betas'] = (0.9, 0.999)
                    param_group['lr'] = 1e-3
            else:
                update_epochs = 5
                for param_group in optimizer.param_groups:
                    param_group['betas'] = (0.6, 0.999)
                    param_group['lr'] = 5e-5

        for ep in range(update_epochs):  # 更新 5 Epochs
            for bx, by in finetune_loader:
                bx, by = bx.cuda(), by.cuda()
                optimizer.zero_grad()
                loss = criterion(model(bx).view(-1), by)
                loss.backward()
                optimizer.step()

# ================= 6. 結論與匯出 =================
if len(all_preds_df) > 0:
    final_predictions = pd.concat(all_preds_df, ignore_index=True)
    print("\n完成 所有月份預測總筆數:", len(final_predictions))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_predictions.to_csv(f"expanding_ensemble_predictions_{timestamp}.csv", index=False)
    print(f"Parameters: ")
    print(f" - NUM_ENSEMBLES={NUM_ENSEMBLES}, BATCH_SIZE={BATCH_SIZE},")
    print(f" - dropout_rate={dropout_rate},")
    print(f" - shrcd={shrcd}, prc_limit={prc_limit},")
    if USE_ZSCORE:
        print(" - Z-score normalization on IVS")

