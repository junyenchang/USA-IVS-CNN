import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import time
import os
import matplotlib.pyplot as plt

from src.data.dataset import IVSDataset
from src.path import OptionPath

t0 = time.time()
print("Loading dataset...")
dataset = IVSDataset(
    OptionPath.IVS_ALL,
    start_year=1996, end_year=2024,
    shrcd=(10, 11),
    exchcd=(1, 2, 3),
    # prc_limit=5,
    )
X = dataset.X.squeeze(1).numpy()
dates = pd.to_datetime(dataset.dates)
y_raw = dataset.y.numpy()
permnos = dataset.permnos

# 2. 扁平化曲面
n_samples, n_T, n_Delta = X.shape
X_flat = np.log1p(X.reshape(n_samples, -1))

print("Preparing dataframe...")
feature_names = [f"IV_T{t}_D{d}" for t in range(n_T) for d in range(n_Delta)]

def cs_normalize_returns(df, col='target_return_raw'):
    """Cross-sectional normalize within each month"""
    def _normalize(group):
        g = group[col]
        # lo, hi = g.quantile(0.01), g.quantile(0.99)
        # g = g.clip(lo, hi)
        # return g
        return (g - g.mean()) / (g.std() + 1e-8)
    return df.groupby('date', group_keys=False).apply(_normalize)

df = pd.DataFrame({'date': dates, 'permno': permnos, 'target_return_raw': y_raw})
df['target_return'] = cs_normalize_returns(df)
# df['target_return'] = df['target_return_raw']

# 3. 截面標準化 (Cross-sectional Standardization) 處理共線性
df_X = pd.DataFrame(X_flat, columns=feature_names)
df_X['date'] = df['date']

grouped = df_X.groupby('date')
# 向量化計算 Mean 與 Std 並廣播回原維度
# df_X[feature_names] = (df_X[feature_names] - grouped[feature_names].transform('mean')) / grouped[feature_names].transform('std')
# df_X[feature_names] = df_X[feature_names].fillna(0) # 預防 std=0 產生的 nan

# 4. Out-of-Sample (OOS) 預測 - 高效的 Expanding Window Ridge (GPU Accelerated)
print("Starting Fast Expanding Window Ridge Regression...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 轉換為 PyTorch Tensor 加入 GPU 運算
# X_tensor = torch.log1p(torch.tensor(df_X[feature_names].values, dtype=torch.float32, device=device))
X_tensor = torch.tensor(df_X[feature_names].values, dtype=torch.float32, device=device)
y_tensor = torch.tensor(df['target_return'].values, dtype=torch.float32, device=device)
dates_array = df['date'].values

unique_dates = np.sort(df['date'].unique())
train_length = 84
alpha = 0.1

# 預先計算每個月的 X^T X 與 X^T y
# Ridge 的公式為 w = (X^T X + alpha*I)^{-1} X^T y
# 我們可以利用 Expanding window 的特性，直接將過去的 (X^T X) 矩陣累加，而不需要每次重新傳入幾十萬筆資料。
months_XtX = []
months_Xty = []
months_masks = []
months_n = []

for date in unique_dates:
    mask = (dates_array == date)
    months_masks.append(mask)
    months_n.append(mask.sum())

    Xt = X_tensor[mask]
    yt = y_tensor[mask]
    months_XtX.append(Xt.T @ Xt)
    months_Xty.append(Xt.T @ yt)

# 起始矩陣累加
A = sum(months_XtX[:train_length])
b = sum(months_Xty[:train_length])
n_train = sum(months_n[:train_length])
I = torch.eye(X_tensor.shape[1], device=device) * alpha

oos_preds = np.full(n_samples, np.nan)

# 高速回圈：每次只需解 180x180 的矩陣，耗時幾乎為 0
for i in range(train_length, len(unique_dates)):
    test_mask = months_masks[i]
    if test_mask.sum() == 0:
        continue

    # 計算權重 w = (A + alpha * I)^{-1} b
    w = torch.linalg.solve(A + I, b)

    # 預測當月
    Xt_test = X_tensor[test_mask]
    pred = Xt_test @ w
    oos_preds[test_mask] = pred.cpu().numpy()

    # 累加上當月資料作為下個月的 Training (Expanding)
    A += months_XtX[i]
    b += months_Xty[i]

df['oos_pred'] = oos_preds
res_df = df.dropna(subset=['oos_pred']).copy()

print("\n=== Regression Metrics (Out-of-Sample) ===")
def calc_ic(group):
    if len(group) < 10: return np.nan
    return spearmanr(group['oos_pred'], group['target_return_raw'])[0]

monthly_ic = res_df.groupby('date').apply(calc_ic, include_groups=False).dropna() # type: ignore
ic_mean = monthly_ic.mean()
ic_ir = ic_mean / monthly_ic.std()
t_stat = ic_mean / (monthly_ic.std() / np.sqrt(len(monthly_ic)))

# Long-Short Portfolio
res_df.loc[:, 'rank'] = res_df.groupby('date')['oos_pred'].transform(lambda x: pd.qcut(x, 10, labels=False, duplicates='drop'))
portfolio_rets = res_df.groupby(['date', 'rank'])['target_return_raw'].mean().unstack()
portfolio_medians = res_df.groupby(['date', 'rank'])['target_return_raw'].median().unstack()

ls_strategy = portfolio_rets[9] - portfolio_rets[0]
ann_ret = ls_strategy.mean() * 12
ann_vol = ls_strategy.std() * np.sqrt(12)
sharpe = ann_ret / ann_vol

output_text = f"""=== Ridge Regression Evaluation Results ===
Total runtime: {time.time() - t0:.2f} seconds
Data span: {df['date'].min().date()} to {df['date'].max().date()}

[Rank IC]
Mean Rank IC: {ic_mean * 100:.4f}%
Information Ratio (IR): {ic_ir:.4f}
t-stat: {t_stat:.2f}

[Long-Short Portfolio]
Rank-10 (High Prediction) Mean Return: {portfolio_rets[9].mean()*12*100:.2f}% (Annualized)
Rank-1  (Low Prediction) Mean Return : {portfolio_rets[0].mean()*12*100:.2f}% (Annualized)
L/S Strategy Mean Return: {ann_ret*100:.2f}% (Annualized)
L/S Strategy Volatility : {ann_vol*100:.2f}% (Annualized)
L/S Strategy Sharpe Ratio: {sharpe:.4f}
"""

print(output_text)
print(portfolio_medians.mean())
print(portfolio_rets.mean() * 12 * 100)

# 儲存結果
output_dir = "DB/Results/Regression_Baseline"
os.makedirs(output_dir, exist_ok=True)
with open(f"{output_dir}/regression_performance.txt", "w") as f:
    f.write(output_text)

print(f"Results successfully saved to {output_dir}/regression_performance.txt")

dates_index = portfolio_rets.index
cum_long = (1 + portfolio_rets[9].fillna(0.0)).cumprod()
cum_short = (1 - portfolio_rets[0].fillna(0.0)).cumprod()
cum_ls = (1 + ls_strategy.fillna(0.0)).cumprod()

plt.figure(figsize=(11, 6))
plt.plot(dates_index, cum_ls, label=f'Long-Short Strategy (Sharpe: {sharpe:.2f}, Cumulative Return : {cum_ls.iloc[-1]:.2f})', color='black', linewidth=2.2)
plt.plot(dates_index, cum_long, label=f'Long Side Only (Rank 10) Cumulative Return : {cum_long.iloc[-1]:.2f}', color='green', alpha=0.7, linestyle='--')
plt.plot(dates_index, cum_short, label=f'Short Side Only (Rank 1) Cumulative Return : {cum_short.iloc[-1]:.2f}', color='red', alpha=0.7, linestyle='--')
plt.yscale('log')
plt.title('Out-of-Sample Cumulative Performance (Ridge Baseline)', fontsize=14, fontweight='bold')
plt.xlabel('Date'); plt.ylabel('Cumulative Return (Wealth Index)'); plt.grid(True, linestyle='--', alpha=0.5); plt.legend(loc='upper left'); plt.tight_layout()

chart_path = f"{output_dir}/performance_chart.png"
plt.savefig(chart_path, dpi=300)
plt.close()
print(f"Performance chart successfully saved to {chart_path}")
