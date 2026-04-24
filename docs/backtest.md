# CNN IVS 策略回測計算邏輯架構

## 1. 核心參數設定
| 參數名稱 | 符號 | 論文設定值 | 說明 |
| :--- | :--- | :--- | :--- |
| **基礎交易手續費率** | $\kappa$ | 0.001 (10 bps) | 包含雙邊交易與滑價 |
| **非微型股放空年費率** | $s_{large}$ | 0.025 (2.5%) | 借券成本 (年化) |
| **微型股放空年費率** | $s_{micro}$ | 0.048 (4.8%) | 借券成本 (年化) |
| **微型股定義** | - | NYSE 底部 20% | 市值低於 NYSE 第 20 百分位數 |

---

## 2. 每月回測流程 (Monthly Loop)

可以從 config 設定**基礎交易手續費率** 為 10 bps or 20 bps。

做多時需要計算基礎交易手續費率；放空時則需要額外再加上**固定放空成本**，等於 `手續費 + 放空成本`。

### Step 1: 原始總報酬計算 ($R_{raw, t}$)
根據 CNN 模型在 $t-1$ 月底輸出的預測值進行排序，建立 Long-Short Portfolio（Top 10% vs Bottom 10%）。
$$R_{raw, t} = \sum_{i=1}^{N} w_{i,t} \cdot R_{i,t}$$
* $w_{i,t}$：第 $t$ 月初分配給股票 $i$ 的權重（做多為正，放空為負）。
* $R_{i,t}$：股票 $i$ 在第 $t$ 月的含息總報酬率。

### Step 2: 單位重平衡成本計算 ($TC_t$)
這是**調倉**產生的手續費。計算重點在於「目標權重」與「漂移權重」之間的差異。
$$TC_t = \kappa \cdot \sum_{i=1}^{N} |w_{i,t} - w_{i,t-}|$$
* $w_{i,t-}$：**漂移權重 (Drifted Weight)**。由於股價波動，月底的實際權重會偏離月初設定的權重。
    * 計算公式：$w_{i,t-} = \frac{w_{i,t-1}(1+R_{i,t-1})}{\sum w_{j,t-1}(1+R_{j,t-1})}$
* **程式實作注意**：Long 和 Short 部位在「賣出舊股」與「買進新股」時都要計算這個 0.1%。

### Step 3: 固定放空成本計算 ($SC_t$)
這是**持有空單**產生的借券利息，僅針對權重為負的部位計算。
$$SC_t = \sum_{i \in \text{Short}} |w_{i,t}| \times \frac{s_i}{12}$$
* **條件判定邏輯**：
    ```python
    if is_microcap(i):
        s_i = 0.048
    else:
        s_i = 0.025
    ```
* **關鍵點**：這是「持有成本」，只要該部位是 Short 且跨過該月，就必須扣除。

### Step 4: 扣除成本後的淨報酬 ($R_{net, t}$)
這是最終反映在資產曲線上的月報酬率。
$$R_{net, t} = R_{raw, t} - TC_t - SC_t$$

---

## 3. 程式開發邏輯建議 (Pseudo-code)

```python
# 1. 判定放空成本費率 (基於市值)
df['shorting_fee_monthly'] = np.where(df['is_non_micro'] is False, 0.048/12, 0.025/12)

# 2. 計算原始報酬 (基於模型預測權重 w)
portfolio_return = (df['weight'] * df['return']).sum()

# 3. 計算重平衡成本 (需記錄前一月權重與漂移)
turnover = np.abs(df['target_weight'] - df['drifted_weight']).sum()
transaction_cost = turnover * 0.001

# 4. 計算放空利息 (僅針對負權重部位)
short_leg = df[df['target_weight'] < 0]
shorting_cost = (short_leg['target_weight'].abs() * short_leg['shorting_fee_monthly']).sum()

# 5. 淨報酬
net_return = portfolio_return - transaction_cost - shorting_cost
```

---

## 4. 績效評估指標
在計算完 `net_return` 時間序列後，需計算以下指標與 **SPY** 進行對照：
1.  **年化淨報酬 (Annualized Net Return)**
2.  **年化波動率 (Annualized Volatility)**
3.  **夏普比率 (Sharpe Ratio)**：需扣除無風險利率。
4.  **最大回撤 (Max Drawdown)**

