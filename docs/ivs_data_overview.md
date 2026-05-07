# Implied Volatility Surface (IVS) Data Introduction

## IVS Definition
隱含波動率表面（Implied Volatility Surface, IVS）是將市場中不同履約價與到期日的選擇權隱含波動率，透過插值映射至固定的 moneyness（10 個 levels）與 maturity（18 個 maturities）格點上，形成 10×18 的規則矩陣。
- **目的**：消除不同股票間到期日與履約價不一致的問題，使其具備橫截面可比性。
- **維度 (Shape)**: `(18, 10)` -> `Axis 0: Maturity (Days)`, `Axis 1: Moneyness (Delta)`。

## Raw Data Source

原始 IVS 資料由 OptionMetrics 取得，並與 CRSP 股票資料進行合併。

本資料集為個股隱含波動率曲面（Implied Volatility Surface, IVS）資料，結合對應股票基本資訊與報酬資料。

資料以表格形式儲存，每筆資料對應某檔股票於特定月份中的一個隱含波動率網格點。

---

## 資料觀測單位

資料表中的每一列代表：**單一股票 × 單一觀測月份 × 單一 IV 網格點**

其中：

* 股票由 `secid` / `permno` 識別
* 觀測月份由 `opt_date` 表示
* IV 網格點由：

  * `days`（剩餘到期天數）
  * `delta`（履約價相對位置）

共同決定

---

## 資料結構

同一支股票在同一觀測月份下，會有多筆資料列。這些資料列對應不同的：

* `days`
* `delta`

組合。

所有網格點共同構成該月份的完整隱含波動率曲面。

例如：

股票 `103404` 在 `1996-02-29` 可能包含：

| days | delta | impl_volatility |
| ---- | ----- | --------------- |
| 30   | -50   | 0.289986        |
| 30   | -45   | 0.290099        |
| 30   | -40   | 0.295790        |
| 30   | -35   | 0.334507        |
| ...  | ...   | ...             |

這些點共同描述該月份的 IV Surface。

---

## 欄位說明

### 識別欄位

| 欄位名稱     | 說明                  |
| -------- | ------------------- |
| `secid`  | OptionMetrics 證券識別碼 |
| `permno` | CRSP 證券識別碼          |

---

### 時間欄位

| 欄位名稱                | 說明            |
| ------------------   | ------------- |
| `opt_date`           | 選擇權資料觀測日期     |
| `crsp_date`          | 對應 CRSP 資料月份 |

---

### IV 曲面欄位

會由 [src/data/dataset.py](/src/data/dataset.py) 讀取原始 parquet 資料後，依據 `days` 和 `delta` 進行 pivot，形成 10x18 的 IV Surface 矩陣。

| 欄位名稱              | 說明           |
| ----------------- | ------------ |
| `days`            | 剩餘到期天數       |
| `delta`           | Delta bucket |
| `impl_volatility` | 該網格點的隱含波動率   |

---

### 公司屬性欄位

用於挑選樣本範圍，或回測時進行分組分析

| 欄位名稱           | 說明              |
| -------------- | --------------- |
| `market_cap`   | 公司市值            |
| `shrcd`        | CRSP share code |
| `size_group`   | 市值分類            |
| `is_non_micro` | 是否為非微型股  |

---

### 報酬欄位

[src/data/dataset.py](/src/data/dataset.py) 會將 `crsp_monthly_return` 當月報酬率往前移動一個月，命名為 `future_return`，作為模型的 label。

| 欄位名稱                  | 說明      |
| --------------------- | ------- |
| `crsp_monthly_return` | 當月報酬率    |
| `future_return`       | 下一個月的報酬率 |

---

## 資料特性

本資料具有以下結構特徵：

### 1. Panel Data

同一股票會跨多個月份重複出現。

---

### 2. Surface Structure

同一股票月份下存在多個 `(days, delta)` 網格點。

資料並非單列即可完整描述單一 observation，而需聚合後形成完整曲面。

---

### 3. 多層級索引關係

可視為：

* 第一層：股票 (`secid`)
* 第二層：月份 (`opt_date`)
* 第三層：IV 網格 (`days`, `delta`)

---

## 範例

以下為單一 IV Surface 的部分節錄資料：

| secid  | opt_date   | days | delta | impl_volatility |
| ------ | ---------- | ---- | ----- | --------------- |
| 103404 | 1996-02-29 | 30   | -50   | 0.289986        |
| 103404 | 1996-02-29 | 30   | -45   | 0.290099        |
| 103404 | 1996-02-29 | 30   | -40   | 0.295790        |

表示：

股票 `103404` 在 `1996-02` 的 IV Surface 中，
30 天到期下不同 delta 的隱含波動率分布。

## Surface Dimensions

每個完整 IV Surface 具有：

- Delta 維度：10
- Maturity 維度：18

因此每筆股票月份 observation 可表示為：

10 × 18 matrix

## 資料樣本展示 (Reshaped View)
單一 Observation 轉換後如下所示：
`Shape: (18, 10)`
```text
           Delta_-50  Delta_-45  ...  Delta_-10
Days_30    [ 0.289 ]  [ 0.290 ]  ...  [ 0.312 ]
Days_60    [ 0.275 ]  [ 0.278 ]  ...  [ 0.295 ]
...           ...        ...     ...     ...
Days_730   [ 0.250 ]  [ 0.252 ]  ...  [ 0.260 ]
```
