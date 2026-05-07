# Tensor Construction

## Overview

本文件說明原始 row-wise IVS parquet 資料如何轉換為可供 CNN 模型使用的張量格式。

原始資料中的每一列僅代表單一 IV 網格點，因此必須經過整理、對齊與重塑，才能形成完整的二維隱含波動率表面。

最終輸出格式為：

```python
(B, 1, T, D)
```

其中：

* `B`：樣本數
* `1`：單通道
* `T`：maturity 維度
* `D`：delta 維度

---

## Processing Pipeline

資料轉換流程如下：

```text
Raw parquet files
    ↓
年度資料合併
    ↓
時間欄位標準化
    ↓
報酬標籤對齊
    ↓
殘缺 IV Surface 過濾
    ↓
同月重複樣本去除
    ↓
Pivot 為 2D matrix
    ↓
Tensor stacking
    ↓
Channel expansion
    ↓
PyTorch Dataset
```

---

## 1. Data Loading

載入指定年份區間內的 parquet 檔案並合併。

每個 parquet 檔案包含：

* 單一年份
* row-wise IVS observations
* 對應 CRSP 資訊

合併後形成完整 panel dataset。

---

## 2. Temporal Standardization

將：

* `opt_date`
* `crsp_date`

轉換為 datetime 格式，以便進行月份對齊與時間操作。

---

## 3. Label Alignment

為建立監督式學習標籤，需要將股票報酬與對應 IV Surface 對齊。

### Return Pool Construction

首先抽取唯一報酬資料：

* `permno`
* `crsp_date`
* `crsp_monthly_return`

---

### Forward Alignment

將報酬月份向前平移一期：

```text
Return month t+1
→ assigned to IV surface at month t
```

因此某月份 IV Surface 會對應預測下一期報酬。

---

## 4. Incomplete Surface Filtering

僅保留完整 IV Surface。

完整條件：

每組：

```text
(secid, opt_date)
```

必須包含全部 delta × maturity 網格點。若存在缺失值，則整組樣本移除。

此步驟確保 CNN 輸入具有固定維度。

---

## 5. Monthly Deduplication

同一股票在同月份可能存在多筆 observation。保留： **該月份最後一個觀測日期** 作為代表樣本。

目的：避免同月重複樣本造成資料洩漏或權重失衡。

---

## 6. Matrix Construction

### Pivot Operation

每組：

```text
(secid, opt_date)
```

轉換為矩陣：

* row → `days`
* column → `delta`
* value → `impl_volatility`

形成：

```text
T × D
```

IV Surface matrix。

---

## 7. Delta Ordering

欄位順序固定為：

```text
Call deltas → Put deltas
```

即：

```text
+10, +20, ..., +50, -50, ..., -10
```

此排序保持跨樣本空間一致性。

---

## 8. Optional Maturity Selection

若指定 `grid_T`，

則僅保留指定 maturity rows。

例如：

```python
grid_T = [30, 60, 90, 180]
```

可縮減輸入高度。

---

## 9. Tensor Stacking

所有月份矩陣沿樣本維度堆疊：

```python
(T, D)
→
(N, T, D)
```

其中：

`N` 為單一股票 across time 的 observation 數。

---

## 10. Channel Expansion

為符合 PyTorch CNN 輸入格式，

新增 channel 維度：

```python
(N, T, D)
→
(N, 1, T, D)
```

其中：

* `1` 表示單通道灰階影像

---

## 11. Final Dataset Structure

最終 Dataset 包含：

### Input Tensor

```python
X.shape = (B, 1, T, D)
```

### Target

```python
y.shape = (B,)
```

### Metadata

每筆樣本同時保留：

* `opt_date`
* `permno`

供：

* 時間切分
* 回測分組
* 投資組合建構

使用。

---

## Example

原始資料：

| secid  | opt_date   | days | delta | impl_volatility |
| ------ | ---------- | ---- | ----- | --------------- |
| 103404 | 1996-02-29 | 30   | -50   | 0.289986        |
| 103404 | 1996-02-29 | 30   | -45   | 0.290099        |
| ...    | ...        | ...  | ...   | ...             |

經 pivot 後：

| days \ delta | +10 | +20 | ... | -50 |
| ------------ | --- | --- | --- | --- |
| 30           | ... | ... | ... | ... |
| 60           | ... | ... | ... | ... |
| ...          | ... | ... | ... | ... |

形成：

```python
(18, 10)
```

再擴充為：

```python
(1, 18, 10)
```

供 CNN 使用。

---

## Dataset Output Interface

每次取樣返回：

```python
(x, y, opt_date, permno)
```

其中：

* `x`：IV Surface tensor
* `y`：目標報酬
* `opt_date`：觀測日期
* `permno`：股票識別碼
