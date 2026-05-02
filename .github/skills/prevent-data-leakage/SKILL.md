---
name: 'prevent-data-leakage'
description: 確保在量化金融與時間序列預測模型中，避免未來資料洩漏（Look-ahead Bias），並確保滾動更新（Rolling/Expanding Window）與回測邏輯的嚴謹性。
---
# Financial Model Safety & Prevent Data Leakage

在處理金融時間序列資料、特徵工程、模型訓練或回測擴充時，請嚴格遵守以下準則，以避免最致命的未來函數（Look-ahead Bias）與資料洩漏問題：

## 1. 杜絕未來資料洩漏 (Prevent Look-ahead Bias)
- **嚴格的時間對齊**：在預測 $T+1$ 期（或未來週期）的報酬時，所有的特徵 $X$ 只能使用 $\le T$ 時間點已知的資訊。
- **轉換與正規化 (Transforms & Normalization)**：
  - 若有如 `zscore`, `minmax` 或其他特徵縮放，**絕對不可**使用全樣本的統計量（例如全局平均數或標準差）。
  - 所有統計量必須僅使用歷史滾動視窗（Trailing Window/Rolling Window）計算，或者僅在「訓練集 (Train Set)」上 Fit，然後 Transform 到驗證/測試集上。

## 2. 滾動與擴展視窗更新 (Rolling & Expanding Windows Strategy)
- **邊界隔離 (Boundary Isolation)**：在設計隨時間推進的 Backtest 或模型更新機制時，確保第 $K$ 次更新的訓練資料 `[T_start, T_N]`，與其驗證/測試資料 `[T_N+1, T_M]` 之間沒有任何重疊 (Overlap)。

## 3. 增量更新機制設計 (Incremental & Online Learning)
- **狀態隔離**：如果是基於新資料進行模型的 fine-tuning 或增量更新，請確保載入的模型權重是前一個時間節點的 snapshot，且在此階段只接觸過去未見過的「最新歷史資料」。
- **批次驗證**：在完成一次視窗滾動訓練後，務必使用一小段獨立的 Out-of-Sample 資料進行 assertion 檢查，確保預測邏輯跟上一個視窗的末端沒有產生不合理的資料斷層或偷看未來的極值。

## 4. 回測的真實性檢查 (Backtest Realism Checks)
- **交易延遲與成本**：確認訊號產生（Signal Generation）到實際執行交易 (Execution) 的時間差。例如，如果用 $T$ 日收盤資料算出訊號，實際交易應發生在 $T+1$ 日開盤或收盤，其報酬率計算必須準確對齊。
- **邊界測試 (Sanity Checks)**：在跑完整的滾動回測前，先手動檢查幾個隨機時間點的 Inference，確認輸入的 IVS tensor 沒有混入目標日期的標的物報酬率 (Target Returns)。
