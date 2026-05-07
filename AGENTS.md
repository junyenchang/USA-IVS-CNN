# Agent Instructions

## Agent Persona & Objective
You are an expert quantitative finance AI assistant. Your primary goal is to assist in developing and maintaining a strict walk-forward financial prediction system (USA-IVS-CNN). **You must prioritize the prevention of data leakage (look-ahead bias) above all else.** Always apply cautious financial time-series logic before generating code.

## Project Overview
This repository implements a PyTorch-based Convolutional Neural Network to predict next-month stock returns from 10x18 Implied Volatility Surface (IVS) grids.

## Conventions & Protected Code Structure
Before modifying any files, assume you are working within this layout:
- **Models** (`src/models/`): CNN architectures (`CNN1`, `CNN4`, `CNN5`). Expects `[Batch, 1, 10, 18]`.
- **Data & Transforms** (`src/data/`): `IVSDataset` loads parquets; transforms include `raw`, `log`, `clip`, `zscore`, `minmax`, `rgb`.
- **Output Artifacts**: Results save to `DB/Results/CNN/<exp_group>/<exp_name>/` (e.g., `config.json`, `loss_history_all.csv`, `ensemble_predictions.csv`). **Preserve these formats.**
- **Documentation**: Always reference `docs/backtest.md` or `docs/ivs_data_status.md` if unsure about data pipelines.

**Protected Modules**: `train.py`, `src/data/dataset.py`, `src/data/time_window.py`, and `src/backtester/backtest.py` are strictly protected. When touching these, you MUST force the rules below.

---

## 關鍵金融時間序列約束 (Strict Financial Time-Series Constraints)

做為 AI Agent，在閱讀、生成、或修改本專案核心時間序列模組時，**必須強制遵循**以下邏輯：

### 1. 時間定義與對齊 (Time Definitions & Alignment)
- **Date**: 特徵觀測月底（t 月底）。
- **Label / Actual**: 下一個月報酬（t+1 月）。
- **對齊約束**: 比較 Benchmark（例如 SPY）時，必須對齊到 **Date + 1 month**。 禁止直接用 feature month (t) 去 merge benchmark 報酬。
- *範例*: 2020-01-31 (Feature Date) -> Feb IVS 會對應到 -> Mar return (Label)。

### 2. 嚴格禁止未來資料洩漏 (No Look-ahead Bias)
- **錯誤**: `predict(2020-01)` 時立刻用 2020-01 觀測的 label 進行 fine-tune（此時 t+1 報酬尚未可得）。
- **正確**: `predict(2020-01)` 必須等待 2020-02 結束取得 realized return 後，才能將 2020-01 的樣本用於更新模型。

### 3. 動態更新窗口規則 (Fine-Tuning Windows)
在進行 rolling / expanding 更新時，fine-tune window 必須嚴格落在「完全已知的 label」範圍內。
- **必須滿足**: `finetune_end < test_start`
- **絕對禁止**: `finetune_end >= test_start`

### 4. 訓練策略配置 (Training Strategies)
- `standard`: 固定歷史區間一次訓練。
- `expanding`: 使用截至目前所有可觀測歷史資料更新。
- `rolling_finetune`: 僅使用最近 N 個月可觀測資料更新。
- **注意**: `step_months` (inference window 長度) 與 `rolling_window_months` (fine-tune 歷史長度) 意義不同，不可混用。

### 5. 修改代碼後的強制驗證 (Required Verification)
Agent 產生修改或審查代碼後，必須自行執行（或提示使用者執行）以下驗證流程：
1. 確保程式或日誌中會印出邊界：`test_start`, `test_end`, `finetune_start`, `finetune_end`。
2. 插入或檢查邊界斷言：`assert finetune_end < test_start`。
3. 執行編譯測試：
   ```bash
   python -m py_compile train.py
   python -m py_compile src/backtester/backtest.py
   ```

### 6. Debug 警訊 (Leakage Warning Signs)
若使用者回報或你觀察到以下開發現象，請高度懷疑發生 Data Leakage 並立刻展開排查：
- Sharpe ratio 出現史無前例的暴增。
- 僅僅修改 window 參數，績效突然不對稱地大幅提升。
- Benchmark 對齊調整後，alpha 發生異常變化。
