---
name: 'write-optimized-code'
description: 撰寫符合正確性、最小侵入性、具備良好設計原則且具備高效運算（如 Numpy、PyTorch 向量化）的程式碼指南。
---
# Write Optimized Code

當你需要實作新功能、修復 Bug、或進行重構時，請嚴格遵守以下指引：

## 1. 最小侵入法修改 (Minimal Invasive Modification)
- **先理解上下文**：在改動前，必須先了解現有的資料流與架構（例如 IVS shape `[Batch, 1, 10, 18]`）。
- **隔離改動範圍**：專注於達成需求所需的最小必要修改，不要輕易重構與需求無關的模組。
- **維持介面穩定**：保持現有函式的 signature（包含輸入參數與回傳型別）不變，以避免破壞下游模組（例如既有的 backtester 或是 trainer 流程）。

## 2. 程式設計原則 (Design Principles & Correctness)
- **單一職責原則 (SRP)**：確保每個函式與類別只負責一件事情。複雜的邏輯應該被抽離成獨立的輔助函式。
- **DRY (Don't Repeat Yourself)**：盡量重用專案內 `src/utils/` 或現成的邏輯，避免重複造輪子。
- **Pythonic 且具防呆性**：撰寫符合 Python 慣用法的程式碼，並在必要時加上 Type Hints (型別提示) 與邊界條件檢查。

## 3. 運算效率與效能 (Computational Efficiency)
- **優先使用向量化 (Vectorization)**：絕對避免能在陣列層級解決的問題卻使用 Python 全域的 `for` 迴圈處理。大量依賴 `numpy` 或 `torch` 的矩陣與張量運算功能。
- **記憶體管理**：特別是在處理龐大資料集（如 IVS Parquet 檔）時，善用 in-place 操作（如 `+=`, `*=`）來減少不必要的記憶體配置與複製。
- **避免設備傳輸瓶頸**：在 PyTorch 的操作中，避免不必要的 CPU/GPU 來回資料傳輸。
- **選擇最佳資料結構**：確保查詢操作使用 `set` 或 `dict`，從根本降低時間複雜度 (Big-O)。

## 4. 自我驗證 (Self-Verification)
- 提交代碼前再次檢查：是否無意中修改了無關的邏輯？
- Numpy/Torch 的維度 (shape) 在轉換過程中是否被正確保留？
