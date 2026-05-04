---
name: 'write-optimized-code'
description: 撰寫符合正確性、具備良好設計原則且具備高效運算（如 Numpy、PyTorch 向量化）的程式碼，並在確保介面穩定的前提下進行整合。
---
# Write Optimized Code

當你需要實作新功能、修復 Bug、或進行重構時，請務必**綜合考量**「執行效能」、「系統設計」與「安全整合」。請嚴格遵守以下優先順序與指引：

## 1. 運算效率與效能 (Computational Efficiency) —— 【實作首要目標】
- **強制向量化 (Vectorization)**：絕對避免在陣列層級使用 Python 全域的 `for` 迴圈處理。必須優先依賴 `numpy` 或 `torch` 的矩陣與張量運算功能。
- **記憶體與設備管理**：在處理龐大資料集（如 IVS Parquet 檔或 `[Batch, 1, 10, 18]` 這類高維度張量）時，善用 in-place 操作（如 `+=`, `*=`）。在 PyTorch 中，嚴格避免不必要的 CPU/GPU 來回資料傳輸。
- **最佳資料結構**：確保查詢與比對操作使用 `set` 或 `dict`，從根本降低時間複雜度 (Big-O)。

## 2. 程式設計原則 (Design Principles & Correctness) —— 【程式碼品質】
- **單一職責與高內聚 (SRP)**：確保每個函式只負責一件事情。如果為了提升效能而導致邏輯過於複雜，必須將其抽離成獨立的 private 輔助函式。
- **DRY 與重用性**：優先尋找並重用專案內 `src/utils/` 或現成的邏輯，避免重複造輪子。
- **Pythonic 與型別防呆**：撰寫符合現代 Python 慣用法的程式碼，強制加上 Type Hints (型別提示) 並實作必要的邊界條件檢查。

## 3. 安全整合與介面穩定 (Safe Integration & API Stability) —— 【邊界約束】
- **內部重構，外部向後相容**：我們鼓勵為了提升效能與改善設計而修改現有程式碼。若需擴充介面以支援新功能，**請僅新增帶有預設值的可選參數 (Optional Parameters with Defaults)**。
- **嚴禁破壞性修改**：絕不可修改既有參數的順序、移除既有參數，或改變回傳值的型別與結構，以確保 downstream 模組（如 backtester 或 trainer）在不修改呼叫方式的前提下仍能正常運作。
- **隔離副作用**：了解現有的資料流與架構後再動手。改動應專注於達成需求，若需修改共用模組，必須確保不破壞其他依賴該模組的功能。

## 4. 自我驗證 (Self-Verification)
提交代碼前，請自行檢核以下三點：
1. Numpy/Torch 的維度 (shape) 在轉換過程中是否被正確保留且對齊？
2. 是否已經盡可能消除了 Python 層級的迴圈運算？
3. 對外暴露的介面（API）是否與修改前完全一致？
