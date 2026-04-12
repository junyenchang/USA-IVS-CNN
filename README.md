# USA IVS CNN

## Idea
* Key Paper
    * Kelly, Bryan T., Boris Kuznetsov, Semyon Malamud, Teng Andrea Xu. "Deep learning from implied volatility surfaces." Swiss Finance Institute Research Paper 2360 (2023)

## 環境與前置作業
1. **Python 環境**: 建議使用 Python 3.8 以上版本。
2. **WRDS 帳號**: 確保擁有可以存取 **OptionMetrics** 與 **CRSP** 資料庫的 Wharton Research Data Services (WRDS) 帳號。
3. **安裝套件**:
   在專案根目錄下安裝所需的相依套件：
   ```bash
   pip install -r requirements.txt
   pip install -e .
   # 將 src 作為 package 載入，方便於 notebooks 中使用
   ```

## 如何抓取資料 (Data)

專案內提供自動化的下載腳本 `download_ivs.py`，負責從 WRDS (OptionMetrics, CRSP) 抓取、清理並整合指定年份範圍內的選擇權隱含波動率曲面 (IVS) 以及對應的結算與市值資料。

### 下載步驟

1. **設定 WRDS 帳號**

   本專案使用環境變數來管理帳號資訊以確保安全性。請在專案根目錄下建立一個 `.env` 檔案，並加入以下內容以設定您的 WRDS 帳號名稱：
   ```ini
   WRDS_USERNAME=你的WRDS帳號 # ex: WRDS_USERNAME=OOXXOO
   ```
   （程式會自動從 `.env` 檔案讀取此環境變數）

2. **執行下載腳本**

   在終端機執行以下指令：
   ```bash
   python download_ivs.py
   ```
   > **提醒：** 抓取大量資料可能需要**耗費數個小時**，因此強烈建議在 `tmux` 背景 session 中執行此腳本，以避免終端機連線中斷導致下載失敗。

   如果是第一次建立連線，程式會提示你輸入 WRDS 帳號、密碼，同時需要 Duo Security 驗證。
   並建議你建立本地憑證檔案 (`.pgpass`)，後續即可自動登入不用再打密碼和驗證。

3. **資料儲存與輸出**
   - 預設會下載 `1996` 至 `2024` 年的資料。
   - 所有處理完的資料會依序被產出並儲存在專案內的 `DB/OptionDB/USA_IVS/` 目錄中。

### 使用 Notebook 探索與自訂抓取範圍
如果對於想改變抓取的資料範圍（例如：針對特定公司或條件限制），可以直接修改腳本或 `src/wrds_client.py` 內建的 SQL 指令。如果想了解內建 SQL 指令的詳細解說及邏輯運作方式，請參考 [docs/sql_query_explanation.md](docs/sql_query_explanation.md)。

在進行長時間且大規模的資料抓取前，可先開啟 `notebooks/` 資料夾進行測試：
- `download_ivs.ipynb`: 提供互動式的下載環境。您可以先在這裡逐步修改並執行過濾條件，確認有正確抓取到您想要的資料，再交由腳本進行大範圍下載。
- `ivs_view.ipynb`: 展示如何讀取與檢視已下載儲存的資料。
