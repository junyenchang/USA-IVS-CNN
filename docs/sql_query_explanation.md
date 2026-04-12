# WRDS SQL 查詢指令詳細說明

詳細說明 `src/wrds_client.py` 中用來從 WRDS (Wharton Research Data Services) 抓取資料的 SQL 查詢指令。使用 CTE (Common Table Expressions，即 `WITH` 語法) 將複雜的資料處理流程拆解成多個步驟。

以下是各個 CTE 區塊的詳細說明：

## 1. `eom_dates` (計算每月底日期)
```sql
WITH eom_dates AS (
    SELECT secid, date_trunc('month', date) AS month_start, MAX(date) AS eom_date
    FROM optionm.vsurfd{year}
    WHERE date >= '{year}-01-01' AND date <= '{year}-12-31'
    GROUP BY secid, date_trunc('month', date)
)
```
- **目的**：找出每檔選擇權 (secid) 每個月的最後一個交易日 (`eom_date`)。
- **作法**：從 OptionMetrics 選擇權波動率曲面資料表 (`optionm.vsurfd{year}`) 撈取該年度資料，依照 `secid` 和月份 (`month_start`) 分組，並取該組內日期最大的值。

## 2. `filtered_ivs` (篩選選擇權隱含波動率資料)
```sql
filtered_ivs AS (
    SELECT v.secid, v.date, v.days, v.delta, v.impl_volatility
    FROM optionm.vsurfd{year} v
    INNER JOIN eom_dates e ON v.secid = e.secid AND v.date = e.eom_date
    WHERE v.days != 10 AND ABS(v.delta) IN (10, 15, 20, 25, 30, 35, 40, 45, 50)
)
```
- **目的**：過濾出需要的隱含波動率 (Implied Volatility, IV) 資料。
- **作法**：
  - 只保留每個月最後一個交易日 (`eom_date`) 的資料。
  - 排除到期天數 (`days`) 為 10 天的資料。
  - 限制 Delta 絕對值為 10, 15, ..., 50 的資料。

## 3. `raw_link` & `valid_link` & `target_permnos` (OptionMetrics 與 CRSP 的連結)
```sql
raw_link AS (
    SELECT secid, permno, sdate, edate,
        LAG(edate) OVER (PARTITION BY permno ORDER BY sdate ASC, secid ASC) AS prev_edate
    FROM wrdsapps.opcrsphist
    WHERE score = 1
),
valid_link AS (
    SELECT secid, permno, sdate, edate
    FROM raw_link
    WHERE prev_edate IS NULL OR sdate > prev_edate
),
target_permnos AS (
    SELECT DISTINCT permno
    FROM valid_link
    WHERE sdate <= '{year}-12-31' AND (edate >= '{year}-01-01' OR edate IS NULL)
)
```
- **目的**：建立 `secid` (OptionMetrics 識別碼) 與 `permno` (CRSP 識別碼) 的正確連結。
- **作法**：
  - `raw_link`: 找出匹配度最高 (`score = 1`) 的連結，並透過 `LAG` 函數取得上一筆結束日期 (`prev_edate`)，用以檢查區間重疊。
  - `valid_link`: 過濾掉日期重疊的異常資料，確保連結有效。
  - `target_permnos`: 取出該年度內有效的標的股票 `permno` 列表。

## 4. `daily_returns` & `monthly_accumulated` (計算 CRSP 月報酬率)
```sql
daily_returns AS (
    SELECT d.permno, date_trunc('month', d.date) AS month_start,
        (1 + COALESCE(d.ret, 0)) * (1 + COALESCE(dl.dlret, 0)) AS gross_ret
    FROM crsp.dsf d
    INNER JOIN target_permnos tp ON d.permno = tp.permno
    LEFT JOIN crsp.dsedelist dl ON d.permno = dl.permno AND d.date = dl.dlstdt
    WHERE d.date >= '{year}-01-01' AND d.date <= '{year}-12-31'
),
monthly_accumulated AS (
    SELECT permno, date_trunc('month', month_start)::date AS crsp_date,
        EXP(SUM(LN(GREATEST(gross_ret, 1e-10)))) - 1 AS crsp_monthly_return
    FROM daily_returns
    GROUP BY permno, month_start
)
```
- **目的**：使用 CRSP 日報酬資料 (`crsp.dsf`) 計算每月累積報酬率。
- **作法**：
  - `daily_returns`: 將日報酬 (`ret`) 與下市報酬 (`dlret`) 結合，計算出每日的毛報酬 (Gross Return)。
  - `monthly_accumulated`: 使用對數運算 `EXP(SUM(LN(...)))` 將日毛報酬連乘，最後減 1 計算出每月的總報酬率 (`crsp_monthly_return`)。

## 5. `crsp_me` (計算公司市值)
```sql
crsp_me AS (
    SELECT m.permno, date_trunc('month', m.date) AS month_start, ABS(m.prc) * m.shrout AS me, n.exchcd, n.shrcd
    FROM crsp.msf m
    LEFT JOIN crsp.msenames n ON m.permno = n.permno AND m.date >= n.namedt AND m.date <= n.nameendt
    WHERE m.date >= '{year}-01-01' AND m.date <= '{year}-12-31' AND m.prc IS NOT NULL AND m.shrout IS NOT NULL
)
```
- **目的**：計算各股票每月的市值 (Market Equity, ME)。
- **作法**：透過收盤價絕對值 (`ABS(prc)`) 乘以流通股數 (`shrout`) 計算市值。同時透過 `crsp.msenames` 取得交易所代碼 (`exchcd`) 與股票類別碼 (`shrcd`)。

## 6. `nyse_bkp` (計算 NYSE 交易所市值分界點)
```sql
nyse_bkp AS (
    SELECT month_start,
        PERCENTILE_CONT(0.20) WITHIN GROUP (ORDER BY me) AS p20,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY me) AS p50,
        PERCENTILE_CONT(0.80) WITHIN GROUP (ORDER BY me) AS p80
    FROM crsp_me
    WHERE exchcd = 1 AND shrcd IN (10, 11)
    GROUP BY month_start
)
```
- **目的**：以 NYSE 上市公司的市值建立 20%、50%、80% 的分位數（Breakpoints），用以將公司分類規模。
- **作法**：限定為 NYSE 交易所 (`exchcd = 1`) 且為普通股 (`shrcd IN (10, 11)`)，針對每月市值進行分位數的計算 (`PERCENTILE_CONT`)。

## 7. `SELECT` 主要查詢區塊 (整合所有資料與分類)
```sql
SELECT
    f.secid, l.permno, f.date AS opt_date, f.days, f.delta, f.impl_volatility,
    m.crsp_date, m.crsp_monthly_return, c_me.me AS market_cap,
    CASE
        WHEN c_me.me > bkp.p80 THEN 'Mega'
        WHEN c_me.me > bkp.p50 AND c_me.me <= bkp.p80 THEN 'Large'
        WHEN c_me.me > bkp.p20 AND c_me.me <= bkp.p50 THEN 'Small'
        WHEN c_me.me <= bkp.p20 THEN 'Micro'
        ELSE NULL
    END AS size_group,
    CASE WHEN c_me.me > bkp.p20 THEN TRUE ELSE FALSE END AS is_non_micro
FROM filtered_ivs f
INNER JOIN valid_link l ON f.secid = l.secid AND f.date >= l.sdate AND (f.date <= l.edate OR l.edate IS NULL)
LEFT JOIN monthly_accumulated m ON l.permno = m.permno AND date_trunc('month', f.date) = m.crsp_date
LEFT JOIN crsp_me c_me ON l.permno = c_me.permno AND date_trunc('month', f.date) = c_me.month_start
LEFT JOIN nyse_bkp bkp ON date_trunc('month', f.date) = bkp.month_start
```
- **目的**：將先前的暫存表彙整，並為每一筆資料標註上規模群組標籤。
- **作法**：
  - 將篩選過的隱含波動率表 (`filtered_ivs`) 透過 `valid_link` 連結至 CRSP 的 `permno`。
  - 將月報酬率 (`monthly_accumulated`) 及市值資料 (`crsp_me`) 依據 `permno` 與月份合併進來。
  - 將市值以該月份對應的 NYSE 分界點 (`nyse_bkp`) 大小分為 `Mega`、`Large`、`Small`、`Micro` 四種規模。
  - `is_non_micro`: 標記是否為大於 20% 分位數的非微型股 (Non-micro)。
