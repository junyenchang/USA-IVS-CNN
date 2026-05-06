import wrds
import time
import os
import pandas as pd

class WRDSClient:
    """處理 WRDS 資料庫抓取與資料壓縮的 DataLoader"""

    def __init__(self, username: str):
        print(f"正在連線至 WRDS (使用者: {username})...")
        self.username = username
        self.db = wrds.Connection(wrds_username=self.username)

    def _build_sql_query(self, year: int) -> str:
        """建立指定年份的 SQL 查詢字串 (私有方法)"""
        return f"""
        WITH eom_dates AS (
            SELECT secid, date_trunc('month', date) AS month_start, MAX(date) AS eom_date
            FROM optionm.vsurfd{year}
            WHERE date >= '{year}-01-01' AND date <= '{year}-12-31'
            GROUP BY secid, date_trunc('month', date)
        ),
        filtered_ivs AS (
            SELECT v.secid, v.date, v.days, v.delta, v.impl_volatility
            FROM optionm.vsurfd{year} v
            INNER JOIN eom_dates e
                ON v.secid = e.secid AND v.date = e.eom_date
            WHERE v.days != 10
            AND ABS(v.delta) IN (10, 15, 20, 25, 30, 35, 40, 45, 50)
        ),
        raw_link AS (
            SELECT secid, permno, sdate, edate,
                LAG(edate) OVER (
                    PARTITION BY permno
                    ORDER BY sdate ASC, secid ASC -- 加上 secid ASC 消除同日期的隨機性
                ) AS prev_edate
            FROM wrdsapps.opcrsphist
            WHERE score = 1
        ),
        valid_link AS (
            SELECT secid, permno, sdate, edate
            FROM raw_link
            WHERE prev_edate IS NULL OR sdate > prev_edate
        ),
        target_permnos AS (
            SELECT DISTINCT vl.permno
            FROM valid_link AS vl
            INNER JOIN crsp.msenames n
                ON vl.permno = n.permno
                -- 確保抓的是這家公司在該年度的正確屬性 (因為公司可能轉板)
                AND n.namedt <= '{year}-12-31'
                AND n.nameendt >= '{year}-01-01'
            WHERE vl.sdate <= '{year}-12-31'
            AND (vl.edate >= '{year}-01-01' OR vl.edate IS NULL)
            -- 限制為美國三大交易所普通股
            AND n.shrcd IN (10, 11) -- Ensure only common stocks
            AND n.exchcd IN (1, 2, 3) -- Ensure only NYSE, AMEX, and NASDAQ stocks
            AND n.siccd NOT IN (6770, 6799, 6722, 6726) -- Exclude financial stocks (SIC codes for banks and financial institutions)
        ),
        daily_returns AS (
            SELECT d.permno,
                date_trunc('month', d.date) AS month_start,
                (1 + COALESCE(d.ret, 0)) * (1 + COALESCE(dl.dlret, 0)) AS gross_ret
            FROM crsp.dsf d
            LEFT JOIN crsp.dsedelist dl
                ON d.permno = dl.permno AND d.date = dl.dlstdt
            WHERE d.date >= '{year}-01-01' AND d.date <= '{year}-12-31'
        ),
        monthly_accumulated AS (
            SELECT permno,
                date_trunc('month', month_start)::date AS crsp_date,
                EXP(SUM(LN(GREATEST(gross_ret, 1e-10)))) - 1 AS crsp_monthly_return
            FROM daily_returns
            GROUP BY permno, month_start
        ),
        -- 計算所有股票的月度市值與交易所資訊
        crsp_me AS (
            SELECT
                m.permno,
                date_trunc('month', m.date) AS month_start,
                ABS(m.prc) * m.shrout AS me,
                n.exchcd,
                n.shrcd
            FROM crsp.msf m
            LEFT JOIN crsp.msenames n
                ON m.permno = n.permno
                AND m.date >= n.namedt AND m.date <= n.nameendt
            WHERE m.date >= '{year}-01-01' AND m.date <= '{year}-12-31'
            AND m.prc IS NOT NULL AND m.shrout IS NOT NULL
            AND n.shrcd IN (10, 11)
            AND n.exchcd IN (1, 2, 3)
            AND n.siccd NOT IN (6770, 6799, 6722, 6726)
        ),
        -- 算出 NYSE 普通股的 20, 50, 80 分位數
        nyse_bkp AS (
            SELECT
                month_start,
                PERCENTILE_CONT(0.20) WITHIN GROUP (ORDER BY me) AS p20,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY me) AS p50,
                PERCENTILE_CONT(0.80) WITHIN GROUP (ORDER BY me) AS p80
            FROM crsp_me
            WHERE exchcd = 1 AND shrcd IN (10, 11)
            GROUP BY month_start
        )

        -- 最終整合：加上規模分組標籤
        SELECT
            f.secid,
            l.permno,
            f.date AS opt_date,
            f.days,
            f.delta,
            f.impl_volatility,
            m.crsp_date,
            m.crsp_monthly_return,
            c_me.me AS market_cap,
            c_me.shrcd,

            -- 依據市值落點給予分類標籤
            CASE
                WHEN c_me.me > bkp.p80 THEN 'Mega'
                WHEN c_me.me > bkp.p50 AND c_me.me <= bkp.p80 THEN 'Large'
                WHEN c_me.me > bkp.p20 AND c_me.me <= bkp.p50 THEN 'Small'
                WHEN c_me.me <= bkp.p20 THEN 'Micro'
                ELSE NULL
            END AS size_group,

            -- 判斷是否為非微型股 (Non-micro)
            CASE
                WHEN c_me.me > bkp.p20 THEN TRUE
                ELSE FALSE
            END AS is_non_micro

        FROM filtered_ivs f
        INNER JOIN valid_link l
            ON f.secid = l.secid
            AND f.date >= l.sdate
            AND (f.date <= l.edate OR l.edate IS NULL)
        INNER JOIN target_permnos tp -- 提早過濾掉非目標的 Permno
            ON l.permno = tp.permno
        LEFT JOIN monthly_accumulated m
            ON l.permno = m.permno
            AND date_trunc('month', f.date) = m.crsp_date
        -- 對接個別市值
        INNER JOIN crsp_me c_me
            ON l.permno = c_me.permno
            AND date_trunc('month', f.date) = c_me.month_start
        -- 對接全市場的 NYSE 分界點
        LEFT JOIN nyse_bkp bkp
            ON date_trunc('month', f.date) = bkp.month_start
        """

    def _build_sql_query_all_assets(self, year: int, start_date: str, end_date: str) -> str:
        """建立指定年份與日期區間的 SQL 查詢字串，不限制股票屬性及交易所"""
        return f"""
        WITH eom_dates AS (
            SELECT secid, date_trunc('month', date) AS month_start, MAX(date) AS eom_date
            FROM optionm.vsurfd{year}
            WHERE date >= '{start_date}' AND date <= '{end_date}'
            GROUP BY secid, date_trunc('month', date)
        ),
        filtered_ivs AS (
            SELECT v.secid, v.date, v.days, v.delta, v.impl_volatility
            FROM optionm.vsurfd{year} v
            INNER JOIN eom_dates e
                ON v.secid = e.secid AND v.date = e.eom_date
            WHERE v.days != 10
            AND ABS(v.delta) IN (10, 15, 20, 25, 30, 35, 40, 45, 50)
        ),
        raw_link AS (
            SELECT secid, permno, sdate, edate,
                LAG(edate) OVER (
                    PARTITION BY permno
                    ORDER BY sdate ASC, secid ASC
                ) AS prev_edate
            FROM wrdsapps.opcrsphist
            WHERE score = 1
        ),
        valid_link AS (
            SELECT secid, permno, sdate, edate
            FROM raw_link
            WHERE prev_edate IS NULL OR sdate > prev_edate
        ),
        target_permnos AS (
            SELECT DISTINCT vl.permno
            FROM valid_link AS vl
            INNER JOIN crsp.msenames n
                ON vl.permno = n.permno
                AND n.namedt <= '{end_date}'
                AND n.nameendt >= '{start_date}'
            WHERE vl.sdate <= '{end_date}'
            AND (vl.edate >= '{start_date}' OR vl.edate IS NULL)
            -- 移除普通股及交易所限制
        ),
        daily_returns AS (
            SELECT d.permno,
                date_trunc('month', d.date) AS month_start,
                (1 + COALESCE(d.ret, 0)) * (1 + COALESCE(dl.dlret, 0)) AS gross_ret
            FROM crsp.dsf d
            LEFT JOIN crsp.dsedelist dl
                ON d.permno = dl.permno AND d.date = dl.dlstdt
            WHERE d.date >= '{start_date}' AND d.date <= '{end_date}'
        ),
        monthly_accumulated AS (
            SELECT permno,
                date_trunc('month', month_start)::date AS crsp_date,
                EXP(SUM(LN(GREATEST(gross_ret, 1e-10)))) - 1 AS crsp_monthly_return
            FROM daily_returns
            GROUP BY permno, month_start
        ),
        crsp_me AS (
            SELECT
                m.permno,
                date_trunc('month', m.date) AS month_start,
                ABS(m.prc) * m.shrout AS me,
                n.exchcd,
                n.shrcd,
                n.siccd
            FROM crsp.msf m
            LEFT JOIN crsp.msenames n
                ON m.permno = n.permno
                AND m.date >= n.namedt AND m.date <= n.nameendt
            WHERE m.date >= '{start_date}' AND m.date <= '{end_date}'
            AND m.prc IS NOT NULL AND m.shrout IS NOT NULL
            -- 移除普通股及交易所限制
        ),
        nyse_bkp AS (
            SELECT
                month_start,
                PERCENTILE_CONT(0.20) WITHIN GROUP (ORDER BY me) AS p20,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY me) AS p50,
                PERCENTILE_CONT(0.80) WITHIN GROUP (ORDER BY me) AS p80
            FROM crsp_me
            WHERE exchcd = 1 AND shrcd IN (10, 11)
            GROUP BY month_start
        )

        SELECT
            f.secid,
            l.permno,
            f.date AS opt_date,
            f.days,
            f.delta,
            f.impl_volatility,
            m.crsp_date,
            m.crsp_monthly_return,
            c_me.me AS market_cap,
            c_me.shrcd,
            c_me.exchcd,
            c_me.siccd,

            CASE
                WHEN c_me.me > bkp.p80 THEN 'Mega'
                WHEN c_me.me > bkp.p50 AND c_me.me <= bkp.p80 THEN 'Large'
                WHEN c_me.me > bkp.p20 AND c_me.me <= bkp.p50 THEN 'Small'
                WHEN c_me.me <= bkp.p20 THEN 'Micro'
                ELSE NULL
            END AS size_group,

            CASE
                WHEN c_me.me > bkp.p20 THEN TRUE
                ELSE FALSE
            END AS is_non_micro

        FROM filtered_ivs f
        INNER JOIN valid_link l
            ON f.secid = l.secid
            AND f.date >= l.sdate
            AND (f.date <= l.edate OR l.edate IS NULL)
        INNER JOIN target_permnos tp
            ON l.permno = tp.permno
        LEFT JOIN monthly_accumulated m
            ON l.permno = m.permno
            AND date_trunc('month', f.date) = m.crsp_date
        INNER JOIN crsp_me c_me
            ON l.permno = c_me.permno
            AND date_trunc('month', f.date) = c_me.month_start
        LEFT JOIN nyse_bkp bkp
            ON date_trunc('month', f.date) = bkp.month_start
        """

    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """優化 DataFrame 的記憶體與型別 (私有方法)"""
        df['secid'] = df['secid'].astype('Int32')
        df['permno'] = df['permno'].astype('Int32')
        df['days'] = df['days'].astype('Int16')
        df['market_cap'] = df['market_cap'].astype('float32')
        df['delta'] = df['delta'].round().astype('Int16')
        df['size_group'] = df['size_group'].astype('category')
        df['is_non_micro'] = df['is_non_micro'].astype('bool')
        df['impl_volatility'] = df['impl_volatility'].astype('float32')
        df['crsp_monthly_return'] = df['crsp_monthly_return'].astype('float32')
        if 'shrcd' in df.columns:
            df['shrcd'] = df['shrcd'].astype('Int16')
        if 'exchcd' in df.columns:
            df['exchcd'] = df['exchcd'].astype('Int16')
        if 'siccd' in df.columns:
            df['siccd'] = df['siccd'].astype('Int32')
        return df

    def fetch_and_save_year(self, year: int, output_dir: str) -> None:
        """抓取指定年份資料並儲存為 Parquet, 封裝了所有例外處理與打點日誌"""
        print(f"正在抓取 {year} 年的資料...")
        start_time = time.time()
        sql_query = self._build_sql_query(year)
        max_retries = 5

        for attempt in range(max_retries):
            try:
                print(f"正在執行 SQL 查詢與雲端運算... (第 {attempt + 1} 次嘗試)")
                query_start = time.time()
                df = self.db.raw_sql(sql_query, date_cols=['opt_date', 'crsp_date'])
                print(f"SQL 查詢完成 耗時: {time.time() - query_start:.2f} 秒。共抓取 {len(df)} 筆資料。")

                if not df.empty:
                    print("正在進行資料型別壓縮...")
                    df = self._optimize_dataframe(df)

                    # 儲存檔案
                    file_name = os.path.join(output_dir, f'option_ivs_crsp_{year}.parquet')
                    df.to_parquet(file_name, engine='pyarrow')

                    # 結算與列印
                    total_time = time.time() - start_time
                    file_size_mb = os.path.getsize(file_name) / (1024 * 1024)

                    print("-" * 40)
                    print(f"▶ 總結 {year} 年: 耗時 {total_time/60 :.2f} 分鐘 | {len(df):,} 筆 | {file_size_mb:.2f} MB")
                    print("-" * 40)
                else:
                    print(f"測試區間 {year} 內沒有抓取到符合條件的資料。")

                break # 成功執行，跳出重試迴圈

            except Exception as e:
                print(f"抓取 {year} 年資料時發生錯誤: {e}")
                if attempt < max_retries - 1:
                    print("嘗試重置 WRDS 連線並重新抓取...")
                    try:
                        self.db.close()
                    except:
                        pass

                    time.sleep(5)
                    try:
                        self.db = wrds.Connection(wrds_username=self.username)
                        print("WRDS 連線已成功重置")
                    except Exception as re_e:
                        print(f"重置連線失敗，請檢查網路狀態: {re_e}")
                else:
                    print(f"已達到最大重試次數 ({max_retries})，放棄抓取 {year} 年資料。")

    def fetch_and_save_year_all_assets(self, year: int, output_dir: str) -> None:
        """抓取指定年份的所有資產資料並儲存為 Parquet, 不做產業與交易所過濾"""
        start_time = time.time()

        if year >= 2021:
            print(f"[{year} 年] 進入大資料年份，自動拆分為半年分別抓取...")
            segments = [
                (f"{year}-01-01", f"{year}-06-30"),
                (f"{year}-07-01", f"{year}-12-31")
            ]
        else:
            print(f"正在抓取 {year} 年所有資產的資料 (無限制)...")
            segments = [(f"{year}-01-01", f"{year}-12-31")]

        max_retries = 3
        df_list = []

        for start_date, end_date in segments:
            success = False
            for attempt in range(max_retries):
                try:
                    print(f"執行 SQL 查詢 ({start_date} 至 {end_date}) ... (第 {attempt + 1} 次嘗試)")
                    sql_query = self._build_sql_query_all_assets(year, start_date, end_date)
                    query_start = time.time()

                    df_chunk = self.db.raw_sql(sql_query, date_cols=['opt_date', 'crsp_date'])
                    print(f"區間查詢完成 耗時: {time.time() - query_start:.2f} 秒。共抓取 {len(df_chunk)} 筆資料。")

                    if not df_chunk.empty:
                        df_list.append(df_chunk)

                    success = True
                    break

                except Exception as e:
                    error_msg = str(e).lower()
                    print(f"抓取區間 {start_date} ~ {end_date} 發生錯誤: {e}")

                    if "timed out" in error_msg and start_date == f"{year}-01-01" and end_date == f"{year}-12-31":
                        print("檢測到 Timeout 超時，切換為每半年抓取策略...")
                        segments = [
                            (f"{year}-01-01", f"{year}-06-30"),
                            (f"{year}-07-01", f"{year}-12-31")
                        ]
                        df_list = []
                        # Recursively call with the new segments by simulating a re-run logic
                        # But since we are inside segments loop, changing segments list while iterating is tricky.
                        # For simplicity, we can let it fail the current loop and we should ideally handle it outside.

                    # Alternative to nested loops for retry: just re-assign and break outer loop?
                    # Since we modify it in place, it's better to break and restart. But let's keep it simple:
                    if "timed out" in error_msg and start_date == f"{year}-01-01" and end_date == f"{year}-12-31":
                        break # Let's handle the downgrade logic slightly cleaner

                    if attempt < max_retries - 1:
                        print("重置連線中...")
                        try:
                            self.db.close()
                        except: pass
                        time.sleep(5)
                        try:
                            self.db = wrds.Connection(wrds_username=self.username)
                        except: pass
                    else:
                        print(f"區間 {start_date} ~ {end_date} 達最大重試次數，放棄。")

            if not success and start_date == f"{year}-01-01" and end_date == f"{year}-12-31" and len(segments) == 1:
                # Fallback to half-year logic and run loops manually
                print("每半年抓取一次...")
                df_list = []
                segments = [
                    (f"{year}-01-01", f"{year}-06-30"),
                    (f"{year}-07-01", f"{year}-12-31")
                ]
                for sd, ed in segments:
                    sub_success = False
                    for sub_attempt in range(max_retries):
                        try:
                            print(f"執行 SQL 查詢 ({sd} 至 {ed}) ... (第 {sub_attempt + 1} 次嘗試)")
                            sql_query = self._build_sql_query_all_assets(year, sd, ed)
                            query_start = time.time()
                            df_chunk = self.db.raw_sql(sql_query, date_cols=['opt_date', 'crsp_date'])
                            print(f"區間查詢完成 耗時: {time.time() - query_start:.2f} 秒。共抓取 {len(df_chunk)} 筆資料。")
                            if not df_chunk.empty: df_list.append(df_chunk)
                            sub_success = True
                            break
                        except Exception as sub_e:
                            print(f"抓取區間 {sd} ~ {ed} 發生錯誤: {sub_e}")
                            if sub_attempt < max_retries - 1:
                                try: self.db.close()
                                except: pass
                                time.sleep(5)
                                try: self.db = wrds.Connection(wrds_username=self.username)
                                except: pass
                    if not sub_success:
                        print(f"區間資料抓取失敗，中止 {year} 抓取進度")
                        return
                success = True

            if not success:
                 print(f"區間資料抓取失敗，中止 {year} 抓取進度")
                 return

        if df_list:
            print("正在合併與進行資料型別壓縮...")
            final_df = pd.concat(df_list, ignore_index=True)
            final_df = self._optimize_dataframe(final_df)

            file_name = os.path.join(output_dir, f'option_ivs_crsp_{year}.parquet')
            final_df.to_parquet(file_name, engine='pyarrow')

            total_time = time.time() - start_time
            file_size_mb = os.path.getsize(file_name) / (1024 * 1024)

            print("-" * 40)
            print(f"▶ 總結 {year} 年: 耗時 {total_time/60 :.2f} 分鐘 | {len(final_df):,} 筆 | {file_size_mb:.2f} MB")
            print("-" * 40)
        else:
            print(f"測試區間 {year} 內沒有抓取到符合條件的資料。")

    def fetch_spy_benchmark(self, output_dir: str, start_year: int=1996, end_year: int=2025):
        """抓取 SPY ETF 作為回測基準 (月報酬率)"""
        print(f"正在抓取 SPY (PERMNO: 84398) 由 {start_year} 至 {end_year} 的月度資料...")

        sql_query = f"""
        SELECT
            date_trunc('month', date)::date AS date,
            ret AS spy_ret
        FROM crsp.msf
        WHERE permno = 84398
        AND date >= '{start_year}-01-01'
        AND date <= '{end_year}-12-31'
        ORDER BY date
        """

        df = self.db.raw_sql(sql_query, date_cols=['date'])

        if not df.empty:
            file_name = os.path.join(output_dir, 'spy_benchmark_monthly.parquet')
            df.to_parquet(file_name, engine='pyarrow')
            print(f"SPY 基準資料已儲存至 {file_name}")
        else:
            print("未抓取到 SPY 資料。")

    def fetch_rf_rate(self, output_dir: str, start_year: int=1996, end_year: int=2025):
        """抓取 Fama-French 的月度無風險利率"""
        print(f"正在抓取 Fama-French 的月度無風險利率 (RF) 由 {start_year} 至 {end_year}...")

        sql_query = f"""
            SELECT date, rf
            FROM ff.factors_monthly
            WHERE date >= '{start_year}-01-01' AND date <= '{end_year}-12-31'
        """

        df = self.db.raw_sql(sql_query, date_cols=['date'])

        if not df.empty:
            file_name = os.path.join(output_dir, 'fama_french_rf_monthly.parquet')
            df.to_parquet(file_name, engine='pyarrow')
            print(f"Fama-French 無風險利率資料已儲存至 {file_name}")
        else:
            print("未抓取到 Fama-French 無風險利率資料。")

