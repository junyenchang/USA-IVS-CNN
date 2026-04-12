import wrds
import time
import os
import pandas as pd

class WRDSClient:
    """處理 WRDS 資料庫抓取與資料壓縮的 DataLoader"""

    def __init__(self, username: str):
        print(f"正在連線至 WRDS (使用者: {username})...")
        self.db = wrds.Connection(wrds_username=username)

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
            INNER JOIN eom_dates e ON v.secid = e.secid AND v.date = e.eom_date
            WHERE v.days != 10 AND ABS(v.delta) IN (10, 15, 20, 25, 30, 35, 40, 45, 50)
        ),
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
        ),
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
        ),
        crsp_me AS (
            SELECT m.permno, date_trunc('month', m.date) AS month_start, ABS(m.prc) * m.shrout AS me, n.exchcd, n.shrcd
            FROM crsp.msf m
            LEFT JOIN crsp.msenames n ON m.permno = n.permno AND m.date >= n.namedt AND m.date <= n.nameendt
            WHERE m.date >= '{year}-01-01' AND m.date <= '{year}-12-31' AND m.prc IS NOT NULL AND m.shrout IS NOT NULL
        ),
        nyse_bkp AS (
            SELECT month_start,
                PERCENTILE_CONT(0.20) WITHIN GROUP (ORDER BY me) AS p20,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY me) AS p50,
                PERCENTILE_CONT(0.80) WITHIN GROUP (ORDER BY me) AS p80
            FROM crsp_me
            WHERE exchcd = 1 AND shrcd IN (10, 11)
            GROUP BY month_start
        )
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
        return df

    def fetch_and_save_year(self, year: int, output_dir: str) -> None:
        """抓取指定年份資料並儲存為 Parquet, 封裝了所有例外處理與打點日誌"""
        print(f"正在抓取 {year} 年的資料...")
        start_time = time.time()
        sql_query = self._build_sql_query(year)

        try:
            print("正在執行 SQL 查詢與雲端運算...")
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

        except Exception as e:
            print(f"抓取 {year} 年資料時發生錯誤: {e}")