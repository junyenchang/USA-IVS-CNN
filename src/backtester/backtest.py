import os
import typing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.path import OptionPath

def compute_long_short_weights(
    long_cond: pd.Series,
    short_cond: pd.Series,
    weighting_scheme: str = "equal",
    market_cap: typing.Optional[pd.Series] = None,
    cap_threshold: typing.Optional[float] = None,
) -> pd.Series:
    """
    依據 long_cond / short_cond 布林遮罩計算 target_weight。

    Args:
        long_cond, short_cond: 布林 Series，index 須與要輸出的權重一致
        weighting_scheme: "equal"（等權重）或 "value"（封頂市值加權）
        market_cap: weighting_scheme="value" 時，該組股票的市值 (與 long_cond 同 index)
        cap_threshold: 當月 NYSE 80% 分位數門檻值
    """
    weights = pd.Series(0.0, index=long_cond.index)

    effective_scheme = weighting_scheme
    if effective_scheme == "value":
        if market_cap is None or cap_threshold is None or pd.isna(cap_threshold):
            print("[警告] 缺少 market_cap 或當月 threshold, 該期退回等權重計算")
            effective_scheme = "equal"

    if effective_scheme == "value":
        if market_cap is not None:
            capped_mcap = market_cap.fillna(0.0).clip(upper=cap_threshold)
        long_w = capped_mcap.where(long_cond, 0.0)
        short_w = capped_mcap.where(short_cond, 0.0)

        long_sum = long_w.sum()
        short_sum = short_w.sum()

        if long_sum > 0:
            weights.loc[long_cond] = long_w[long_cond] / long_sum
        if short_sum > 0:
            weights.loc[short_cond] = -(short_w[short_cond] / short_sum)
    else:
        n_long = long_cond.sum()
        n_short = short_cond.sum()
        if n_long > 0:
            weights.loc[long_cond] = 1.0 / n_long
        if n_short > 0:
            weights.loc[short_cond] = -1.0 / n_short

    return weights

class BacktestEngine:
    def __init__(self, preds_df: pd.DataFrame, stock_info_dir: str, base_fee_bps: int = 10, task_type: str = "regression", jump_threshold: float = 0.0, ls_quantile: int = 10, weighting_method: str = "equal", nyse_breakpoint_path: typing.Optional[str] = None, nyse_breakpoint_date_col: str = "date", nyse_breakpoint_value_col: str = "cap80"):
        """
        Args:
            preds_df: 模型輸出的 ensemble_predictions.csv
            stock_info_dir: 股票資訊資料夾路徑
            base_fee_bps: 基礎手續費 (例如 10 bps = 0.001)
            task_type: 任務類型 (regression/classification)
            jump_threshold: classification 任務中的跳躍閾值
            ls_quantile: 平分成幾組 (預設 10 組，即看前 10% 與後 10%)
            weighting_method: 權重分配方法 (equal/value)
        """
        self.base_fee = base_fee_bps / 10000.0
        self.task_type = task_type
        self.jump_threshold = jump_threshold
        self.ls_quantile = ls_quantile
        self.weighting_method = weighting_method
        self.actual_col = 'ActualRaw' if 'ActualRaw' in preds_df.columns else 'Actual'

        self.nyse_breakpoints: typing.Optional[pd.Series] = None
        if self.weighting_method == "value":
            if nyse_breakpoint_path is None:
                raise ValueError("weighting_method='value' 必須提供 nyse_breakpoint_path")
            bp_df = pd.read_parquet(nyse_breakpoint_path)
            bp_df['YearMonth'] = pd.to_datetime(bp_df[nyse_breakpoint_date_col]).dt.to_period('M')
            # 用 Series 存起來，索引是 YearMonth，方便 group.name 查表
            self.nyse_breakpoints = bp_df.set_index('YearMonth')[nyse_breakpoint_value_col]

        market_df = pd.read_parquet(os.path.join(stock_info_dir, 'market_metadata.parquet'))
        self.df = self._prepare_data(preds_df, market_df)

    def _prepare_data(self, preds_df: pd.DataFrame, market_data_df: pd.DataFrame) -> pd.DataFrame:
        """合併預測結果與市值特徵，並計算多空權重"""
        preds_df['Date'] = pd.to_datetime(preds_df['Date'])
        df = pd.merge(preds_df, market_data_df, on=['Date', 'Permno'], how='left')

        df['Date'] = pd.to_datetime(df['Date'])
        df['YearMonth'] = df['Date'].dt.to_period('M')
        df = df.sort_values(['YearMonth', 'Date', 'Permno'])
        df = df.drop_duplicates(subset=['YearMonth', 'Permno'], keep='last')

        # 2. 分組計算 Quantile 決定 Long/Short 部位 (每月月底)
        def assign_weights(group: pd.DataFrame) -> pd.DataFrame:
            valid_preds = group['Pred'].dropna()
            if len(valid_preds) < self.ls_quantile:
                group['target_weight'] = 0
                return group

            low_pct = 1.0 / self.ls_quantile
            high_pct = 1.0 - low_pct

            q_low = valid_preds.quantile(low_pct)
            q_high = valid_preds.quantile(high_pct)

            # 給予權重 (Equal Weight，多空各 100% 資金)
            if self.task_type == "classification" and self.jump_threshold < 0:
                # 預測大於負向 jump 跌幅的機率時：機率越高代表越可能大跌
                # 買入 (Long) 機率最低的群組，做空 (Short) 機率最高的群組
                long_cond = group['Pred'] <= q_low
                short_cond = group['Pred'] >= q_high
            else:
                # Regression 或正向 Jump
                long_cond = group['Pred'] >= q_high
                short_cond = group['Pred'] <= q_low

            cap_threshold = np.nan
            if self.weighting_method == "capped_value" and self.nyse_breakpoints is not None:
                cap_threshold = self.nyse_breakpoints.get(group.name, np.nan) # type: ignore

            group['target_weight'] = compute_long_short_weights(
                long_cond, short_cond,
                weighting_scheme=self.weighting_method,
                market_cap=group.get('market_cap'),
                cap_threshold=cap_threshold,
            )

            return group

        if 'target_weight' not in df.columns:
            df = pd.DataFrame(df.groupby('YearMonth').apply(assign_weights).reset_index(drop=True))
        else:
            print("Data already contains 'target_weight' column, skipping weight assignment.")
        return df

    def run_simulation(self) -> pd.DataFrame:
        """執行跨期的每月回測計算"""
        months = sorted(self.df['YearMonth'].unique())

        results = []
        detailed_records = [] # 記錄每個月的個股明細
        drifted_weights = pd.Series(dtype=float) # 記錄上一期的結算漂移權重

        for t_month in months:
            current_month = pd.DataFrame(self.df[self.df['YearMonth'] == t_month].set_index('Permno'))

            # --- 1. 目標權重與實際報酬 ---
            w_target = current_month['target_weight'].fillna(0)
            returns = current_month[self.actual_col].fillna(0)
            is_microcap = current_month['is_microcap'].fillna(False).astype(bool)

            all_assets = w_target.index.union(drifted_weights.index)
            w_target = w_target.reindex(all_assets, fill_value=0.0)
            w_drifted = drifted_weights.reindex(all_assets, fill_value=0.0)
            returns = returns.reindex(all_assets, fill_value=0.0)
            is_microcap = is_microcap.reindex(all_assets, fill_value=False)

            # --- 2. 計算交易手續費 (TC_t) ---
            turnover = (w_target - w_drifted).abs()
            fee_multiplier = np.where(is_microcap, 2.0, 1.0) # 微型股手續費加倍
            transaction_costs = turnover * self.base_fee * fee_multiplier  # 這是每一檔股票的 TC Detail
            total_tc = transaction_costs.sum()

            # --- 3. 計算放空成本 (SC_t) ---
            short_positions = w_target[w_target < 0].abs()
            short_fee_rate = np.where(is_microcap.loc[short_positions.index], 0.048, 0.025) / 12.0
            shorting_costs = pd.Series(0.0, index=all_assets) # 初始化為 0
            shorting_costs.loc[short_positions.index] = short_positions * short_fee_rate # 這是每一檔股票的 SC Detail
            total_sc = shorting_costs.sum()

            # --- 4. 計算原始報酬與淨報酬 ---
            raw_return = (w_target * returns).sum()
            net_return = raw_return - total_tc - total_sc

            # 只儲存「目標有持倉」或「產生了交易換手」的股票，濾掉完全沒動到的股票
            active_mask = (w_target != 0) | (turnover != 0)
            active_assets = all_assets[active_mask]

            month_detail = pd.DataFrame({
                'Date': t_month + 1,  # 持倉在 t_month 月底決定，報酬在 t_month+1 月底實現，這樣才能跟 benchmark 的月報酬正確比較
                'Permno': active_assets,
                'Weight': w_target.loc[active_assets],
                'Return': returns.loc[active_assets],
                'Turnover': turnover.loc[active_assets],
                'TC_Fee': transaction_costs.loc[active_assets],
                'SC_Fee': shorting_costs.loc[active_assets],
                'Is_Microcap': is_microcap.loc[active_assets]
            })
            detailed_records.append(month_detail)

            # --- 5. 計算月底的漂移權重 ---
            end_of_month_values = w_target * (1 + returns)
            portfolio_multiplier = 1 + raw_return
            if portfolio_multiplier != 0:
                drifted_weights = end_of_month_values / portfolio_multiplier
            else:
                drifted_weights = pd.Series(0, index=all_assets)

            results.append({
                'Date': t_month + 1,  # 報酬在 t_month+1 月底實現
                'Raw_Return': raw_return,
                'Turnover': turnover.sum() / 2,
                'TC': total_tc,
                'SC': total_sc,
                'Net_Return': net_return
            })

        self.holdings_detail: pd.DataFrame = pd.concat(detailed_records, ignore_index=True)

        return pd.DataFrame(results)

    def save_holdings_report(self, save_folder: str):
        """將每月的持倉與明細匯出成 CSV"""
        if hasattr(self, 'holdings_detail'):
            save_path = os.path.join(save_folder, "holdings_detail.csv")
            self.holdings_detail['Date'] = self.holdings_detail['Date'].astype(str)
            self.holdings_detail.to_csv(save_path, index=False)
            print(f"Save holdings detail report to: {save_path}")
        else:
            print("尚未執行回測，無法儲存持倉報表。")

    @staticmethod
    def calculate_metrics(portfolio_df: pd.DataFrame, save=True, save_path: str = "backtest_metrics.txt", rf_path: str = "fama_french_rf_monthly.parquet") -> typing.Dict[str, str]:
        """計算績效指標"""
        df = portfolio_df.copy()
        rf_rate_df = pd.read_parquet(rf_path)
        df['YearMonth'] = pd.to_datetime(df['Date'].astype(str)).dt.to_period('M')
        rf_rate_df['YearMonth'] = pd.to_datetime(rf_rate_df['date'].astype(str)).dt.to_period('M')
        merged = pd.merge(df, rf_rate_df[['YearMonth', 'rf']], on='YearMonth', how='inner')

        excess_return = merged['Net_Return'] - merged['rf']

        annualized_return = merged['Net_Return'].mean() * 12
        annualized_vol = merged['Net_Return'].std() * np.sqrt(12)

        annualized_excess_return = excess_return.mean() * 12
        excess_vol = excess_return.std() * np.sqrt(12)
        sharpe_ratio = annualized_excess_return / excess_vol if excess_vol != 0 else 0

        cumulative_return = (1 + merged['Net_Return']).cumprod()
        running_max = cumulative_return.cummax()
        drawdown = (cumulative_return - running_max) / running_max
        max_drawdown = drawdown.min()

        downside_diff = np.minimum(0, merged['Net_Return'] - merged['rf'])
        annualized_downside_vol = np.sqrt((downside_diff ** 2).mean()) * np.sqrt(12)
        sortino_ratio = annualized_excess_return / annualized_downside_vol if annualized_downside_vol != 0 else 0

        metrics = {
            "Annualized Return": f"{annualized_return:.2%}",
            "Annualized Volatility": f"{annualized_vol:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Sortino Ratio": f"{sortino_ratio:.2f}"
        }

        if save:
            with open(save_path, "w", encoding="utf-8") as f:
                for k, v in metrics.items():
                    line = f"{k}: {v}"
                    f.write(line + "\n")

        return metrics

    @staticmethod
    def save_and_plot_performance(portfolio_df: pd.DataFrame, benchmark_path: str, save_folder: str, label: str = "CNN IVS Strategy"):
        """
        1. 算出累積報酬並與 SPY 比較畫圖
        2. 將結果儲存進原本的實驗資料夾
        """
        portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'].astype(str))

        spy_df = pd.read_parquet(benchmark_path)
        spy_df['Date'] = pd.to_datetime(spy_df['date'])
        spy_df['YearMonth'] = spy_df['Date'].dt.to_period('M')
        spy_df = spy_df.rename(columns={'spy_ret': 'SPY_Return'})

        # 用 YearMonth 去除重複對齊 (如果 SPY 也是日資料，則轉月資料)
        spy_monthly = spy_df.groupby('YearMonth')['SPY_Return'].last().reset_index()
        portfolio_df['YearMonth'] = portfolio_df['Date'].dt.to_period('M')
        merged_df = pd.merge(portfolio_df, spy_monthly, on='YearMonth', how='left')
        merged_df['SPY_Return'] = merged_df['SPY_Return'].fillna(0.0)

        # 計算累積績效曲線 (Wealth Index)
        # 起始資金為 1 的話: Cumulative = (1 + return).cumprod()
        merged_df['Strategy_CumRet'] = (1 + merged_df['Net_Return']).cumprod()
        merged_df['SPY_CumRet'] = (1 + merged_df['SPY_Return']).cumprod()

        plt.figure(figsize=(10, 6))
        plt.plot(merged_df['Date'], merged_df['Strategy_CumRet'], label=label, color='red')
        plt.plot(merged_df['Date'], merged_df['SPY_CumRet'], label='SPY Benchmark', color='blue', alpha=0.7)

        plt.title('Cumulative Performance vs Benchmark')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(save_folder, "cumulative_performance.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

        csv_path = os.path.join(save_folder, "backtest_timeseries.csv")
        merged_df.drop(columns=['YearMonth']).to_csv(csv_path, index=False)

        print(f"回測圖表與時序結果已儲存至: {save_folder}")
        return merged_df

    @staticmethod
    def save_decile_analysis(preds_df: pd.DataFrame, save_folder: str):
        """
        依據預測值分 10 組，計算每一組的每月 Pred 與 Actual 平均值
        """
        actual_col = 'ActualRaw' if 'ActualRaw' in preds_df.columns else 'Actual'
        df_clean = preds_df.dropna(subset=['Pred', actual_col]).copy()
        df_clean['Date'] = pd.to_datetime(df_clean['Date'])
        df_clean['Decile'] = df_clean.groupby('Date')['Pred'].transform(
            lambda x: pd.qcut(x, 10, labels=False, duplicates='drop') + 1
        )
        monthly_decile_stats = df_clean.groupby(['Date', 'Decile'])[[actual_col, 'Pred']].mean().reset_index()
        monthly_decile_stats.to_csv(os.path.join(save_folder, "monthly_decile_returns.csv"), index=False)
        summary_df = monthly_decile_stats.groupby('Decile')[['Pred', actual_col]].mean()
        summary_df.columns = ['Mean_Predicted_Value', 'Mean_Actual_Return']

        top_decile = summary_df.index.max()
        bottom_decile = summary_df.index.min()

        # Convert to dictionary to avoid Pandas Scalar type issues in Pylance/Pyright when doing arithmetic operations on the summary statistics
        summary_dict = summary_df.to_dict(orient='index')
        spread_pred = summary_dict[top_decile]['Mean_Predicted_Value'] - summary_dict[bottom_decile]['Mean_Predicted_Value']
        spread_actual = summary_dict[top_decile]['Mean_Actual_Return'] - summary_dict[bottom_decile]['Mean_Actual_Return']

        spread_row = pd.DataFrame(
            [[spread_pred, spread_actual]],
            columns=summary_df.columns,
            index=['Decile_10_minus_1']
        )
        summary_df = pd.concat([summary_df, spread_row])
        summary_csv_path = os.path.join(save_folder, "decile_summary.csv")
        summary_df.to_csv(summary_csv_path)
        print(f"\n=== 10分組分析摘要已儲存至: {summary_csv_path} ===")
        print(summary_df)

        pivot_returns = monthly_decile_stats.pivot(index='Date', columns='Decile', values=actual_col)

        plt.figure(figsize=(12, 6))
        pivot_returns.plot(ax=plt.gca(), cmap='coolwarm', alpha=0.85, linewidth=1.5)

        plt.title("Equal Weight Monthly Return by Prediction Decile", fontsize=14, fontweight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Monthly Return", fontsize=12)
        plt.axhline(0, color='gray', linestyle='--', alpha=0.5)

        plt.legend(title="Decile", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(save_folder, "decile_monthly_returns.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

def calculate_size_sharpe_with_costs(folder: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdings = pd.read_csv(os.path.join(folder, "holdings_detail.csv"))
    meta = pd.read_parquet(
        os.path.join(OptionPath.StockInfo_All, "market_metadata.parquet")
    )[["Date", "Permno", "market_cap", "size_group", "is_microcap"]].copy()

    # holdings_detail 的 Date 是實現報酬月，所以要往前移一個月，對齊建倉月份
    holdings["Date"] = pd.to_datetime(holdings["Date"]).dt.to_period("M")
    holdings["SignalMonth"] = holdings["Date"] - 1

    meta["Date"] = pd.to_datetime(meta["Date"]).dt.to_period("M")
    meta = meta.rename(columns={"Date": "SignalMonth"})

    df = holdings.merge(meta, on=["SignalMonth", "Permno"], how="left")
    df["size_group"] = df["size_group"]

    for col in ["Weight", "Return", "TC_Fee", "SC_Fee"]:
        df[col] = df[col].fillna(0.0)

    def monthly_bucket_pnl(group: pd.DataFrame) -> pd.Series:
        gross_return = (group["Weight"] * group["Return"]).sum()
        tc = group["TC_Fee"].sum()
        sc = group["SC_Fee"].sum()
        net_return = gross_return - tc - sc

        return pd.Series(
            {
                "Gross_Return": gross_return,
                "TC": tc,
                "SC": sc,
                "Net_Return": net_return,
                "N": len(group),
            }
        )

    monthly_bucket = (
        df.groupby(["Date", "size_group"], observed=True)
          .apply(monthly_bucket_pnl)
          .reset_index()
    )

    rf = pd.read_parquet(
        os.path.join(OptionPath.RFrate, "fama_french_rf_monthly.parquet")
    )[["date", "rf"]].copy()
    rf["Date"] = pd.to_datetime(rf["date"]).dt.to_period("M")
    rf = rf[["Date", "rf"]]

    monthly_bucket = monthly_bucket.merge(rf, on="Date", how="left")
    monthly_bucket["Excess"] = monthly_bucket["Net_Return"] - monthly_bucket["rf"]

    def sharpe_ratio(x: pd.DataFrame) -> float:
        excess = x["Excess"].dropna()
        if len(excess) < 2:
            return np.nan
        std = excess.std(ddof=1)
        if std == 0 or np.isnan(std):
            return np.nan
        return excess.mean() / std * np.sqrt(12)

    summary = (
        monthly_bucket.groupby("size_group")
        .apply(lambda x: pd.Series({
            "Mean_Net_Return": x["Net_Return"].mean(),
            "Mean_Excess": x["Excess"].mean(),
            # "Vol_Excess": x["Excess"].std(ddof=1),
            "Sharpe": sharpe_ratio(x),
        }))
        .sort_values("Sharpe", ascending=False)
    )

    return summary, monthly_bucket

def build_portfolio_intersection(df_ret: pd.DataFrame, df_prod: pd.DataFrame, prob_is_risk: bool, weighting_method: str="equal", market_cap_df: typing.Optional[pd.DataFrame] = None, nyse_breakpoints: typing.Optional[pd.Series] = None, target_pct: float = 0.1) -> pd.DataFrame:
    """
    交集法建構投資組合 (Intersection Blending)
    邏輯:
      - Long: 報酬預測為前 10% 且 大跌機率為最低 10% 的股票
      - Short: 報酬預測為最後 10% 且 大跌機率為最高 10% 的股票
    """
    df_ret = df_ret[['Date', 'Permno', 'Pred', 'Actual', 'ActualRaw']].rename(columns={'Pred': 'Pred_Ret'})
    df_prod = df_prod[['Date', 'Permno', 'Pred']].rename(columns={'Pred': 'Pred_Prob'})

    df = pd.merge(df_ret, df_prod, on=['Date', 'Permno'], how='inner')
    df['YearMonth'] = pd.to_datetime(df['Date']).dt.to_period('M')

    def calc_weights(group):
        group['target_weight'] = 0.0
        if len(group) < 10:
            return group

        # --- 訊號 1: 報酬預測 (越高越好) ---
        ret_q10 = group['Pred_Ret'].quantile(target_pct)
        ret_q90 = group['Pred_Ret'].quantile(1 - target_pct)

        ret_long_cond = group['Pred_Ret'] >= ret_q90
        ret_short_cond = group['Pred_Ret'] <= ret_q10

        # --- 訊號 2: 機率預測 (根據 prob_is_risk 切換邏輯) ---
        prob_q_high = group['Pred_Prob'].quantile(1 - target_pct)
        prob_q_low = group['Pred_Prob'].quantile(target_pct)
        if prob_is_risk:
            # 風險模型：低機率做多，高機率做空
            prob_long_cond = group['Pred_Prob'] <= prob_q_low
            prob_short_cond = group['Pred_Prob'] >= prob_q_high
        else:
            # 獲利/分類模型：高機率做多，低機率做空
            prob_long_cond = group['Pred_Prob'] >= prob_q_high
            prob_short_cond = group['Pred_Prob'] <= prob_q_low

        final_long_cond = ret_long_cond & prob_long_cond
        final_short_cond = ret_short_cond & prob_short_cond

        cap_threshold = np.nan
        if weighting_method == "value" and nyse_breakpoints is not None:
            cap_threshold = nyse_breakpoints.get(group.name, np.nan)

        group['target_weight'] = compute_long_short_weights(
            final_long_cond, final_short_cond,
            weighting_scheme=weighting_method,
            market_cap=group.get('market_cap'),
            cap_threshold=cap_threshold,
        )
        return group

    return pd.DataFrame(df.groupby('YearMonth').apply(calc_weights).reset_index(drop=True))

def build_portfolio_risk_overlay(df_ret: pd.DataFrame, df_risk: pd.DataFrame, risk_exclude_pct: float = 0.20, target_pct: float = 0.10) -> pd.DataFrame:
    df_ret = df_ret[['Date', 'Permno', 'Pred', 'Actual', 'ActualRaw']].rename(columns={'Pred': 'Pred_Ret'})
    df_risk = df_risk[['Date', 'Permno', 'Pred']].rename(columns={'Pred': 'Pred_Risk'})

    df = pd.merge(df_ret, df_risk, on=['Date', 'Permno'], how='inner')
    df['YearMonth'] = pd.to_datetime(df['Date']).dt.to_period('M')

    def calc_weights(group):
        group['target_weight'] = 0.0
        n_stocks = len(group)
        if n_stocks < 10:
            return group

        # 找出危險名單：大跌機率最高的前 risk_exclude_pct (例: 20%)
        risk_threshold = group['Pred_Risk'].quantile(1 - risk_exclude_pct)
        is_safe = group['Pred_Risk'] <= risk_threshold

        # 安全池中的股票，才能做多 (依報酬預測排序)
        safe_pool = group[is_safe]
        if len(safe_pool) > 0:
            long_thresh = safe_pool['Pred_Ret'].quantile(1 - target_pct)
            long_cond = (group.index.isin(safe_pool.index)) & (group['Pred_Ret'] >= long_thresh)
        else:
            long_cond = pd.Series(False, index=group.index)

        # 全市場中，報酬預測最差的做空
        short_thresh = group['Pred_Ret'].quantile(target_pct)
        short_cond = group['Pred_Ret'] <= short_thresh

        n_long = long_cond.sum()
        n_short = short_cond.sum()

        if n_long > 0:
            group.loc[long_cond, 'target_weight'] = 1.0 / n_long
        if n_short > 0:
            group.loc[short_cond, 'target_weight'] = -1.0 / n_short

        return group

    return pd.DataFrame(df.groupby('YearMonth').apply(calc_weights).reset_index(drop=True))

def build_portfolio_zscore_blending(df_ret: pd.DataFrame, df_prob: pd.DataFrame, prob_is_risk: bool, weight_ret: float = 0.5, target_pct: float = 0.10) -> pd.DataFrame:
    df_ret = df_ret[['Date', 'Permno', 'Pred', 'Actual', 'ActualRaw']].rename(columns={'Pred': 'Pred_Ret'})
    df_prob = df_prob[['Date', 'Permno', 'Pred']].rename(columns={'Pred': 'Pred_Prob'})

    df = pd.merge(df_ret, df_prob, on=['Date', 'Permno'], how='inner')
    df['YearMonth'] = pd.to_datetime(df['Date']).dt.to_period('M')

    def calc_weights(group):
        group['target_weight'] = 0.0
        if len(group) < 10:
            return group

        # 將預測值轉換為橫斷面排名 (0 到 1 之間)
        rank_ret = group['Pred_Ret'].rank(pct=True)
        if prob_is_risk:
            rank_safety = group['Pred_Prob'].rank(pct=True, ascending=False)
        else:
            rank_safety = group['Pred_Prob'].rank(pct=True)

        # 加權合成總分
        group['Combined_Score'] = (rank_ret * weight_ret) + (rank_safety * (1 - weight_ret))

        long_thresh = group['Combined_Score'].quantile(1 - target_pct)
        short_thresh = group['Combined_Score'].quantile(target_pct)

        long_cond = group['Combined_Score'] >= long_thresh
        short_cond = group['Combined_Score'] <= short_thresh

        n_long = long_cond.sum()
        n_short = short_cond.sum()

        if n_long > 0:
            group.loc[long_cond, 'target_weight'] = 1.0 / n_long
        if n_short > 0:
            group.loc[short_cond, 'target_weight'] = -1.0 / n_short

        return group

    return pd.DataFrame((df.groupby('YearMonth').apply(calc_weights).reset_index(drop=True)))
