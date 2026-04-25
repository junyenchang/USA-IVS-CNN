import os
import typing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.path import OptionPath

class BacktestEngine:
    def __init__(self, preds_df: pd.DataFrame, base_fee_bps: int = 10):
        """
        Args:
            preds_df: 模型輸出的 ensemble_predictions.csv
            base_fee_bps: 基礎手續費 (例如 10 bps = 0.001)
        """
        self.base_fee = base_fee_bps / 10000.0
        market_df = pd.read_parquet(os.path.join(OptionPath.StockInfo, 'market_metadata.parquet'))
        self.df = self._prepare_data(preds_df, market_df)

    def _prepare_data(self, preds_df: pd.DataFrame, market_data_df: pd.DataFrame) -> pd.DataFrame:
        """合併預測結果與市值特徵，並計算多空權重"""
        preds_df['Date'] = pd.to_datetime(preds_df['Date'])
        df = pd.merge(preds_df, market_data_df, on=['Date', 'Permno'], how='left')

        df['Date'] = pd.to_datetime(df['Date'])
        df['YearMonth'] = df['Date'].dt.to_period('M')
        df = df.sort_values(['YearMonth', 'Date', 'Permno'])
        df = df.drop_duplicates(subset=['YearMonth', 'Permno'], keep='last') # 每月最後一天的預測值代表該月的持倉決策

        # 2. 分組計算 Quantile 決定 Long/Short 部位 (每月月底)
        def assign_weights(group):
            # 切分 10 組，排除 NaN
            valid_preds = group['Pred'].dropna()
            if len(valid_preds) < 10:
                group['target_weight'] = 0
                return group

            # 計算 10% 與 90% 的分位數閾值
            q10 = valid_preds.quantile(0.10)
            q90 = valid_preds.quantile(0.90)

            # 給予權重 (假設 Equal Weight，多空各 100% 資金)
            long_cond = group['Pred'] >= q90
            short_cond = group['Pred'] <= q10

            n_long = long_cond.sum()
            n_short = short_cond.sum()

            group['target_weight'] = 0.0
            if n_long > 0:
                group.loc[long_cond, 'target_weight'] = 1.0 / n_long
            if n_short > 0:
                group.loc[short_cond, 'target_weight'] = -1.0 / n_short

            return group

        df = pd.DataFrame(df.groupby('YearMonth').apply(assign_weights).reset_index(drop=True))
        return df

    def run_simulation(self) -> pd.DataFrame:
        """執行跨期的每月回測計算"""
        months = sorted(self.df['YearMonth'].unique())

        results = []
        drifted_weights = pd.Series(dtype=float) # 記錄上一期的結算漂移權重

        for t_month in months:
            current_month = self.df[self.df['YearMonth'] == t_month].set_index('Permno')

            # --- 1. 目標權重與實際報酬 ---
            w_target = current_month['target_weight'].fillna(0)
            returns = current_month['Actual'].fillna(0)
            is_microcap = current_month['is_microcap'].fillna(False)

            # 對齊資產池 (包含上月持有但本月不在預測名單內的股票，須平倉)
            all_assets = w_target.index.union(drifted_weights.index)
            w_target = w_target.reindex(all_assets, fill_value=0.0)
            w_drifted = drifted_weights.reindex(all_assets, fill_value=0.0)
            returns = returns.reindex(all_assets, fill_value=0.0)
            is_microcap = is_microcap.reindex(all_assets, fill_value=False)

            # --- 2. 計算交易手續費 (TC_t) ---
            turnover = (w_target - w_drifted).abs()
            # 微型股手續費 x 2
            fee_multiplier = np.where(is_microcap, 2.0, 1.0)
            transaction_costs = turnover * self.base_fee * fee_multiplier
            total_tc = transaction_costs.sum()

            # --- 3. 計算放空成本 (SC_t) ---
            # 僅針對 w_target < 0 的部位，且微型股 4.8%，大型股 2.5%
            short_positions = w_target[w_target < 0].abs()
            short_fee_rate = np.where(is_microcap.loc[short_positions.index], 0.048, 0.025) / 12.0
            shorting_costs = short_positions * short_fee_rate
            total_sc = shorting_costs.sum()

            # --- 4. 計算原始報酬與淨報酬 ---
            raw_return = (w_target * returns).sum()
            net_return = raw_return - total_tc - total_sc

            # --- 5. 計算月底的漂移權重 (為下個月準備) ---
            # 月底資產價值變動 = 原始權重 * (1 + 該月報酬)
            end_of_month_values = w_target * (1 + returns)
            portfolio_multiplier = 1 + raw_return
            # 避免除以 0 (若資產歸零)
            if portfolio_multiplier != 0:
                drifted_weights = end_of_month_values / portfolio_multiplier
            else:
                drifted_weights = pd.Series(0, index=all_assets)

            results.append({
                'Date': t_month,
                'Raw_Return': raw_return,
                'Turnover': turnover.sum() / 2, # 單邊換手率
                'TC': total_tc,
                'SC': total_sc,
                'Net_Return': net_return
            })

        return pd.DataFrame(results)

    @staticmethod
    def calculate_metrics(portfolio_df: pd.DataFrame, save=True, save_path: str = "backtest_metrics.txt") -> typing.Dict[str, str]:
        """計算績效指標"""
        net_ret = portfolio_df['Net_Return']

        annualized_return = net_ret.mean() * 12
        annualized_vol = net_ret.std() * np.sqrt(12)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0

        # Calculate Maximum Drawdown
        cumulative_return = (1 + net_ret).cumprod()
        running_max = cumulative_return.cummax()
        drawdown = (cumulative_return - running_max) / running_max
        max_drawdown = drawdown.min()

        metrics = {
            "Annualized Return": f"{annualized_return:.2%}",
            "Annualized Volatility": f"{annualized_vol:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Max Drawdown": f"{max_drawdown:.2%}"
        }

        if save:
            with open(save_path, "w", encoding="utf-8") as f:
                for k, v in metrics.items():
                    line = f"{k}: {v}"
                    f.write(line + "\n")

        return metrics

    @staticmethod
    def save_and_plot_performance(portfolio_df: pd.DataFrame, benchmark_path: str, save_folder: str):
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

        # === 1. 繪製圖表 ===
        plt.figure(figsize=(10, 6))
        plt.plot(merged_df['Date'], merged_df['Strategy_CumRet'], label='CNN-IVS Strategy', color='red')
        plt.plot(merged_df['Date'], merged_df['SPY_CumRet'], label='SPY Benchmark', color='blue', alpha=0.7)

        plt.title('Cumulative Performance vs Benchmark')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()

        # === 2. 儲存圖片 ===
        plot_path = os.path.join(save_folder, "cumulative_performance.png")
        plt.savefig(plot_path, dpi=300)
        plt.show()

        # === 3. 儲存每期報酬紀錄為 CSV ===
        csv_path = os.path.join(save_folder, "backtest_timeseries.csv")
        merged_df.drop(columns=['YearMonth']).to_csv(csv_path, index=False)

        print(f"回測圖表與時序結果已儲存至: {save_folder}")
        return merged_df
