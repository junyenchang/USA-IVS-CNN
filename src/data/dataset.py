import typing
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from collections import defaultdict

from src.path import OptionPath

class IVSDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        start_year: int=1996,
        end_year: int=2024,
        value_col: str = 'impl_volatility',
        grid_T: typing.Optional[typing.List[int]] = None,
        target_transform: typing.Optional[typing.Callable] = None,
        transform: typing.Optional[typing.Callable] = None,
        global_returns: typing.Optional[pd.DataFrame] = None,
        shrcd: typing.Optional[typing.Tuple[int, ...]] = None,
        exchcd: typing.Optional[typing.Tuple[int, ...]] = None,
        return_outlier_quantile: typing.Optional[float] = None,
        prc_limit: typing.Optional[float] = None
    ):
        self.data_dir = data_dir
        self.value_col = value_col
        self.grid_T = grid_T
        self.target_transform = target_transform
        self.transform = transform
        self.global_returns = global_returns
        self.shrcd = shrcd
        self.exchcd = exchcd
        self.return_outlier_quantile = return_outlier_quantile
        self.prc_limit = prc_limit
        self.X_list: typing.List[torch.Tensor] = []
        self.y_list: typing.List[torch.Tensor] = []
        self.date_list: typing.List[np.ndarray] = []
        self.permno_list: typing.List[np.ndarray] = []

        os.makedirs(OptionPath.Cache, exist_ok=True)

        for year in range(start_year, end_year + 1):
            grid_suffix = "" if grid_T is None else "_" + "-".join(map(str, sorted(grid_T)))
            filter_suffix = ""
            if self.shrcd is not None:
                filter_suffix += "_shr" + "-".join(map(str, sorted(self.shrcd)))
            if self.exchcd is not None:
                filter_suffix += "_exc" + "-".join(map(str, sorted(self.exchcd)))
            if self.return_outlier_quantile is not None:
                q = int(self.return_outlier_quantile) * 100
                filter_suffix += f"_roq{q}pct"
            if self.prc_limit is not None:
                filter_suffix += f"_prc{self.prc_limit}"

            cache_file = os.path.join(OptionPath.Cache, f"ivs_tensor_{year}{grid_suffix}{filter_suffix}.pt")

            if os.path.exists(cache_file):
                cached = torch.load(cache_file, weights_only=False) # Changed to False because NumPy arrays are loaded
                self.X_list.append(cached['X'])
                self.y_list.append(cached['y'])
                self.date_list.append(cached['dates'])
                self.permno_list.append(cached['permnos'])
            else:
                X_year, y_year, dates_year, permnos_year = self._process_year(year)

                if X_year.shape[0] > 0:
                    torch.save({
                        'X': X_year,
                        'y': y_year,
                        'dates': dates_year,
                        'permnos': permnos_year
                    }, cache_file)

                self.X_list.append(X_year)
                self.y_list.append(y_year)
                self.date_list.append(dates_year)
                self.permno_list.append(permnos_year)

        valid_X = [x for x in self.X_list if x.shape[0] > 0 and len(x.shape) == 4]
        valid_y = [y for y in self.y_list if y.shape[0] > 0]
        valid_dates = [d for d in self.date_list if len(d) > 0]
        valid_permnos = [p for p in self.permno_list if len(p) > 0]

        if len(valid_X) > 0:
            self.X = torch.cat(valid_X, dim=0)
            self.y = torch.cat(valid_y, dim=0)
            self.dates = np.concatenate(valid_dates, axis=0)
            self.permnos = np.concatenate(valid_permnos, axis=0)

            if self.transform is not None and getattr(self.transform, 'is_cross_sectional', False):
                self.X = self.transform(self.X, dates=self.dates)

            self.y_raw = self.y.clone()

            if self.target_transform is not None:
                try:
                    transformed = self.target_transform(self.y.numpy(), dates=self.dates)
                except TypeError:
                    transformed = self.target_transform(self.y.numpy())
                self.y = torch.tensor(transformed, dtype=torch.float32)
        else:
            self.X = torch.empty((0, 1, 0, 0))
            self.y = torch.empty((0,))
            self.y_raw = torch.empty((0,))
            self.dates = np.array([])
            self.permnos = np.array([])

    def set_transform(self, transform: typing.Optional[typing.Callable]):
        self.transform = transform
        if self.transform is not None and getattr(self.transform, 'is_cross_sectional', False):
            if self.X.shape[0] > 0:
                self.X = self.transform(self.X, dates=self.dates)

    def _process_year(self, year: int) -> typing.Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        file_path = f"{self.data_dir}/option_ivs_crsp_{year}.parquet"
        if not os.path.exists(file_path):
            return torch.empty((0, 1, 0, 0)), torch.empty((0,)), np.array([]), np.array([])

        raw_df = pd.read_parquet(file_path)
        if self.shrcd is not None:
            raw_df = raw_df[raw_df['shrcd'].isin(self.shrcd)]
        if self.exchcd is not None:
            raw_df = raw_df[raw_df['exchcd'].isin(self.exchcd)]

        raw_df['opt_date'] = pd.to_datetime(raw_df['opt_date'])
        raw_df['crsp_date'] = pd.to_datetime(raw_df['crsp_date'])

        if self.global_returns is not None:
            unique_returns = self.global_returns
        else:
            dfs = [raw_df]
            next_year_file = f"{self.data_dir}/option_ivs_crsp_{year + 1}.parquet"
            if os.path.exists(next_year_file):
                next_df = pd.read_parquet(next_year_file, columns=['permno', 'crsp_date', 'crsp_monthly_return'])
                next_df['crsp_date'] = pd.to_datetime(next_df['crsp_date'])
                dfs.append(next_df)

            returns_concat = pd.concat(dfs, ignore_index=True)
            unique_returns = returns_concat[['permno', 'crsp_date', 'crsp_monthly_return']].drop_duplicates()
            unique_returns['target_for_month'] = unique_returns['crsp_date'].dt.to_period('M') - 1 # type: ignore
            unique_returns = unique_returns[['permno', 'target_for_month', 'crsp_monthly_return']]
            unique_returns.rename(columns={'crsp_monthly_return': 'future_return'}, inplace=True)

        expected_rows = raw_df['days'].nunique() * raw_df['delta'].nunique()
        valid_mask = raw_df.groupby(['secid', 'opt_date'])[self.value_col].transform('count') == expected_rows
        clean_df = raw_df[valid_mask].copy()

        clean_df['current_month'] = clean_df['opt_date'].dt.to_period('M')
        # CROSS-SECTIONAL MAX OP_DATE (instead of firm-level max)
        month_end_dates = clean_df.groupby('current_month')['opt_date'].transform('max')
        clean_df = clean_df[clean_df['opt_date'] == month_end_dates]

        df_merged = pd.merge(
            clean_df,
            unique_returns,
            left_on=['permno', 'current_month'],
            right_on=['permno', 'target_for_month'],
            how='left'
        )
        df_merged = df_merged[df_merged["eom_prc"] >= self.prc_limit] if self.prc_limit is not None else df_merged
        df_merged = df_merged.dropna(subset=['future_return'])

        if self.return_outlier_quantile is not None:
            q = self.return_outlier_quantile
            lower = df_merged.groupby('opt_date')['future_return'].transform(lambda x: x.quantile(q))
            upper = df_merged.groupby('opt_date')['future_return'].transform(lambda x: x.quantile(1 - q))
            df_merged = df_merged[(df_merged['future_return'] >= lower) & (df_merged['future_return'] <= upper)]

        self.df = df_merged
        if df_merged.empty:
            return torch.empty((0, 1, 0, 0)), torch.empty((0,)), np.array([]), np.array([])

        days_order = sorted(df_merged['days'].unique())
        if self.grid_T is not None:
            days_order = [d for d in days_order if d in self.grid_T]

        all_deltas = list(df_merged['delta'].unique())
        put_deltas = sorted([c for c in all_deltas if c < 0 and c >= -50])
        call_deltas = sorted([c for c in all_deltas if c > 0 and c <= 50])
        col_order = call_deltas + put_deltas

        # Perform a single global pivot table operation
        pivot_df = df_merged.pivot_table(
            index=['secid', 'permno', 'opt_date', 'future_return'],
            columns=['days', 'delta'],
            values=self.value_col
        )

        # Ensure sorting matches grouped format
        pivot_df = pivot_df.sort_index(level=['secid', 'opt_date'])

        # Enforce column structure matching standard IV surfaces
        new_columns = pd.MultiIndex.from_product([days_order, col_order], names=['days', 'delta'])
        pivot_df = pivot_df.reindex(columns=new_columns)

        # Vectorized Tensor Conversion
        n_days = len(days_order)
        n_deltas = len(col_order)
        X_np = pivot_df.values.reshape(-1, n_days, n_deltas).astype(np.float32)
        X_year = torch.tensor(X_np).unsqueeze(1)

        # Meta properties natively extracted
        idx_df = pivot_df.index.to_frame(index=False)
        y_year = torch.tensor(idx_df['future_return'].values.astype(np.float32))
        dates_year = np.array(idx_df['opt_date'].values)
        permnos_year = np.array(idx_df['permno'].values.astype(np.int32))

        return X_year, y_year, dates_year, permnos_year

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor, str, int, float]:
        opt_date_str = str(self.dates[idx])
        permno = self.permnos[idx]

        x = self.X[idx]
        if self.transform is not None and not getattr(self.transform, 'is_cross_sectional', False):
            x = self.transform(x)

        return x, self.y[idx], opt_date_str, permno, self.y_raw[idx].item()
