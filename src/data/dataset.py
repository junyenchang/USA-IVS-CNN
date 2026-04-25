import typing
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class IVSDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        start_year: int=1996,
        end_year: int=2021,
        value_col: str = 'impl_volatility',
        grid_T: typing.Optional[typing.List[int]] = None,
        target_transform: typing.Optional[typing.Callable] = None,
        transform: typing.Optional[typing.Callable] = None
    ):
        self.value_col = value_col
        self.grid_T = grid_T
        self.target_transform = target_transform
        self.transform = transform

        all_dfs: typing.List[pd.DataFrame] = []
        for year in range(start_year, end_year + 1):
            file_path = f"{data_dir}/option_ivs_crsp_{year}.parquet"
            df = pd.read_parquet(file_path)
            all_dfs.append(df)

        raw_df = pd.concat(all_dfs, ignore_index=True)
        raw_df['opt_date'] = pd.to_datetime(raw_df['opt_date'])
        raw_df['crsp_date'] = pd.to_datetime(raw_df['crsp_date'])

        # 2. 萃取「全域報酬池 (Returns Pool)」
        # 這裡會包含全部的 (permno, crsp_date, crsp_monthly_return)
        unique_returns = raw_df[['permno', 'crsp_date', 'crsp_monthly_return']].drop_duplicates()

        # 將時間轉換為月份 Period，並計算「這個報酬是哪個月份的 Target」
        unique_returns['target_for_month'] = unique_returns['crsp_date'].dt.to_period('M') - 1 # type: ignore
        unique_returns = unique_returns[['permno', 'target_for_month', 'crsp_monthly_return']]
        unique_returns.rename(columns={'crsp_monthly_return': 'future_return'}, inplace=True)

        # 過濾殘缺的 IVS
        expected_rows = raw_df['days'].nunique() * raw_df['delta'].nunique()
        valid_mask = raw_df.groupby(['secid', 'opt_date'])[self.value_col].transform('count') == expected_rows
        clean_df = raw_df[valid_mask].copy()

        # Vectorized 對接未來的報酬 (Label)
        clean_df['current_month'] = clean_df['opt_date'].dt.to_period('M') # type: ignore

        self.df = pd.merge(
            clean_df,
            unique_returns,
            left_on=['permno', 'current_month'],
            right_on=['permno', 'target_for_month'],
            how='left' # 使用 left merge，找不到未來報酬的就是 NaN
        )

        # 濾掉最終沒有未來報酬的無效樣本
        self.df = self.df.dropna(subset=['future_return'])

        # 5. 分組張量轉換 (不用再做 shift )
        self.X_list = []
        self.y_list = []
        self.date_list = []
        self.permno_list = []

        grouped_stock = self.df.groupby('secid')
        for secid, group in grouped_stock:
            group = group.sort_values(by='opt_date')
            X_base, dates = self._create_base_tensor(group)

            # 從整理好的資料中直接拉出已經對齊好的 future_return
            returns_aligned = group.drop_duplicates(subset=['opt_date']).set_index('opt_date')['future_return']
            valid_y = returns_aligned.reindex(dates).values

            if self.target_transform is not None:
                valid_y = self.target_transform(valid_y)

            self.X_list.append(X_base)
            self.y_list.append(valid_y)
            self.date_list.append(dates)
            self.permno_list.extend([group['permno'].iloc[0]] * len(dates))

        if len(self.X_list) > 0:
            self.X = torch.tensor(np.concatenate(self.X_list, axis=0), dtype=torch.float32)
            # 擴充維度：(N, 1, T_len, Delta_len) 以符合 CNN 需求
            self.X = self.X.unsqueeze(1)

            # 這裡對齊 y 的 dtype，若是分類目標通常為 long，迴歸為 float32
            # 依賴 numpy 轉換後的型別，或者在 target_transform 中指定也可以
            y_concat = np.concatenate(self.y_list, axis=0)
            self.y = torch.tensor(y_concat).squeeze()
            self.dates = np.concatenate(self.date_list, axis=0)
            self.permnos = np.array(self.permno_list)
        else:
            self.X = torch.empty((0, 1, 0, 0))
            self.y = torch.empty((0,))
            self.dates = np.array([])
            self.permnos = np.array([])

    def _create_base_tensor(self, df: pd.DataFrame) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        將指定股票的資料轉換為特徵張量。屬於內部方法。
        """
        all_deltas: typing.List[float] = list(df['delta'].unique())

        put_deltas: typing.List[float] = sorted([c for c in all_deltas if c < 0 and c >= -50])
        call_deltas: typing.List[float] = sorted([c for c in all_deltas if c > 0 and c <= 50])
        col_order = call_deltas + put_deltas

        images_list: typing.List[np.ndarray] = []
        dates_list: typing.List = []
        grouped = df.groupby('opt_date')

        for date, group in grouped:
            matrix = group.pivot_table(index='days', columns='delta', values=self.value_col)
            matrix = matrix.reindex(columns=col_order)

            if self.grid_T is not None:
                matrix = matrix.loc[matrix.index.isin(self.grid_T)]

            images_list.append(matrix.values)
            dates_list.append(date)

        X_base = np.stack(images_list, axis=0)
        return X_base, np.array(dates_list)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> typing.Tuple[torch.Tensor, torch.Tensor, str, int]:
        opt_date_str = str(self.dates[idx])
        permno = self.permnos[idx]

        x = self.X[idx]
        if self.transform is not None:
            x = self.transform(x)

        return x, self.y[idx], opt_date_str, permno
