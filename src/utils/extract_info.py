import os
import typing
import pandas as pd

def build_market_metadata(data_dir: str, out_put_dir: str, start_year=1996, end_year=2024):
    all_meta: typing.List[pd.DataFrame] = []

    for year in range(start_year, end_year + 1):
        yearly_file = os.path.join(data_dir, f"option_ivs_crsp_{year}.parquet")
        if not os.path.exists(yearly_file):
            print(f"Skip year {year} - no files found.")
            continue
        print(f"Processing year {year}")

        cols_to_use = ['permno', 'opt_date', 'size_group', 'is_non_micro', 'market_cap']
        df = pd.read_parquet(yearly_file, columns=cols_to_use)
        df = df.drop_duplicates(subset=['permno', 'opt_date'])

        df['is_microcap'] = ~df['is_non_micro']  # reverse the is_non_micro to get is_microcap
        df = df.rename(columns={'opt_date': 'Date', 'permno': 'Permno'})
        df = df[['Date', 'Permno', 'market_cap', 'is_microcap', 'size_group']]

        all_meta.append(df)

    final_meta = pd.concat(all_meta, ignore_index=True)
    final_meta = final_meta.sort_values(['Date', 'Permno']).reset_index(drop=True)

    save_path = os.path.join(out_put_dir, 'market_metadata.parquet')
    final_meta.to_parquet(save_path, index=False)
    print(f"Stock Information for backtest saved. File Size: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")
