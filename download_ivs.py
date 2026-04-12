from src.path import OptionPath
from src.wrds_client import WRDSClient

def main():
    OptionPath.ensure_dirs() # ensure output directories exist

    loader = WRDSClient(username='jimsir49')
    years_to_fetch = range(1996, 2025) # 1996-2024


    for year in years_to_fetch:
        loader.fetch_and_save_year(year=year, output_dir=OptionPath.IVS)

if __name__ == "__main__":
    main()