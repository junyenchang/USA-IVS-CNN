import os
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()

# 往上找兩層：第一層是 src，第二層就是專案根目錄
PROJECT_ROOT = CURRENT_FILE.parent.parent

DB_ROOT = os.path.join(PROJECT_ROOT, "DB")

class OptionPath:
    ROOT: str = os.path.join(DB_ROOT, "OptionDB")

    IVS: str = os.path.join(ROOT, "USA_IVS")

    @classmethod
    def ensure_dirs(cls) -> None:
        """確保所有必要資料夾存在。"""
        for path in [cls.IVS]:
            os.makedirs(path, exist_ok=True)

OptionPath.ensure_dirs()