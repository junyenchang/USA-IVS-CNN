from setuptools import setup, find_packages

setup(
    name="usa_ivs_cnn",
    version="0.1",
    # 自動尋找資料夾底下包含 __init__.py 的目錄作為套件
    packages=find_packages(),
)