import os
import json

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import asdict
from configs.default import BaselineConfig

class ExperimentLogger:
    def __init__(self, config: BaselineConfig):
        self.config = config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if config.exp_group:
            self.exp_dir = os.path.join(config.result_dir, f"{config.exp_group}_{timestamp}", config.exp_name)
        else:
            self.exp_dir = os.path.join(config.result_dir, f"{config.exp_name}_{timestamp}")

        os.makedirs(self.exp_dir, exist_ok=True)
        self._save_config()

    def _save_config(self):
        """將 dataclass 轉換為 dictionary 並儲存為 JSON"""
        config_path = os.path.join(self.exp_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.config), f, indent=4, ensure_ascii=False)

    def save_all_loss_histories(self, histories: list):
        """將所有 ensemble 的 loss 合併儲存為一個 CSV 與一張圖"""
        # 將所有的 history 轉成一個字典
        combined_data = {}
        for i, h in enumerate(histories):
            # 使用 pd.Series 可以自動處理長度不一致 (Early Stopping) 的問題，不足的 Epoch 會填補 NaN
            combined_data[f'model_{i}_train_loss'] = pd.Series(h['train_loss'])
            combined_data[f'model_{i}_val_loss'] = pd.Series(h['val_loss'])

        # 轉換為 DataFrame
        df = pd.DataFrame(combined_data)
        df.index.name = 'Epoch'
        df.index += 1  # 讓 Epoch 從 1 開始顯示

        # 1. 儲存為單一 CSV
        csv_path = os.path.join(self.exp_dir, "loss_history_all.csv")
        df.to_csv(csv_path)

        # 2. 畫在同一張圖上
        plt.figure(figsize=(10, 6))
        for i in range(len(histories)):
            # 訓練集統一用藍色系，驗證集統一用橘色系，加上透明度(alpha)避免線條互相遮蔽
            plt.plot(df.index, df[f'model_{i}_train_loss'], color='tab:blue', alpha=0.5,
                     label='Train Loss' if i == 0 else "")
            plt.plot(df.index, df[f'model_{i}_val_loss'], color='tab:orange', alpha=0.5,
                     label='Validation Loss' if i == 0 else "")

        plt.title("Ensemble Models Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        # 避免相同的 Label 出現好幾次
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.savefig(os.path.join(self.exp_dir, "loss_curve_all.png"))
        plt.close()

    def save_loss_history(self, history: dict, ensemble_idx: int):
        """儲存 loss 並畫圖"""
        # history 可能是 {'train_loss': [...], 'val_loss': [...]}
        df = pd.DataFrame(history)
        csv_path = os.path.join(self.exp_dir, f"loss_history_model_{ensemble_idx}.csv")
        df.to_csv(csv_path, index=False)

        plt.figure()
        plt.plot(df['train_loss'], label='Train Loss')
        plt.plot(df['val_loss'], label='Validation Loss')
        plt.title(f"Model {ensemble_idx} Loss")
        plt.legend()
        plt.savefig(os.path.join(self.exp_dir, f"loss_curve_model_{ensemble_idx}.png"))
        plt.close()

    def save_predictions(self, predictions_df: pd.DataFrame, filename: str = "ensemble_predictions.csv"):
        """儲存最終 Ensemble 預測結果"""
        save_path = os.path.join(self.exp_dir, filename)
        predictions_df.to_csv(save_path, index=False)
