import os
import typing
import torch
import torch.nn as nn
import copy
from tqdm import tqdm

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        """
        Args:
            patience (int): 容忍幾個 epoch 驗證集 loss 沒有改善。
            min_delta (float): 被判定為有改善的最小變化量。
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_weights = None

        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class Trainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, task_type: str, device: torch.device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.task_type = task_type
        self.device = device
        self.model.to(self.device)

    def train_epoch(self, dataloader: torch.utils.data.DataLoader):
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        pbar = tqdm(dataloader, desc="Training", leave=False)
        for batch_data in pbar:
            X_batch: torch.Tensor = batch_data[0].to(self.device)
            y_batch: torch.Tensor = batch_data[1].to(self.device)
            batch_size: int = X_batch.size(0)

            self.optimizer.zero_grad() # 1. 清空梯度

            predictions: torch.Tensor = self.model(X_batch)  # 2. Forward pass (正向傳播預測)

            # 確保 shape 一致，cnn.py 輸出的 predictions 可能是 (Batch, 1) 但 y_batch 是 (Batch,)
            predictions = predictions.squeeze(-1)

            loss: torch.Tensor = self.criterion(predictions, y_batch)  # 3. 計算誤差

            loss.backward() # 4. Backward pass (反向傳播計算梯度)

            self.optimizer.step() # 5. 更新權重

            total_loss += loss.item() * batch_size
            total_samples += batch_size
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        return avg_loss

    def evaluate(self, dataloader: torch.utils.data.DataLoader):
        self.model.eval() # 設定為評估模式
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad(): # 評估時不需要計算梯度，節省記憶體
            for batch_data in dataloader:
                X_batch = batch_data[0].to(self.device)
                y_batch = batch_data[1].to(self.device)
                batch_size: int = X_batch.size(0)

                predictions: torch.Tensor = self.model(X_batch).squeeze(-1)
                loss: torch.Tensor = self.criterion(predictions, y_batch)

                total_loss += loss.item() * batch_size
                total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        return avg_loss

    def fit(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, epochs: int, early_stopping: typing.Optional['EarlyStopping'] = None):
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if early_stopping:
                early_stopping(val_loss, self.model)
                if early_stopping.early_stop:
                    if early_stopping.best_weights is not None:
                        self.model.load_state_dict(early_stopping.best_weights)
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        if early_stopping and early_stopping.best_weights is not None and not early_stopping.early_stop:
            self.model.load_state_dict(early_stopping.best_weights)
        return history

    def predict(self, dataloader: torch.utils.data.DataLoader):
        """新增的預測方法，回傳預測值與真實值"""
        self.model.eval()
        all_preds: typing.List[float] = []
        all_targets: typing.List[float] = []
        all_dates: typing.List[str] = []
        all_permnos: typing.List[int] = []

        with torch.no_grad():
            for batch_data in dataloader:
                X_batch: torch.Tensor = batch_data[0].to(self.device)
                y_batch: torch.Tensor = batch_data[1].to(self.device)
                dates_batch: typing.List[str] = batch_data[2]
                permnos_batch: torch.Tensor = batch_data[3]
                predictions: torch.Tensor = self.model(X_batch).squeeze(-1)

                if self.task_type == "classification":
                    predictions = torch.sigmoid(predictions)

                all_preds.extend(predictions.cpu().tolist())
                all_targets.extend(y_batch.cpu().tolist())
                all_dates.extend(dates_batch)           # strings list 不會被 DataLoader 轉型所以沒事
                all_permnos.extend(permnos_batch.tolist())


        return all_preds, all_targets, all_dates, all_permnos
