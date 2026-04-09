from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMDemandForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        prediction = self.output(last_hidden)
        return prediction.squeeze(-1)


@dataclass
class TrainerConfig:
    learning_rate: float = 1e-3
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 32


class SupplyChainTrainer:
    def __init__(self, input_size: int, config: TrainerConfig | None = None) -> None:
        self.config = config or TrainerConfig()
        self.device = torch.device("cpu")
        self.model = LSTMDemandForecaster(
            input_size=input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def train(self, train_x: np.ndarray, train_y: np.ndarray, epochs: int = 1) -> dict[str, float]:
        dataset = TensorDataset(
            torch.from_numpy(train_x).float(),
            torch.from_numpy(train_y).float(),
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        self.model.train()
        final_loss = 0.0
        for _ in range(epochs):
            running_loss = 0.0
            for features, targets in loader:
                features = features.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                preds = self.model(features)
                loss = self.criterion(preds, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += float(loss.item())

            final_loss = running_loss / max(1, len(loader))

        return {"train_loss": final_loss}

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        if len(x) == 0:
            return {"rmse": float("nan"), "mape": float("nan"), "loss": float("nan")}

        self.model.eval()
        with torch.no_grad():
            features = torch.from_numpy(x).float().to(self.device)
            targets = torch.from_numpy(y).float().to(self.device)
            preds = self.model(features)
            mse = self.criterion(preds, targets).item()

        preds_np = preds.cpu().numpy()
        y_np = targets.cpu().numpy()

        rmse = float(np.sqrt(np.mean((preds_np - y_np) ** 2)))
        denom = np.maximum(np.abs(y_np), 1e-6)
        mape = float(np.mean(np.abs((preds_np - y_np) / denom)) * 100.0)

        return {"rmse": rmse, "mape": mape, "loss": float(mse)}

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            features = torch.from_numpy(x).float().to(self.device)
            preds = self.model(features)
            return preds.cpu().numpy().astype(np.float32)

    def get_weights(self) -> list[np.ndarray]:
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_weights(self, weights: list[np.ndarray]) -> None:
        current_state = self.model.state_dict()
        if len(weights) != len(current_state):
            raise ValueError(f"Weight count mismatch: expected {len(current_state)}, got {len(weights)}")

        new_state: dict[str, Any] = {}
        for (name, old_value), new_value in zip(current_state.items(), weights):
            tensor = torch.tensor(new_value, dtype=old_value.dtype)
            new_state[name] = tensor

        self.model.load_state_dict(new_state, strict=True)
