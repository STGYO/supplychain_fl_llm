from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data.synthetic_generator import GeneratorConfig, generate_synthetic_data

REQUIRED_COLUMNS = [
    "time_step",
    "product_id",
    "demand",
    "inventory",
    "lead_time",
    "unit_cost",
    "transport_cost",
    "emissions",
    "stockout_penalty",
    "disruption_flag",
]

FEATURE_COLUMNS = REQUIRED_COLUMNS.copy()
TARGET_COLUMN = "demand"


@dataclass
class LocalSequenceDataset:
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    feature_columns: list[str]



def ensure_partner_dataset(dataset_path: str, partner_id: int, seed: int = 42) -> Path:
    target = Path(dataset_path)
    if target.exists():
        return target

    output_dir = target.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_synthetic_data(
        GeneratorConfig(partners=max(partner_id, 3), products=5, timesteps=180, seed=seed, output_dir=str(output_dir))
    )

    if not target.exists():
        raise FileNotFoundError(f"Dataset generation completed but partner file was not found: {target}")

    return target


def load_partner_dataframe(dataset_path: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    return df


def _build_sequences(df: pd.DataFrame, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []

    for _product_id, product_df in df.groupby("product_id"):
        series = product_df.sort_values("time_step").reset_index(drop=True)
        features = series[FEATURE_COLUMNS].astype(np.float32).to_numpy()
        targets = series[TARGET_COLUMN].astype(np.float32).to_numpy()

        if len(series) <= sequence_length:
            continue

        for idx in range(sequence_length, len(series)):
            x_rows.append(features[idx - sequence_length : idx])
            y_rows.append(float(targets[idx]))

    if not x_rows:
        raise ValueError("No training sequences were created. Increase dataset size or reduce sequence length.")

    x = np.stack(x_rows).astype(np.float32)
    y = np.asarray(y_rows, dtype=np.float32)
    return x, y


def _normalize_train_val(train_x: np.ndarray, val_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feature_mean = train_x.mean(axis=(0, 1), keepdims=True)
    feature_std = train_x.std(axis=(0, 1), keepdims=True)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)

    train_norm = (train_x - feature_mean) / feature_std
    val_norm = (val_x - feature_mean) / feature_std
    return train_norm.astype(np.float32), val_norm.astype(np.float32)


def load_sequence_dataset(dataset_path: str, sequence_length: int = 12, val_ratio: float = 0.2) -> LocalSequenceDataset:
    df = load_partner_dataframe(dataset_path)
    x, y = _build_sequences(df=df, sequence_length=sequence_length)

    split_idx = max(1, int(len(x) * (1.0 - val_ratio)))
    split_idx = min(split_idx, len(x) - 1)

    train_x = x[:split_idx]
    train_y = y[:split_idx]
    val_x = x[split_idx:]
    val_y = y[split_idx:]

    train_x, val_x = _normalize_train_val(train_x, val_x)

    return LocalSequenceDataset(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        feature_columns=FEATURE_COLUMNS,
    )
