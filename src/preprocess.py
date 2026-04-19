"""清洗、划分、标准化与 DataLoader 构建。"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.config import AppConfig
from src.data import INPUT_COLUMNS, TARGET_COLUMNS, quality_report_before_clean

logger = logging.getLogger(__name__)


@dataclass
class ProcessedDataBundle:
    """训练用张量与 DataLoader，以及 notebook 风格的 train/test 划分数据。"""

    train_loader: DataLoader
    test_loader: DataLoader
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    X_train_raw: np.ndarray
    X_test_raw: np.ndarray
    y_train_raw: np.ndarray
    y_test_raw: np.ndarray
    X_scaler: StandardScaler
    y_scalers: List[StandardScaler]
    feature_names: List[str]
    target_names: List[str]


def clean_dataframe(
    df: pd.DataFrame,
    cfg: AppConfig,
    report_lines: List[str],
) -> pd.DataFrame:
    """按 notebook 逻辑清洗：仅保留 V_pi < 阈值。"""
    out = df.copy()
    n0 = len(out)
    vmax = float(cfg.data.filter_v_pi_max)
    out = out[out["V_pi"] < vmax].reset_index(drop=True)
    report_lines.append(f"V_pi 阈值过滤 (< {vmax})：{n0} -> {len(out)}")
    if len(out) == 0:
        raise ValueError("清洗后样本数为 0，请检查数据源或 V_pi 阈值设置。")
    return out


def build_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    def to_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).float(),
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
        )

    return (
        to_loader(X_train, y_train, shuffle=True),
        to_loader(X_test, y_test, shuffle=False),
    )


def _fit_target_scalers(y_train: np.ndarray) -> Tuple[List[StandardScaler], np.ndarray]:
    scalers: List[StandardScaler] = []
    scaled_cols = []
    for i in range(y_train.shape[1]):
        scaler = StandardScaler()
        scaled_cols.append(scaler.fit_transform(y_train[:, i : i + 1]))
        scalers.append(scaler)
    return scalers, np.hstack(scaled_cols)


def transform_targets(y: np.ndarray, y_scalers: Sequence[StandardScaler]) -> np.ndarray:
    cols = [scaler.transform(y[:, i : i + 1]) for i, scaler in enumerate(y_scalers)]
    return np.hstack(cols)


def inverse_transform_targets(
    y_scaled: np.ndarray, y_scalers: Sequence[StandardScaler]
) -> np.ndarray:
    cols = [scaler.inverse_transform(y_scaled[:, i : i + 1]) for i, scaler in enumerate(y_scalers)]
    return np.hstack(cols)


def save_scalers(
    X_scaler: StandardScaler,
    y_scalers: Sequence[StandardScaler],
    run_dir: Path,
) -> None:
    with (run_dir / "x_scaler.pkl").open("wb") as f:
        pickle.dump(X_scaler, f)
    with (run_dir / "y_scalers.pkl").open("wb") as f:
        pickle.dump(list(y_scalers), f)


def load_scalers(run_dir: Path) -> Tuple[StandardScaler, List[StandardScaler]]:
    with (run_dir / "x_scaler.pkl").open("rb") as f:
        X_scaler = pickle.load(f)
    with (run_dir / "y_scalers.pkl").open("rb") as f:
        y_scalers = pickle.load(f)
    return X_scaler, list(y_scalers)


def save_split_indices(
    run_dir: Path,
    i_train: np.ndarray,
    i_test: np.ndarray,
) -> None:
    payload = {
        "train": i_train.astype(int).tolist(),
        "test": i_test.astype(int).tolist(),
    }
    (run_dir / "split_indices.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def load_split_indices(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    path = run_dir / "split_indices.json"
    if not path.is_file():
        raise FileNotFoundError(f"未找到 {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return (
        np.asarray(data["train"], dtype=int),
        np.asarray(data["test"], dtype=int),
    )


def write_data_report_md(
    path: Path,
    raw_quality: dict,
    report_lines: List[str],
    basic_stats_before: pd.DataFrame,
    basic_stats_after: pd.DataFrame,
) -> None:
    lines = [
        "# 数据与清洗报告",
        "",
        "## 清洗前质量摘要（JSON）",
        "```json",
        json.dumps(raw_quality, indent=2, ensure_ascii=False, default=str),
        "```",
        "",
        "## 清洗步骤",
        "\n".join(f"- {x}" for x in report_lines),
        "",
        "## 清洗前 describe（CSV 文本块）",
        "```text",
        basic_stats_before.to_csv(),
        "```",
        "",
        "## 清洗后 describe（CSV 文本块）",
        "```text",
        basic_stats_after.to_csv(),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def prepare_training_data(
    df: pd.DataFrame,
    cfg: AppConfig,
    run_dir: Path,
) -> ProcessedDataBundle:
    """按 notebook 一致逻辑准备 train/test、scaler 与 DataLoader。"""
    report_lines: List[str] = []
    raw_q = quality_report_before_clean(df, 0.0, cfg.data.filter_v_pi_max)
    stats_before = df.describe().T

    cleaned = clean_dataframe(df, cfg, report_lines)
    stats_after = cleaned.describe().T

    X = cleaned[INPUT_COLUMNS].to_numpy(dtype=np.float64)
    y = cleaned[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    all_idx = np.arange(len(X))

    X_train_raw, X_test_raw, y_train_raw, y_test_raw, i_train, i_test = train_test_split(
        X,
        y,
        all_idx,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state,
    )
    report_lines.append(
        f"train_test_split(test_size={cfg.data.test_size}, random_state={cfg.data.random_state}) "
        f"-> {len(X_train_raw)}/{len(X_test_raw)}"
    )
    save_split_indices(run_dir, i_train, i_test)

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train_raw)
    X_test = X_scaler.transform(X_test_raw)
    y_scalers, y_train = _fit_target_scalers(y_train_raw)
    y_test = transform_targets(y_test_raw, y_scalers)
    save_scalers(X_scaler, y_scalers, run_dir)

    meta = {
        "raw_quality": raw_q,
        "cleaning_steps": report_lines,
        "test_size": cfg.data.test_size,
        "random_state": cfg.data.random_state,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    (run_dir / "cleaning_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    write_data_report_md(
        run_dir / "data_report.md",
        raw_q,
        report_lines,
        stats_before,
        stats_after,
    )
    stats_after.to_csv(run_dir / "data_stats.csv", encoding="utf-8")
    logger.info("预处理完成：%s", run_dir / "data_report.md")

    train_loader, test_loader = build_dataloaders(
        X_train,
        y_train,
        X_test,
        y_test,
        cfg.training.batch_size,
        cfg.training.num_workers,
    )

    return ProcessedDataBundle(
        train_loader=train_loader,
        test_loader=test_loader,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        y_train_raw=y_train_raw,
        y_test_raw=y_test_raw,
        X_scaler=X_scaler,
        y_scalers=y_scalers,
        feature_names=list(INPUT_COLUMNS),
        target_names=list(TARGET_COLUMNS),
    )


def rebuild_bundle_for_eval(
    df: pd.DataFrame,
    cfg: AppConfig,
    run_dir: Path,
) -> ProcessedDataBundle:
    """使用训练时保存的切分索引与 scaler 重新构造 train/test 数据。"""
    report_lines: List[str] = []
    cleaned = clean_dataframe(df, cfg, report_lines)
    X = cleaned[INPUT_COLUMNS].to_numpy(dtype=np.float64)
    y = cleaned[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    i_train, i_test = load_split_indices(run_dir)
    for name, idx in (("train", i_train), ("test", i_test)):
        if len(idx) == 0 or int(idx.max()) >= len(X) or int(idx.min()) < 0:
            raise ValueError(f"split_indices.json 与当前数据不兼容（{name} 索引越界或为空）")

    X_train_raw, X_test_raw = X[i_train], X[i_test]
    y_train_raw, y_test_raw = y[i_train], y[i_test]
    X_scaler, y_scalers = load_scalers(run_dir)
    X_train = X_scaler.transform(X_train_raw)
    X_test = X_scaler.transform(X_test_raw)
    y_train = transform_targets(y_train_raw, y_scalers)
    y_test = transform_targets(y_test_raw, y_scalers)
    train_loader, test_loader = build_dataloaders(
        X_train,
        y_train,
        X_test,
        y_test,
        cfg.training.batch_size,
        cfg.training.num_workers,
    )

    return ProcessedDataBundle(
        train_loader=train_loader,
        test_loader=test_loader,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        y_train_raw=y_train_raw,
        y_test_raw=y_test_raw,
        X_scaler=X_scaler,
        y_scalers=y_scalers,
        feature_names=list(INPUT_COLUMNS),
        target_names=list(TARGET_COLUMNS),
    )
