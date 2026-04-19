"""清洗、划分、标准化与 DataLoader 构建。"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.config import AppConfig
from src.data import ALL_COLUMNS, INPUT_COLUMNS, TARGET_COLUMNS, quality_report_before_clean

logger = logging.getLogger(__name__)


@dataclass
class ProcessedDataBundle:
    """训练用张量与 DataLoader，以及划分后的 numpy（含测试集原始物理量用于导出）。"""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    X_test_raw: np.ndarray
    y_test_raw: np.ndarray
    X_scaler: StandardScaler
    y_scaler: StandardScaler
    feature_names: List[str]
    target_names: List[str]


def _mask_outliers_iqr(
    values: np.ndarray, col_names: List[str], k: float
) -> np.ndarray:
    """返回 True 表示该行在任一选定列上超出训练集 IQR 范围（基于传入的 values 统计）。"""
    mask = np.zeros(len(values), dtype=bool)
    for j, _ in enumerate(col_names):
        col = values[:, j]
        q1, q3 = np.percentile(col, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        mask |= (col < lo) | (col > hi)
    return mask


def _mask_outliers_zscore(values: np.ndarray, threshold: float) -> np.ndarray:
    mask = np.zeros(len(values), dtype=bool)
    for j in range(values.shape[1]):
        col = values[:, j]
        mu, sig = col.mean(), col.std(ddof=0)
        if sig < 1e-12:
            continue
        z = np.abs((col - mu) / sig)
        mask |= z > threshold
    return mask


def _winsorize_train_apply_all(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    ql: float,
    qu: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按训练集分位数对 train/val/test 同步裁剪（列方向）。"""
    lo = np.quantile(train, ql, axis=0)
    hi = np.quantile(train, qu, axis=0)
    def clip_arr(a: np.ndarray) -> np.ndarray:
        return np.clip(a, lo, hi)
    return clip_arr(train), clip_arr(val), clip_arr(test)


def clean_dataframe(
    df: pd.DataFrame,
    cfg: AppConfig,
    report_lines: List[str],
) -> pd.DataFrame:
    """
    清洗流程（顺序固定，便于复现与审计）：

    1. 可选：完全重复行去重。
    2. 可选：以 **最后一列对应字段 V_pi**（txt 第 11 个逗号分隔字段）为门控，仅保留
       ``v_pi_min <= V_pi <= v_pi_max``（默认 [0, 500]）。
    3. 可选：再移除 ``V_pi <= 0``（与区间门控独立，由配置控制）。
    """
    out = df.copy()
    n0 = len(out)
    if cfg.remove_duplicate_rows:
        out = out.drop_duplicates()
        report_lines.append(f"去完全重复行: {n0} -> {len(out)}")
    if cfg.filter_v_pi_range:
        n1 = len(out)
        lo, hi = float(cfg.v_pi_min), float(cfg.v_pi_max)
        mask = (out["V_pi"] >= lo) & (out["V_pi"] <= hi)
        out = out[mask].reset_index(drop=True)
        report_lines.append(
            f"V_pi 物理区间过滤 [{lo}, {hi}]（txt 第 11 列 / 列名 V_pi）: {n1} -> {len(out)}"
        )
    if cfg.remove_nonpositive_vpi:
        n2 = len(out)
        out = out[out["V_pi"] > 0].reset_index(drop=True)
        report_lines.append(f"移除 V_pi<=0: {n2} -> {len(out)}")
    if len(out) == 0:
        raise ValueError(
            "清洗后样本数为 0：请检查 V_pi 区间配置、数据源或是否过度去重。"
        )
    return out


def stratified_split_indices(
    n: int,
    ratios: List[float],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 train/val/test 的整数索引（先 shuffle 再按比例切分）。"""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    tr, va, te = ratios
    n_test = int(round(n * te))
    n_val = int(round(n * va))
    n_train = n - n_val - n_test
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"划分后样本过少: train={n_train}, val={n_val}, test={n_test}，请调整比例或数据量"
        )
    i_train = idx[:n_train]
    i_val = idx[n_train : n_train + n_val]
    i_test = idx[n_train + n_val :]
    return i_train, i_val, i_test


def apply_train_only_outliers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: AppConfig,
    report_lines: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    仅在训练集上估计阈值：
    - iqr/zscore: 从训练集删除离群行（val/test 不动）
    - quantile_clip: 对 train/val/test 同步 winsorize（阈值来自 train）
    """
    strat = cfg.outlier_strategy
    if strat == "none":
        report_lines.append("outlier_strategy=none：不对数值做裁剪/删除（除配置项外）。")
        return X_train, y_train, X_val, y_val, X_test, y_test

    cols = cfg.outlier_apply_to
    if cols == "all":
        train_mat = np.hstack([X_train, y_train])
        val_mat = np.hstack([X_val, y_val])
        test_mat = np.hstack([X_test, y_test])
        names = INPUT_COLUMNS + TARGET_COLUMNS
    else:
        train_mat = y_train.copy()
        val_mat = y_val.copy()
        test_mat = y_test.copy()
        names = TARGET_COLUMNS

    if strat == "quantile_clip":
        ql = cfg.outlier_config.quantile_lower
        qu = cfg.outlier_config.quantile_upper
        tr2, va2, te2 = _winsorize_train_apply_all(train_mat, val_mat, test_mat, ql, qu)
        report_lines.append(
            f"quantile_clip: 按训练集分位数 [{ql}, {qu}] 对 {cols} 列 winsorize。"
        )
        if cols == "all":
            d = len(INPUT_COLUMNS)
            X_train, y_train = tr2[:, :d], tr2[:, d:]
            X_val, y_val = va2[:, :d], va2[:, d:]
            X_test, y_test = te2[:, :d], te2[:, d:]
        else:
            y_train, y_val, y_test = tr2, va2, te2
        return X_train, y_train, X_val, y_val, X_test, y_test

    if strat == "iqr":
        mask = _mask_outliers_iqr(train_mat, names, cfg.outlier_config.iqr_k)
    elif strat == "zscore":
        mask = _mask_outliers_zscore(train_mat, cfg.outlier_config.zscore_threshold)
    else:
        raise ValueError(f"未知 outlier_strategy: {strat}")

    removed = int(mask.sum())
    kept = ~mask
    X_train, y_train = X_train[kept], y_train[kept]
    report_lines.append(
        f"{strat}: 在训练子集上检测 {cols} 离群，删除训练行 {removed}，保留 {len(X_train)}。"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
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
        to_loader(X_val, y_val, shuffle=False),
        to_loader(X_test, y_test, shuffle=False),
    )


def save_scalers(
    X_scaler: StandardScaler,
    y_scaler: StandardScaler,
    run_dir: Path,
) -> None:
    with (run_dir / "x_scaler.pkl").open("wb") as f:
        pickle.dump(X_scaler, f)
    with (run_dir / "y_scaler.pkl").open("wb") as f:
        pickle.dump(y_scaler, f)


def load_scalers(run_dir: Path) -> Tuple[StandardScaler, StandardScaler]:
    with (run_dir / "x_scaler.pkl").open("rb") as f:
        X_scaler = pickle.load(f)
    with (run_dir / "y_scaler.pkl").open("rb") as f:
        y_scaler = pickle.load(f)
    return X_scaler, y_scaler


def save_split_indices(
    run_dir: Path,
    i_train: np.ndarray,
    i_val: np.ndarray,
    i_test: np.ndarray,
) -> None:
    """保存对清洗后矩阵行的划分索引，便于 eval 阶段完全复现。"""
    payload = {
        "train": i_train.astype(int).tolist(),
        "val": i_val.astype(int).tolist(),
        "test": i_test.astype(int).tolist(),
    }
    (run_dir / "split_indices.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def load_split_indices(run_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = run_dir / "split_indices.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"未找到 {path}。请使用本仓库训练产生的 run 目录，或先完成一次训练。"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return (
        np.asarray(data["train"], dtype=int),
        np.asarray(data["val"], dtype=int),
        np.asarray(data["test"], dtype=int),
    )


def rebuild_bundle_for_eval(
    df: pd.DataFrame,
    cfg: AppConfig,
    run_dir: Path,
) -> ProcessedDataBundle:
    """
    与训练阶段相同的清洗、划分与离群处理，但使用已保存的 StandardScaler 仅做 transform。
    用于独立 eval / infer 流程，避免重新拟合 scaler 造成分布偏移。
    """
    report_lines: List[str] = []
    cleaned = clean_dataframe(df, cfg, report_lines)
    X = cleaned[INPUT_COLUMNS].to_numpy(dtype=np.float64)
    y = cleaned[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    i_tr, i_va, i_te = load_split_indices(run_dir)
    for name, idx in ("train", i_tr), ("val", i_va), ("test", i_te):
        if len(idx) == 0 or int(idx.max()) >= len(X) or int(idx.min()) < 0:
            raise ValueError(
                f"split_indices.json 与当前数据不兼容（{name} 索引越界或为空）。"
                f"请确认 data_path 指向与训练相同的清洗后样本空间。"
            )

    X_train, y_train = X[i_tr], y[i_tr]
    X_val, y_val = X[i_va], y[i_va]
    X_test, y_test = X[i_te], y[i_te]
    X_train, y_train, X_val, y_val, X_test, y_test = apply_train_only_outliers(
        X_train, y_train, X_val, y_val, X_test, y_test, cfg, report_lines
    )

    X_scaler, y_scaler = load_scalers(run_dir)
    X_train_s = X_scaler.transform(X_train)
    y_train_s = y_scaler.transform(y_train)
    X_val_s = X_scaler.transform(X_val)
    y_val_s = y_scaler.transform(y_val)
    X_test_s = X_scaler.transform(X_test)
    y_test_s = y_scaler.transform(y_test)

    train_loader, val_loader, test_loader = build_dataloaders(
        X_train_s,
        y_train_s,
        X_val_s,
        y_val_s,
        X_test_s,
        y_test_s,
        cfg.training.batch_size,
        cfg.training.num_workers,
    )

    return ProcessedDataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        X_train=X_train_s,
        X_val=X_val_s,
        X_test=X_test_s,
        y_train=y_train_s,
        y_val=y_val_s,
        y_test=y_test_s,
        X_test_raw=X_test,
        y_test_raw=y_test,
        X_scaler=X_scaler,
        y_scaler=y_scaler,
        feature_names=list(INPUT_COLUMNS),
        target_names=list(TARGET_COLUMNS),
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
    """
    完整预处理流水线：质量报告 -> 清洗 -> 划分 -> 训练集离群处理 -> 标准化 -> DataLoader。
    将 data_report.md 与 cleaning 元数据写入 run_dir。
    """
    report_lines: List[str] = []
    raw_q = quality_report_before_clean(df, cfg.v_pi_min, cfg.v_pi_max)
    stats_before = df.describe().T

    cleaned = clean_dataframe(df, cfg, report_lines)
    stats_after = cleaned.describe().T

    X = cleaned[INPUT_COLUMNS].to_numpy(dtype=np.float64)
    y = cleaned[TARGET_COLUMNS].to_numpy(dtype=np.float64)

    i_tr, i_va, i_te = stratified_split_indices(len(X), cfg.split_ratios, cfg.random_seed)
    save_split_indices(run_dir, i_tr, i_va, i_te)
    X_train, y_train = X[i_tr], y[i_tr]
    X_val, y_val = X[i_va], y[i_va]
    X_test, y_test = X[i_te], y[i_te]
    report_lines.append(
        f"划分 train/val/test = {cfg.split_ratios}，样本数 "
        f"{len(X_train)}/{len(X_val)}/{len(X_test)}"
    )

    X_train, y_train, X_val, y_val, X_test, y_test = apply_train_only_outliers(
        X_train, y_train, X_val, y_val, X_test, y_test, cfg, report_lines
    )

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_s = X_scaler.fit_transform(X_train)
    y_train_s = y_scaler.fit_transform(y_train)
    X_val_s = X_scaler.transform(X_val)
    y_val_s = y_scaler.transform(y_val)
    X_test_s = X_scaler.transform(X_test)
    y_test_s = y_scaler.transform(y_test)

    save_scalers(X_scaler, y_scaler, run_dir)

    meta = {
        "raw_quality": raw_q,
        "cleaning_steps": report_lines,
        "split_ratios": cfg.split_ratios,
        "n_train": int(len(X_train_s)),
        "n_val": int(len(X_val_s)),
        "n_test": int(len(X_test_s)),
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

    train_loader, val_loader, test_loader = build_dataloaders(
        X_train_s,
        y_train_s,
        X_val_s,
        y_val_s,
        X_test_s,
        y_test_s,
        cfg.training.batch_size,
        cfg.training.num_workers,
    )

    return ProcessedDataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        X_train=X_train_s,
        X_val=X_val_s,
        X_test=X_test_s,
        y_train=y_train_s,
        y_val=y_val_s,
        y_test=y_test_s,
        X_test_raw=X_test,
        y_test_raw=y_test,
        X_scaler=X_scaler,
        y_scaler=y_scaler,
        feature_names=list(INPUT_COLUMNS),
        target_names=list(TARGET_COLUMNS),
    )
