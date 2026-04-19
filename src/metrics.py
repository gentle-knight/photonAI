"""MAE / RMSE / R²，支持逐维与整体平均。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """单变量或多变量整体 R²（按 sklearn 定义在最后一维聚合）。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
    if ss_tot < 1e-15:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def per_output_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, names: List[str]
) -> Dict[str, Dict[str, float]]:
    """对每个输出维度计算 MAE、RMSE、R2。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        mae = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-15 else float("nan")
        out[name] = {"mae": mae, "rmse": rmse, "r2": r2}
    return out


def overall_avg_mae_rmse(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float]:
    """三输出维度上 MAE/RMSE 的简单算术平均。"""
    per = per_output_metrics(y_true, y_pred, ["t0", "t1", "t2"])
    mae = float(np.mean([per[k]["mae"] for k in per]))
    rmse = float(np.mean([per[k]["rmse"] for k in per]))
    return mae, rmse


@dataclass
class FullMetricsReport:
    split: str
    loss: float
    per_target: Dict[str, Dict[str, float]]
    overall_mae: float
    overall_rmse: float
    overall_r2: float


def compute_full_report(
    split: str,
    loss: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str],
) -> FullMetricsReport:
    per = per_output_metrics(y_true, y_pred, target_names)
    mae, rmse = overall_avg_mae_rmse(y_true, y_pred)
    r2 = _r2_score(y_true, y_pred)
    return FullMetricsReport(
        split=split,
        loss=float(loss),
        per_target=per,
        overall_mae=mae,
        overall_rmse=rmse,
        overall_r2=float(r2),
    )


def report_to_flat_dict(
    norm: FullMetricsReport,
    phys: FullMetricsReport,
) -> Dict[str, float | str]:
    """展平为 CSV 一行友好的字典。"""
    row: Dict[str, float | str] = {"split": norm.split}
    for prefix, rep in (("norm", norm), ("phys", phys)):
        row[f"{prefix}_loss"] = rep.loss
        row[f"{prefix}_overall_mae"] = rep.overall_mae
        row[f"{prefix}_overall_rmse"] = rep.overall_rmse
        row[f"{prefix}_overall_r2"] = rep.overall_r2
        for tname, vals in rep.per_target.items():
            for mname, v in vals.items():
                row[f"{prefix}_{tname}_{mname}"] = v
    return row
