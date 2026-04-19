"""加载最优模型并在各划分上评估，导出 CSV / JSON。"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.config import AppConfig
from src.data import INPUT_COLUMNS, TARGET_COLUMNS
from src.losses import build_loss
from src.metrics import (
    FullMetricsReport,
    compute_full_report,
    report_to_flat_dict,
)
from src.model import MLPRegressor
from src.preprocess import ProcessedDataBundle
from src.trainer import evaluate_loss_loader, load_weights

logger = logging.getLogger(__name__)


@torch.no_grad()
def predict_all(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        pr = model(xb).detach().cpu().numpy()
        yt = yb.numpy()
        preds.append(pr)
        trues.append(yt)
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def evaluate_split(
    model: nn.Module,
    criterion: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    y_scaler,
    split_name: str,
) -> Tuple[FullMetricsReport, FullMetricsReport, float]:
    """返回 (标准化空间报告, 物理空间报告, 平均损失)。"""
    loss = evaluate_loss_loader(model, loader, criterion, device)
    pred_n, true_n = predict_all(model, loader, device)
    pred_p = y_scaler.inverse_transform(pred_n)
    true_p = y_scaler.inverse_transform(true_n)
    rep_n = compute_full_report(split_name, loss, true_n, pred_n, TARGET_COLUMNS)
    rep_p = compute_full_report(split_name, loss, true_p, pred_p, TARGET_COLUMNS)
    return rep_n, rep_p, loss


def run_full_evaluation(
    cfg: AppConfig,
    bundle: ProcessedDataBundle,
    run_dir: Path,
    device: torch.device,
) -> Tuple[nn.Module, Dict]:
    """载入 best.pt，在 train/val/test 上评估并写 metrics.csv 与 summary.json；返回模型与摘要。"""
    model = MLPRegressor(
        input_dim=cfg.model.input_dim,
        hidden_dims=cfg.model.hidden_dims,
        output_dim=cfg.model.output_dim,
        batchnorm=cfg.model.batchnorm,
        dropout=cfg.model.dropout,
        residual=cfg.model.residual,
    ).to(device)
    ckpt_best = run_dir / "checkpoints" / "best.pt"
    load_weights(model, ckpt_best, device)

    criterion = build_loss(cfg.loss).to(device)
    y_scaler = bundle.y_scaler

    rows = []
    summary: Dict = {"splits": {}}

    for name, loader in (
        ("train", bundle.train_loader),
        ("val", bundle.val_loader),
        ("test", bundle.test_loader),
    ):
        rep_n, rep_p, loss = evaluate_split(
            model, criterion, loader, device, y_scaler, name
        )
        summary["splits"][name] = {
            "loss": loss,
            "normalized": asdict(rep_n),
            "physical": asdict(rep_p),
        }
        row = report_to_flat_dict(rep_n, rep_p)
        row["split"] = name
        rows.append(row)

    metrics_path = run_dir / "metrics.csv"
    if rows:
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("已写入 %s 与 %s", metrics_path, run_dir / "summary.json")
    return model, summary


def export_test_predictions_csv(
    bundle: ProcessedDataBundle,
    model: nn.Module,
    device: torch.device,
    path: Path,
) -> None:
    """导出测试集物理空间真值、预测与误差。"""
    model.eval()
    pred_n, true_n = predict_all(model, bundle.test_loader, device)
    pred_p = bundle.y_scaler.inverse_transform(pred_n)
    true_p = bundle.y_scaler.inverse_transform(true_n)
    err = pred_p - true_p
    cols: Dict[str, np.ndarray] = {}
    for j, name in enumerate(INPUT_COLUMNS):
        cols[name] = bundle.X_test_raw[:, j]
    for i, name in enumerate(TARGET_COLUMNS):
        cols[f"true_{name}"] = true_p[:, i]
        cols[f"pred_{name}"] = pred_p[:, i]
        cols[f"err_{name}"] = err[:, i]
    pd.DataFrame(cols).to_csv(path, index=False)
    logger.info("已写入 %s", path)
