"""加载模型并在 train/test 上评估，导出 CSV / JSON。"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.config import AppConfig
from src.data import INPUT_COLUMNS, TARGET_COLUMNS
from src.metrics import FullMetricsReport, compute_full_report, report_to_flat_dict
from src.model import create_model_from_config
from src.preprocess import ProcessedDataBundle, inverse_transform_targets
from src.trainer import load_weights

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
        preds.append(pr)
        trues.append(yb.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def _select_checkpoint(run_dir: Path) -> Path:
    ckpt_last = run_dir / "checkpoints" / "last.pt"
    if ckpt_last.is_file():
        return ckpt_last
    ckpt_best = run_dir / "checkpoints" / "best.pt"
    if ckpt_best.is_file():
        return ckpt_best
    raise FileNotFoundError(f"未找到 {run_dir}/checkpoints/last.pt 或 best.pt")


def evaluate_split(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    y_scalers,
    split_name: str,
) -> Tuple[FullMetricsReport, FullMetricsReport, float]:
    pred_n, true_n = predict_all(model, loader, device)
    loss = float(nn.functional.mse_loss(torch.from_numpy(pred_n), torch.from_numpy(true_n)).item())
    pred_p = inverse_transform_targets(pred_n, y_scalers)
    true_p = inverse_transform_targets(true_n, y_scalers)
    rep_n = compute_full_report(split_name, loss, true_n, pred_n, TARGET_COLUMNS)
    rep_p = compute_full_report(split_name, loss, true_p, pred_p, TARGET_COLUMNS)
    return rep_n, rep_p, loss


def run_full_evaluation(
    cfg: AppConfig,
    bundle: ProcessedDataBundle,
    run_dir: Path,
    device: torch.device,
) -> Tuple[nn.Module, Dict]:
    model = create_model_from_config(cfg).to(device)
    load_weights(model, _select_checkpoint(run_dir), device)

    rows = []
    summary: Dict = {"splits": {}}
    for name, loader in (("train", bundle.train_loader), ("test", bundle.test_loader)):
        rep_n, rep_p, loss = evaluate_split(
            model,
            loader,
            device,
            bundle.y_scalers,
            name,
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
    model.eval()
    pred_n, true_n = predict_all(model, bundle.test_loader, device)
    pred_p = inverse_transform_targets(pred_n, bundle.y_scalers)
    true_p = inverse_transform_targets(true_n, bundle.y_scalers)
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
