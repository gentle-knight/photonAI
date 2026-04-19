"""训练曲线、预测散点图与残差图。"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src.config import AppConfig
from src.data import TARGET_COLUMNS
from src.evaluate import predict_all
from src.model import MLPRegressor
from src.preprocess import ProcessedDataBundle
from src.trainer import TrainHistory, load_weights

logger = logging.getLogger(__name__)


def plot_loss_curves(history: TrainHistory, out_path: Path) -> None:
    """绘制 train/val loss 曲线。"""
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.epoch, history.train_loss, label="Train loss", linewidth=2)
    ax.plot(history.epoch, history.val_loss, label="Val loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (normalized target space)")
    ax.set_title("Training / Validation Loss")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("已保存 %s", out_path)


def plot_scatter_true_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str,
    out_path: Path,
    title_suffix: str = "Test set (physical units)",
) -> None:
    """单目标预测 vs 真值散点图。"""
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.35, edgecolors="none", s=18)
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Ideal")
    ax.set_xlabel(f"True {target_name}")
    ax.set_ylabel(f"Predicted {target_name}")
    ax.set_title(f"{target_name}: Pred vs True ({title_suffix})")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str,
    out_path: Path,
    title_suffix: str = "Test set (physical units)",
) -> None:
    """单目标残差直方图 + 预测值横轴散点。"""
    resid = y_pred - y_true
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(resid, bins=40, color="#4C72B0", alpha=0.85)
    axes[0].set_title(f"{target_name}: Residual histogram")
    axes[0].set_xlabel("Pred - True")
    axes[0].set_ylabel("Count")
    axes[1].scatter(y_pred, resid, alpha=0.35, s=16, edgecolors="none")
    axes[1].axhline(0.0, color="r", linestyle="--", linewidth=1.2)
    axes[1].set_xlabel(f"Predicted {target_name}")
    axes[1].set_ylabel("Residual")
    axes[1].set_title(f"{target_name}: Residual vs Pred ({title_suffix})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_all_figures(
    cfg: AppConfig,
    bundle: ProcessedDataBundle,
    run_dir: Path,
    history: TrainHistory,
    device: torch.device,
) -> None:
    """
    生成 loss 曲线与测试集上各目标的散点图、残差图。
    依赖 run_dir/checkpoints/best.pt。
    """
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_loss_curves(history, fig_dir / "loss_curve.png")

    model = MLPRegressor(
        input_dim=cfg.model.input_dim,
        hidden_dims=cfg.model.hidden_dims,
        output_dim=cfg.model.output_dim,
        batchnorm=cfg.model.batchnorm,
        dropout=cfg.model.dropout,
        residual=cfg.model.residual,
    ).to(device)
    load_weights(model, run_dir / "checkpoints" / "best.pt", device)

    pred_n, true_n = predict_all(model, bundle.test_loader, device)
    pred_p = bundle.y_scaler.inverse_transform(pred_n)
    true_p = bundle.y_scaler.inverse_transform(true_n)

    for i, name in enumerate(TARGET_COLUMNS):
        plot_scatter_true_pred(
            true_p[:, i],
            pred_p[:, i],
            name,
            fig_dir / f"scatter_{name}.png",
        )
        plot_residuals(
            true_p[:, i],
            pred_p[:, i],
            name,
            fig_dir / f"residual_{name}.png",
        )
    logger.info("所有图像已写入 %s", fig_dir)
