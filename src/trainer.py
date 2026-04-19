"""训练循环、早停、调度器与 checkpoint。"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

from src.config import AppConfig
from src.losses import build_loss
from src.model import MLPRegressor

logger = logging.getLogger(__name__)


@dataclass
class TrainHistory:
    epoch: List[int]
    train_loss: List[float]
    val_loss: List[float]
    lr: List[float]


def _move_batch(
    batch: Tuple[torch.Tensor, torch.Tensor], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y = batch
    return x.to(device), y.to(device)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        xb, yb = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu()) * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate_loss_loader(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        xb, yb = _move_batch(batch, device)
        pred = model(xb)
        loss = criterion(pred, yb)
        total += float(loss.detach().cpu()) * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


def build_optimizer_and_scheduler(
    model: nn.Module, cfg: AppConfig
) -> Tuple[AdamW, object]:
    opt = AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )
    if cfg.scheduler.type == "cosine":
        sched: torch.optim.lr_scheduler._LRScheduler = CosineAnnealingLR(
            opt, T_max=cfg.training.epochs, eta_min=cfg.scheduler.plateau_min_lr
        )
    elif cfg.scheduler.type == "plateau":
        sched = ReduceLROnPlateau(
            opt,
            mode="min",
            factor=cfg.scheduler.plateau_factor,
            patience=cfg.scheduler.plateau_patience,
            min_lr=cfg.scheduler.plateau_min_lr,
        )
    else:
        raise ValueError(f"未知 scheduler.type: {cfg.scheduler.type}")
    return opt, sched


def fit(
    model: nn.Module,
    cfg: AppConfig,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    run_dir: Path,
    device: torch.device,
) -> TrainHistory:
    """
    训练模型：早停依据验证集损失；保存 best / last 权重到 run_dir/checkpoints。
    同步写入 train_log.csv。
    """
    criterion = build_loss(cfg.loss).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.csv"

    best_val = float("inf")
    best_epoch = -1
    patience_left = cfg.training.early_stopping_patience

    hist = TrainHistory(epoch=[], train_loss=[], val_loss=[], lr=[])

    with log_path.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "best_val"])

        for epoch in range(1, cfg.training.epochs + 1):
            tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            va_loss = evaluate_loss_loader(model, val_loader, criterion, device)

            if cfg.scheduler.type == "cosine":
                scheduler.step()
            elif cfg.scheduler.type == "plateau":
                scheduler.step(va_loss)

            lr_now = float(optimizer.param_groups[0]["lr"])
            hist.epoch.append(epoch)
            hist.train_loss.append(tr_loss)
            hist.val_loss.append(va_loss)
            hist.lr.append(lr_now)

            improved = va_loss + 1e-12 < best_val
            if improved:
                best_val = va_loss
                best_epoch = epoch
                patience_left = cfg.training.early_stopping_patience
                torch.save(
                    {"epoch": epoch, "model_state": model.state_dict(), "val_loss": va_loss},
                    ckpt_dir / "best.pt",
                )
            else:
                patience_left -= 1

            writer.writerow([epoch, tr_loss, va_loss, lr_now, best_val])
            fcsv.flush()

            logger.info(
                "Epoch %d | train_loss=%.6f val_loss=%.6f | best_val=%.6f @%d",
                epoch,
                tr_loss,
                va_loss,
                best_val,
                best_epoch,
            )

            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "val_loss": va_loss},
                ckpt_dir / "last.pt",
            )

            if patience_left <= 0:
                logger.info("早停触发于 epoch %d，最佳 epoch=%d", epoch, best_epoch)
                break

    return hist


def load_weights(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    """从 checkpoint 载入 model_state。"""
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
