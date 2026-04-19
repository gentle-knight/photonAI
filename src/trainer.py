"""Notebook 风格的 MoE + autograd PINN 训练循环。"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.config import AppConfig
from src.preprocess import ProcessedDataBundle

logger = logging.getLogger(__name__)


@dataclass
class TrainHistory:
    epoch: List[int]
    train_loss: List[float]
    test_loss: List[float]


def _move_batch(
    batch: Tuple[torch.Tensor, torch.Tensor], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y = batch
    return x.to(device), y.to(device)


def build_optimizer(model: nn.Module, cfg: AppConfig) -> AdamW:
    beta1, beta2 = cfg.optimizer.betas
    return AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        betas=(beta1, beta2),
    )


def compute_pinn_loss(
    model: nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    criterion: nn.Module,
    cfg: AppConfig,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    x_in = x_batch.detach().clone().requires_grad_(True)
    pred = model(x_in)
    data_loss = criterion(pred, y_batch)

    bw_pred = pred[:, 0]
    il_pred = pred[:, 1]
    vpi_pred = pred[:, 2]
    length_batch = x_in[:, 7]

    losses: Dict[str, torch.Tensor] = {}
    total = data_loss

    if cfg.physics.lambda_bw_mon != 0:
        grads = torch.autograd.grad(
            bw_pred,
            x_in,
            grad_outputs=torch.ones_like(bw_pred),
            create_graph=True,
        )[0]
        d_bw_d_l = grads[:, 7]
        losses["bw_mon"] = torch.mean(torch.relu(d_bw_d_l) ** 2)
        total = total + cfg.physics.lambda_bw_mon * losses["bw_mon"]

    if cfg.physics.lambda_IL_mon != 0:
        grads = torch.autograd.grad(
            il_pred,
            x_in,
            grad_outputs=torch.ones_like(il_pred),
            create_graph=True,
        )[0]
        d_il_d_l = grads[:, 7]
        losses["IL_mon"] = torch.mean(torch.relu(-d_il_d_l) ** 2)
        total = total + cfg.physics.lambda_IL_mon * losses["IL_mon"]

    if cfg.physics.lambda_vpiL != 0:
        vpi_l = vpi_pred * length_batch
        grads = torch.autograd.grad(
            vpi_l,
            x_in,
            grad_outputs=torch.ones_like(vpi_l),
            create_graph=True,
        )[0]
        d_vpi_l_d_l = grads[:, 7]
        losses["vpiL"] = torch.mean(d_vpi_l_d_l**2)
        total = total + cfg.physics.lambda_vpiL * losses["vpiL"]

    if cfg.physics.lambda_smooth != 0:
        grads1 = torch.autograd.grad(
            bw_pred,
            x_in,
            grad_outputs=torch.ones_like(bw_pred),
            create_graph=True,
        )[0]
        d_bw_d_l = grads1[:, 7]
        grads2 = torch.autograd.grad(
            d_bw_d_l,
            x_in,
            grad_outputs=torch.ones_like(d_bw_d_l),
            create_graph=True,
        )[0]
        d2_bw_d_l2 = grads2[:, 7]
        losses["smooth"] = torch.mean(d2_bw_d_l2**2)
        total = total + cfg.physics.lambda_smooth * losses["smooth"]

    return total, data_loss, losses


@torch.no_grad()
def evaluate_full_batch_mse(
    model: nn.Module,
    x: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    if isinstance(x, torch.Tensor):
        x_t = x.to(device)
    else:
        x_t = torch.from_numpy(x).float().to(device)
    if isinstance(y, torch.Tensor):
        y_t = y.to(device)
    else:
        y_t = torch.from_numpy(y).float().to(device)
    pred = model(x_t)
    return float(nn.functional.mse_loss(pred, y_t).item())


def fit(
    model: nn.Module,
    cfg: AppConfig,
    bundle: ProcessedDataBundle,
    run_dir: Path,
    device: torch.device,
) -> TrainHistory:
    """按 notebook 风格训练，并记录每轮全量 train/test MSE。"""
    criterion = nn.MSELoss().to(device)
    optimizer = build_optimizer(model, cfg)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.csv"

    x_train_full = torch.from_numpy(bundle.X_train).float().to(device)
    y_train_full = torch.from_numpy(bundle.y_train).float().to(device)
    x_test_full = torch.from_numpy(bundle.X_test).float().to(device)
    y_test_full = torch.from_numpy(bundle.y_test).float().to(device)

    hist = TrainHistory(epoch=[], train_loss=[], test_loss=[])
    best_test = float("inf")

    with log_path.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["epoch", "train_loss", "test_loss"])

        for epoch in range(cfg.training.epochs):
            model.train()
            for x_batch, y_batch in bundle.train_loader:
                x_batch, y_batch = _move_batch((x_batch, y_batch), device)
                optimizer.zero_grad()
                loss, _, _ = compute_pinn_loss(model, x_batch, y_batch, criterion, cfg)
                loss.backward()
                optimizer.step()

            train_loss = evaluate_full_batch_mse(model, x_train_full, y_train_full, device)
            test_loss = evaluate_full_batch_mse(model, x_test_full, y_test_full, device)

            hist.epoch.append(epoch)
            hist.train_loss.append(train_loss)
            hist.test_loss.append(test_loss)
            writer.writerow([epoch, train_loss, test_loss])
            fcsv.flush()

            if epoch % 10 == 0 or epoch == 0:
                logger.info(
                    "Epoch %5d | Train %.6f | Test %.6f",
                    epoch,
                    train_loss,
                    test_loss,
                )

            payload = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "train_loss": train_loss,
                "test_loss": test_loss,
            }
            torch.save(payload, ckpt_dir / "last.pt")
            if test_loss < best_test:
                best_test = test_loss
                torch.save(payload, ckpt_dir / "best.pt")

    return hist


def load_weights(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    """从 checkpoint 载入 model_state。"""
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
