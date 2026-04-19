"""多输出回归损失：Huber（SmoothL1）与加权 MSE。"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class WeightedHuberLoss(nn.Module):
    """
    在标准化输出空间对每个目标维度加权 SmoothL1（Huber）。

    PyTorch 的 SmoothL1Loss 对应 beta 参数等价于 Huber 的 delta。
    """

    def __init__(self, delta: float = 1.0, weights: List[float] | None = None) -> None:
        super().__init__()
        self.delta = float(delta)
        self.register_buffer(
            "w",
            torch.tensor(weights if weights is not None else [1.0, 1.0, 1.0], dtype=torch.float32),
        )
        self._base = nn.SmoothL1Loss(reduction="none", beta=self.delta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred, target: shape (N, 3)
        Returns:
            标量损失（加权平均）
        """
        elem = self._base(pred, target)
        w = self.w.to(pred.device).view(1, -1)
        return (elem * w).sum(dim=1).mean()


class WeightedMSELoss(nn.Module):
    """逐维加权 MSE，再对 batch 平均。"""

    def __init__(self, weights: List[float] | None = None) -> None:
        super().__init__()
        self.register_buffer(
            "w",
            torch.tensor(weights if weights is not None else [1.0, 1.0, 1.0], dtype=torch.float32),
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err2 = (pred - target) ** 2
        w = self.w.to(pred.device).view(1, -1)
        return (err2 * w).sum(dim=1).mean()


def build_loss(cfg_loss) -> nn.Module:
    """根据配置构造损失函数。"""
    w = list(cfg_loss.target_weights)
    if cfg_loss.type == "huber":
        return WeightedHuberLoss(delta=cfg_loss.huber_delta, weights=w)
    if cfg_loss.type == "weighted_mse":
        return WeightedMSELoss(weights=w)
    raise ValueError(f"未知 loss.type: {cfg_loss.type}")
