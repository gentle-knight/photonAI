"""MoE PINN 模型定义。"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class GaussianActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(x**2))


def weights_init(layer_in: nn.Module) -> None:
    """与 notebook 保持一致的 Kaiming 初始化。"""
    if isinstance(layer_in, nn.Linear):
        nn.init.kaiming_uniform_(layer_in.weight)
        if layer_in.bias is not None:
            layer_in.bias.data.fill_(0.0)


def build_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gaussian":
        return GaussianActivation()
    raise ValueError(f"未知激活函数: {name}")


class ExpertNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation_fn: nn.Module,
        dropout_rate: float = 0.0,
        use_bn: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(type(activation_fn)() if isinstance(activation_fn, nn.ReLU) else activation_fn.__class__())
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixtureOfExperts(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        n_experts: int = 3,
        activation_fn: nn.Module | None = None,
        gating_hidden: int = 32,
        dropout_rate: float = 0.0,
        use_bn: bool = False,
    ) -> None:
        super().__init__()
        act = activation_fn if activation_fn is not None else nn.ReLU()
        self.experts = nn.ModuleList(
            [
                ExpertNN(
                    input_dim,
                    output_dim,
                    hidden_dims,
                    act,
                    dropout_rate=dropout_rate,
                    use_bn=use_bn,
                )
                for _ in range(n_experts)
            ]
        )
        self.gating = nn.Sequential(
            nn.Linear(input_dim, gating_hidden),
            nn.ReLU(),
            nn.Linear(gating_hidden, n_experts),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_weights = self.gating(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        return torch.bmm(expert_outputs, gate_weights.unsqueeze(2)).squeeze(2)


def create_model_from_config(cfg) -> MixtureOfExperts:
    return MixtureOfExperts(
        input_dim=cfg.model.input_dim,
        output_dim=cfg.model.output_dim,
        hidden_dims=cfg.model.hidden_dims,
        n_experts=cfg.model.n_experts,
        activation_fn=build_activation(cfg.model.activation),
        gating_hidden=cfg.model.gating_hidden,
        dropout_rate=cfg.model.dropout_rate,
        use_bn=cfg.model.use_bn,
    )
