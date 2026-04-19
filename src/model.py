"""可配置 MLP 回归模型。"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def kaiming_init_module(m: nn.Module) -> None:
    """对 Linear 使用 Kaiming uniform（ReLU），偏置置零。"""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class MLPRegressor(nn.Module):
    """
    多层感知机回归：输入 8 维，输出 3 维。

    可选 BatchNorm1d、Dropout、以及在相邻层维度相等时的残差相加。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        batchnorm: bool = False,
        dropout: float = 0.0,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.residual = residual
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        self._hidden_blocks = nn.ModuleList()
        for i in range(len(dims) - 2):
            in_d, out_d = dims[i], dims[i + 1]
            seq_layers: list[nn.Module] = [nn.Linear(in_d, out_d)]
            if batchnorm:
                seq_layers.append(nn.BatchNorm1d(out_d))
            seq_layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                seq_layers.append(nn.Dropout(p=dropout))
            self._hidden_blocks.append(nn.Sequential(*seq_layers))
        self._head = nn.Linear(dims[-2], dims[-1])
        self.apply(kaiming_init_module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self._hidden_blocks:
            inp = h
            h = block(inp)
            if self.residual and inp.shape[-1] == h.shape[-1]:
                h = h + inp
        return self._head(h)
