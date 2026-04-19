"""最小冒烟测试：模型 shape 与单 epoch 训练不报错。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.config import load_config
from src.data import load_raw_txt
from src.model import MLPRegressor
from src.preprocess import prepare_training_data
from src.trainer import fit


def _write_synthetic_txt(path: Path, n: int = 64) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 8))
    y = np.zeros((n, 3))
    y[:, 0] = rng.normal(size=n)
    y[:, 1] = rng.normal(size=n)
    # 第 11 列 V_pi 落在默认物理门控 [0, 500] 内
    y[:, 2] = rng.uniform(1.0, 400.0, size=n)
    mat = np.hstack([x, y])
    lines = [",".join(str(v) for v in row) for row in mat]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_mlp_forward_shape() -> None:
    m = MLPRegressor(8, [16, 16], 3, batchnorm=False, dropout=0.0, residual=False)
    x = torch.randn(5, 8)
    y = m(x)
    assert y.shape == (5, 3)


def test_one_epoch_training_pipeline(tmp_path: Path) -> None:
    data_txt = tmp_path / "data.txt"
    _write_synthetic_txt(data_txt, n=80)

    cfg_dict = {
        "data_path": str(data_txt),
        "split_ratios": [0.7, 0.15, 0.15],
        "random_seed": 1,
        "remove_duplicate_rows": False,
        "outlier_strategy": "none",
        "outlier_apply_to": "targets",
        "outlier_config": {
            "iqr_k": 1.5,
            "zscore_threshold": 4.0,
            "quantile_lower": 0.001,
            "quantile_upper": 0.999,
        },
        "remove_nonpositive_vpi": False,
        "filter_v_pi_range": True,
        "v_pi_min": 0.0,
        "v_pi_max": 500.0,
        "model": {
            "input_dim": 8,
            "hidden_dims": [32, 32],
            "output_dim": 3,
            "batchnorm": False,
            "dropout": 0.0,
            "residual": False,
        },
        "optimizer": {"name": "adamw", "lr": 0.01, "weight_decay": 0.0},
        "scheduler": {
            "type": "cosine",
            "plateau_factor": 0.5,
            "plateau_patience": 10,
            "plateau_min_lr": 1e-6,
        },
        "training": {
            "batch_size": 16,
            "epochs": 1,
            "early_stopping_patience": 1,
            "num_workers": 0,
        },
        "loss": {"type": "huber", "huber_delta": 1.0, "target_weights": [1.0, 1.0, 1.0]},
        "output_dir": str(tmp_path / "results"),
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")

    cfg = load_config(cfg_path)
    df = load_raw_txt(data_txt)

    run_dir = tmp_path / "run0"
    run_dir.mkdir()
    (run_dir / "figures").mkdir()
    (run_dir / "checkpoints").mkdir()

    bundle = prepare_training_data(df, cfg, run_dir)
    device = torch.device("cpu")
    model = MLPRegressor(
        input_dim=cfg.model.input_dim,
        hidden_dims=cfg.model.hidden_dims,
        output_dim=cfg.model.output_dim,
        batchnorm=cfg.model.batchnorm,
        dropout=cfg.model.dropout,
        residual=cfg.model.residual,
    ).to(device)
    fit(model, cfg, bundle.train_loader, bundle.val_loader, run_dir, device)
    assert (run_dir / "checkpoints" / "best.pt").is_file()
    meta = json.loads((run_dir / "cleaning_meta.json").read_text(encoding="utf-8"))
    assert meta["n_train"] > 0
