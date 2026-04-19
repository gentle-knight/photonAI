"""最小冒烟测试：MoE + PINN 流程可运行。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.config import load_config
from src.data import load_raw_txt
from src.model import create_model_from_config
from src.preprocess import inverse_transform_targets, prepare_training_data
from src.trainer import compute_pinn_loss, fit


def _write_synthetic_txt(path: Path, n: int = 96) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 8))
    y = np.zeros((n, 3))
    length = np.abs(x[:, 7]) + 0.5
    y[:, 0] = 2.0 - 0.2 * length + rng.normal(scale=0.05, size=n)
    y[:, 1] = 0.3 + 0.1 * length + rng.normal(scale=0.03, size=n)
    y[:, 2] = 20.0 / length + rng.normal(scale=0.2, size=n)
    mat = np.hstack([x, y])
    path.write_text("\n".join(",".join(str(v) for v in row) for row in mat), encoding="utf-8")


def _make_cfg(tmp_path: Path, data_txt: Path, epochs: int = 1) -> Path:
    cfg_dict = {
        "data_path": str(data_txt),
        "data": {
            "test_size": 0.1,
            "random_state": 123,
            "filter_v_pi_max": 500.0,
        },
        "model": {
            "input_dim": 8,
            "output_dim": 3,
            "hidden_dims": [16, 16],
            "n_experts": 4,
            "gating_hidden": 4,
            "dropout_rate": 0.0,
            "use_bn": False,
            "activation": "relu",
        },
        "optimizer": {
            "lr": 0.001,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
        },
        "training": {
            "batch_size": 16,
            "epochs": epochs,
            "num_workers": 0,
        },
        "physics": {
            "lambda_bw_mon": 0.1,
            "lambda_IL_mon": 0.1,
            "lambda_vpiL": 0.05,
            "lambda_smooth": 0.01,
        },
        "output_dir": str(tmp_path / "results"),
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")
    return cfg_path


def test_moe_forward_shape(tmp_path: Path) -> None:
    data_txt = tmp_path / "data.txt"
    _write_synthetic_txt(data_txt, n=32)
    cfg = load_config(_make_cfg(tmp_path, data_txt))
    model = create_model_from_config(cfg)
    x = torch.randn(5, 8)
    y = model(x)
    assert y.shape == (5, 3)


def test_pinn_loss_backpropagates(tmp_path: Path) -> None:
    data_txt = tmp_path / "data.txt"
    _write_synthetic_txt(data_txt, n=40)
    cfg = load_config(_make_cfg(tmp_path, data_txt))
    model = create_model_from_config(cfg)
    x = torch.randn(8, 8)
    y = torch.randn(8, 3)
    loss, data_loss, terms = compute_pinn_loss(model, x, y, torch.nn.MSELoss(), cfg)
    loss.backward()
    assert float(loss.detach()) >= float(data_loss.detach())
    assert "IL_mon" in terms
    assert any(p.grad is not None for p in model.parameters())


def test_one_epoch_training_pipeline(tmp_path: Path) -> None:
    data_txt = tmp_path / "data.txt"
    _write_synthetic_txt(data_txt, n=80)
    cfg = load_config(_make_cfg(tmp_path, data_txt, epochs=1))
    df = load_raw_txt(data_txt)

    run_dir = tmp_path / "run0"
    run_dir.mkdir()
    (run_dir / "figures").mkdir()
    (run_dir / "checkpoints").mkdir()

    bundle = prepare_training_data(df, cfg, run_dir)
    device = torch.device("cpu")
    model = create_model_from_config(cfg).to(device)
    history = fit(model, cfg, bundle, run_dir, device)
    assert (run_dir / "checkpoints" / "last.pt").is_file()
    assert len(history.train_loss) == 1
    meta = json.loads((run_dir / "cleaning_meta.json").read_text(encoding="utf-8"))
    assert meta["n_train"] > 0
    assert meta["n_test"] > 0
    restored = inverse_transform_targets(bundle.y_test[:3], bundle.y_scalers)
    assert restored.shape == (3, 3)
