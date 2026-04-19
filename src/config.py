"""YAML 配置加载与校验。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import yaml


@dataclass
class DataConfig:
    test_size: float = 0.1
    random_state: int = 123
    filter_v_pi_max: float = 500.0


@dataclass
class ModelConfig:
    input_dim: int = 8
    output_dim: int = 3
    hidden_dims: List[int] = field(default_factory=lambda: [64, 128, 64])
    n_experts: int = 60
    gating_hidden: int = 8
    dropout_rate: float = 0.0
    use_bn: bool = True
    activation: str = "relu"


@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    weight_decay: float = 0.05
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])


@dataclass
class TrainingConfig:
    batch_size: int = 128
    epochs: int = 100
    num_workers: int = 0


@dataclass
class PhysicsConfig:
    lambda_bw_mon: float = 0.0
    lambda_IL_mon: float = 0.3
    lambda_vpiL: float = 0.005
    lambda_smooth: float = 0.1


@dataclass
class AppConfig:
    data_path: str
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    physics: PhysicsConfig
    output_dir: str
    best_hyperparams_path: Optional[str] = None
    last_run_dir: Optional[str] = None

    @staticmethod
    def from_dict(raw: dict[str, Any], cfg_dir: Path) -> "AppConfig":
        data_raw = raw.get("data", {})
        model_raw = raw.get("model", {})
        optimizer_raw = raw.get("optimizer", {})
        training_raw = raw.get("training", {})
        physics_raw = raw.get("physics", {})
        best_hyperparams_path = raw.get("best_hyperparams_path")

        if best_hyperparams_path:
            hp_path = Path(best_hyperparams_path)
            if not hp_path.is_absolute():
                cand = (cfg_dir / hp_path).resolve()
                if cand.is_file():
                    hp_path = cand
                else:
                    hp_path = (cfg_dir.parent / hp_path).resolve()
            with hp_path.open("r", encoding="utf-8") as f:
                hp_raw = json.load(f)
            physics_raw = {**hp_raw.get("best_config", {}), **physics_raw}
            best_hyperparams_path = str(hp_path)

        return AppConfig(
            data_path=str(raw["data_path"]),
            data=DataConfig(
                test_size=float(data_raw.get("test_size", raw.get("test_size", 0.1))),
                random_state=int(data_raw.get("random_state", raw.get("random_seed", 123))),
                filter_v_pi_max=float(
                    data_raw.get("filter_v_pi_max", raw.get("v_pi_max", 500.0))
                ),
            ),
            model=ModelConfig(
                input_dim=int(model_raw.get("input_dim", 8)),
                output_dim=int(model_raw.get("output_dim", 3)),
                hidden_dims=list(model_raw.get("hidden_dims", [64, 128, 64])),
                n_experts=int(model_raw.get("n_experts", 60)),
                gating_hidden=int(model_raw.get("gating_hidden", 8)),
                dropout_rate=float(model_raw.get("dropout_rate", 0.0)),
                use_bn=bool(model_raw.get("use_bn", True)),
                activation=str(model_raw.get("activation", "relu")),
            ),
            optimizer=OptimizerConfig(
                lr=float(optimizer_raw.get("lr", 1e-3)),
                weight_decay=float(optimizer_raw.get("weight_decay", 0.05)),
                betas=[float(x) for x in optimizer_raw.get("betas", [0.9, 0.999])],
            ),
            training=TrainingConfig(
                batch_size=int(training_raw.get("batch_size", 128)),
                epochs=int(training_raw.get("epochs", 100)),
                num_workers=int(training_raw.get("num_workers", 0)),
            ),
            physics=PhysicsConfig(
                lambda_bw_mon=float(physics_raw.get("lambda_bw_mon", 0.0)),
                lambda_IL_mon=float(physics_raw.get("lambda_IL_mon", 0.3)),
                lambda_vpiL=float(physics_raw.get("lambda_vpiL", 0.005)),
                lambda_smooth=float(physics_raw.get("lambda_smooth", 0.1)),
            ),
            output_dir=str(raw.get("output_dir", "results")),
            best_hyperparams_path=best_hyperparams_path,
            last_run_dir=raw.get("last_run_dir"),
        )


def load_config(path: str | Path) -> AppConfig:
    """从 YAML 文件加载配置并做基本校验。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("YAML 根节点必须是字典")
    cfg = AppConfig.from_dict(raw, path.parent.resolve())
    if not 0.0 < cfg.data.test_size < 1.0:
        raise ValueError("data.test_size 必须在 (0, 1) 之间")
    if cfg.data.filter_v_pi_max <= 0:
        raise ValueError("data.filter_v_pi_max 必须 > 0")
    if cfg.model.input_dim != 8:
        raise ValueError("model.input_dim 必须为 8")
    if cfg.model.output_dim != 3:
        raise ValueError("model.output_dim 必须为 3")
    if not cfg.model.hidden_dims:
        raise ValueError("model.hidden_dims 不能为空")
    if cfg.model.n_experts < 1:
        raise ValueError("model.n_experts 必须 >= 1")
    if cfg.model.gating_hidden < 1:
        raise ValueError("model.gating_hidden 必须 >= 1")
    if cfg.model.activation not in ("relu", "gaussian"):
        raise ValueError("model.activation 必须为 relu 或 gaussian")
    if len(cfg.optimizer.betas) != 2:
        raise ValueError("optimizer.betas 长度必须为 2")
    if cfg.training.batch_size < 1 or cfg.training.epochs < 1:
        raise ValueError("training.batch_size 与 training.epochs 必须 >= 1")
    return cfg
