"""YAML 配置加载与校验。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import yaml


@dataclass
class ModelConfig:
    input_dim: int = 8
    hidden_dims: List[int] = field(default_factory=lambda: [200, 300, 350, 300, 200])
    output_dim: int = 3
    batchnorm: bool = False
    dropout: float = 0.0
    residual: bool = False


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class SchedulerConfig:
    type: str = "cosine"  # cosine | plateau
    plateau_factor: float = 0.5
    plateau_patience: int = 10
    plateau_min_lr: float = 1e-6


@dataclass
class TrainingConfig:
    batch_size: int = 128
    epochs: int = 300
    early_stopping_patience: int = 30
    num_workers: int = 0


@dataclass
class LossConfig:
    type: str = "huber"  # huber | weighted_mse
    huber_delta: float = 1.0
    target_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])


@dataclass
class OutlierConfig:
    iqr_k: float = 1.5
    zscore_threshold: float = 4.0
    quantile_lower: float = 0.001
    quantile_upper: float = 0.999


@dataclass
class AppConfig:
    data_path: str
    split_ratios: List[float]
    random_seed: int
    remove_duplicate_rows: bool
    outlier_strategy: str
    outlier_config: OutlierConfig
    outlier_apply_to: str  # targets | all
    remove_nonpositive_vpi: bool
    filter_v_pi_range: bool
    v_pi_min: float
    v_pi_max: float
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training: TrainingConfig
    loss: LossConfig
    output_dir: str
    last_run_dir: Optional[str] = None

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "AppConfig":
        m = raw.get("model", {})
        o = raw.get("optimizer", {})
        s = raw.get("scheduler", {})
        t = raw.get("training", {})
        l = raw.get("loss", {})
        oc = raw.get("outlier_config", {})
        return AppConfig(
            data_path=str(raw["data_path"]),
            split_ratios=list(raw["split_ratios"]),
            random_seed=int(raw["random_seed"]),
            remove_duplicate_rows=bool(raw["remove_duplicate_rows"]),
            outlier_strategy=str(raw.get("outlier_strategy", "none")),
            outlier_config=OutlierConfig(
                iqr_k=float(oc.get("iqr_k", 1.5)),
                zscore_threshold=float(oc.get("zscore_threshold", 4.0)),
                quantile_lower=float(oc.get("quantile_lower", 0.001)),
                quantile_upper=float(oc.get("quantile_upper", 0.999)),
            ),
            outlier_apply_to=str(raw.get("outlier_apply_to", "targets")),
            remove_nonpositive_vpi=bool(raw.get("remove_nonpositive_vpi", False)),
            filter_v_pi_range=bool(raw.get("filter_v_pi_range", True)),
            v_pi_min=float(raw.get("v_pi_min", 0.0)),
            v_pi_max=float(raw.get("v_pi_max", 500.0)),
            model=ModelConfig(
                input_dim=int(m.get("input_dim", 8)),
                hidden_dims=list(m.get("hidden_dims", [200, 300, 350, 300, 200])),
                output_dim=int(m.get("output_dim", 3)),
                batchnorm=bool(m.get("batchnorm", False)),
                dropout=float(m.get("dropout", 0.0)),
                residual=bool(m.get("residual", False)),
            ),
            optimizer=OptimizerConfig(
                name=str(o.get("name", "adamw")),
                lr=float(o.get("lr", 1e-3)),
                weight_decay=float(o.get("weight_decay", 1e-4)),
            ),
            scheduler=SchedulerConfig(
                type=str(s.get("type", "cosine")),
                plateau_factor=float(s.get("plateau_factor", 0.5)),
                plateau_patience=int(s.get("plateau_patience", 10)),
                plateau_min_lr=float(s.get("plateau_min_lr", 1e-6)),
            ),
            training=TrainingConfig(
                batch_size=int(t.get("batch_size", 128)),
                epochs=int(t.get("epochs", 300)),
                early_stopping_patience=int(t.get("early_stopping_patience", 30)),
                num_workers=int(t.get("num_workers", 0)),
            ),
            loss=LossConfig(
                type=str(l.get("type", "huber")),
                huber_delta=float(l.get("huber_delta", 1.0)),
                target_weights=[float(x) for x in l.get("target_weights", [1.0, 1.0, 1.0])],
            ),
            output_dir=str(raw.get("output_dir", "results")),
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
    cfg = AppConfig.from_dict(raw)
    sr = cfg.split_ratios
    if len(sr) != 3:
        raise ValueError("split_ratios 必须为长度为 3 的列表 [train, val, test]")
    if abs(sum(sr) - 1.0) > 1e-6:
        raise ValueError(f"split_ratios 之和必须为 1，当前为 {sum(sr)}")
    if cfg.outlier_strategy not in ("none", "iqr", "zscore", "quantile_clip"):
        raise ValueError(f"未知 outlier_strategy: {cfg.outlier_strategy}")
    if cfg.outlier_apply_to not in ("targets", "all"):
        raise ValueError("outlier_apply_to 必须为 targets 或 all")
    if len(cfg.loss.target_weights) != 3:
        raise ValueError("loss.target_weights 长度必须为 3")
    if cfg.filter_v_pi_range and cfg.v_pi_min >= cfg.v_pi_max:
        raise ValueError("启用 filter_v_pi_range 时须满足 v_pi_min < v_pi_max")
    return cfg
