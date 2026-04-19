"""离线推理：从 txt/csv 读取 8 维输入，输出反标准化后的三目标预测。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from src.config import AppConfig, load_config
from src.data import INPUT_COLUMNS, TARGET_COLUMNS, _strip_optional_list_brackets
from src.model import MLPRegressor
from src.preprocess import load_scalers
from src.trainer import load_weights

logger = logging.getLogger(__name__)


def _read_txt_inputs_eight_cols(path: Path) -> pd.DataFrame:
    """读取无表头 txt：逗号分隔，支持整行 ``[...]`` 包裹；取前 8 列为输入。"""
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            s = raw.strip()
            if not s:
                continue
            s = _strip_optional_list_brackets(s)
            parts = [p.strip() for p in s.split(",")]
            if len(parts) < len(INPUT_COLUMNS):
                raise ValueError(
                    f"{path} 第 {line_no} 行列数不足（{len(parts)}），至少需要 {len(INPUT_COLUMNS)} 列输入"
                )
            try:
                row = [float(parts[j]) for j in range(len(INPUT_COLUMNS))]
            except ValueError as e:
                raise ValueError(f"{path} 第 {line_no} 行解析失败: {e}") from e
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} 无有效数据行")
    return pd.DataFrame(rows, columns=INPUT_COLUMNS)


def _read_inputs_table(path: Path) -> pd.DataFrame:
    """读取 8 列输入：支持逗号分隔 txt（可无表头、可带方括号）或 csv。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"输入文件不存在: {path}")
    if path.suffix.lower() in {".csv"}:
        df = pd.read_csv(path)
    else:
        df = _read_txt_inputs_eight_cols(path)
    missing = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"输入缺少列: {missing}；需要列 {INPUT_COLUMNS}")
    return df[INPUT_COLUMNS].astype(float)


def run_inference(
    cfg: AppConfig,
    run_dir: Path,
    input_path: Path,
    output_csv: Path,
    device: torch.device,
) -> None:
    """
    载入 best 模型与 scaler，对输入表进行批量推理并写出 CSV（物理量空间）。
    """
    X = _read_inputs_table(Path(input_path)).to_numpy(dtype=np.float32)
    X_scaler, y_scaler = load_scalers(run_dir)
    Xn = X_scaler.transform(X)

    model = MLPRegressor(
        input_dim=cfg.model.input_dim,
        hidden_dims=cfg.model.hidden_dims,
        output_dim=cfg.model.output_dim,
        batchnorm=cfg.model.batchnorm,
        dropout=cfg.model.dropout,
        residual=cfg.model.residual,
    ).to(device)
    load_weights(model, run_dir / "checkpoints" / "best.pt", device)

    model.eval()
    with torch.no_grad():
        pred_n = model(torch.from_numpy(Xn).float().to(device)).cpu().numpy()
    pred_p = y_scaler.inverse_transform(pred_n)

    out = pd.DataFrame(X, columns=INPUT_COLUMNS)
    for j, name in enumerate(TARGET_COLUMNS):
        out[f"pred_{name}"] = pred_p[:, j]
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    logger.info("推理完成，写入 %s", output_csv)


def infer_cli(
    config_path: str,
    run_dir: Path,
    input_path: str,
    output_csv: Optional[str] = None,
) -> None:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = (
        Path(output_csv)
        if output_csv is not None
        else run_dir / "inference_output.csv"
    )
    run_inference(cfg, run_dir, Path(input_path), out, device)
