"""随机种子、日志与路径工具。"""

from __future__ import annotations

import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """设置 random / numpy / torch 与 cuDNN 行为以保证可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 部分算子仍可能非确定，但满足常规科研复现需求
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (TypeError, AttributeError):
        pass


def make_run_dir(base_output: str | Path) -> Path:
    """在 output_dir 下创建带时间戳的 run 目录。"""
    base = Path(base_output)
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    return run_dir


def setup_logging(
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    配置根 logger：控制台简洁格式，可选文件完整日志。
    """
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    fmt_console = logging.Formatter("%(levelname)s %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt_console)
    root.addHandler(ch)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        root.addHandler(fh)

    return logging.getLogger("mzm")


def find_latest_run(output_dir: str | Path) -> Path:
    """返回 output_dir 下按修改时间最近的一个 run_* 目录。"""
    base = Path(output_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"输出目录不存在: {base}")
    runs = sorted(
        [p for p in base.iterdir() if p.is_dir() and p.name.startswith("run_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"在 {base} 下未找到 run_* 子目录，请先训练模型。")
    return runs[0]


def resolve_path(path: str | Path, base: Optional[Path] = None) -> Path:
    """将路径解析为绝对路径；若提供 base，则相对 base 解析。"""
    p = Path(path)
    if p.is_absolute():
        return p
    if base is not None:
        return (base / p).resolve()
    return p.resolve()
