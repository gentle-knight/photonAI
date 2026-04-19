"""原始 txt 数据加载、字段定义与清洗前质量检查。"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 与论文/数据说明一致的物理列名（含 +）
INPUT_COLUMNS = [
    "PN_offset",
    "Bias_V",
    "Core_width",
    "P+_width",
    "N+_width",
    "P_width",
    "N_width",
    "Phase_length",
]
TARGET_COLUMNS = ["BW_3dB", "IL", "V_pi"]
ALL_COLUMNS = INPUT_COLUMNS + TARGET_COLUMNS
EXPECTED_COLS = 11
# txt 行内从左到右第 11 个字段即 V_pi，与 DataFrame 最后一列一致，用作物理可信区间清洗门控
V_PI_TXT_1BASED_INDEX = 11


def _strip_optional_list_brackets(line: str) -> str:
    """
    去掉仿真/导出常见的整行方括号包裹，例如::

        [-2.15e-07, -10.0, ...] -> -2.15e-07, -10.0, ...

    若行首无 ``[`` 或行尾无 ``]``，则原样返回（兼容无括号格式）。
    """
    s = line.strip()
    if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
        return s[1:-1].strip()
    return s


def load_raw_txt(path: str | Path) -> pd.DataFrame:
    """
    从 txt 读取数据：逗号分隔、11 列浮点；跳过空行与纯空白行。

    支持两种常见行格式（等价）::

        a,b,c,...,k
        [a, b, c, ..., k]

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 行格式或列数不合法
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"数据文件不存在: {path.resolve()}。请将 11 列逗号分隔的 txt 放到该路径，"
            f"或修改 configs/default.yaml 中的 data_path。"
        )

    rows: list[list[float]] = []
    bad_lines: list[tuple[int, str]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            line = _strip_optional_list_brackets(line)
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != EXPECTED_COLS:
                bad_lines.append((line_no, f"列数={len(parts)}，期望 {EXPECTED_COLS}"))
                continue
            try:
                vals = [float(p) for p in parts]
            except ValueError as e:
                bad_lines.append((line_no, str(e)))
                continue
            rows.append(vals)

    if bad_lines:
        preview = "; ".join(f"行{k}:{msg}" for k, msg in bad_lines[:5])
        if len(bad_lines) > 5:
            preview += f"; ... 共 {len(bad_lines)} 行有问题"
        raise ValueError(f"数据解析失败（{path}）。{preview}")

    if not rows:
        raise ValueError(f"文件为空或无非空数据行: {path}")

    df = pd.DataFrame(rows, columns=ALL_COLUMNS)
    logger.info(
        "已加载 %d 行，%d 列（第 %d 列为 V_pi，用于物理区间门控）",
        len(df),
        len(df.columns),
        V_PI_TXT_1BASED_INDEX,
    )
    return df


def basic_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """返回 describe() 风格的汇总（含 count/mean/std/min/max）。"""
    return df.describe().T


def _extreme_report(series: pd.Series, name: str, z: float = 5.0) -> dict:
    """单序列极端值统计：|z-score|>z 的个数（相对自身均值方差）。"""
    s = series.astype(float)
    mu, sig = float(s.mean()), float(s.std(ddof=0))
    if sig < 1e-12:
        return {"column": name, "z_threshold": z, "extreme_count": 0}
    zscores = (s - mu) / sig
    extreme = int((zscores.abs() > z).sum())
    return {"column": name, "z_threshold": z, "extreme_count": extreme, "min": float(s.min()), "max": float(s.max())}


def quality_report_before_clean(
    df: pd.DataFrame,
    v_pi_min: float = 0.0,
    v_pi_max: float = 500.0,
) -> dict:
    """
    清洗前数据质量报告：重复行、同输入异输出、V_pi 相对给定区间的越界计数、V_pi<=0、目标极端值。
    不修改 DataFrame。
    """
    n = len(df)
    dup_mask = df.duplicated(keep=False)
    n_dup_rows = int(dup_mask.sum())
    # 若启用去重，将删除的“重复出现行数” = 总行数 - 去重后行数
    rows_removed_if_dedupe = int(n - df.drop_duplicates().shape[0])

    def _n_unique_target_rows(sub: pd.DataFrame) -> int:
        return sub[TARGET_COLUMNS].drop_duplicates().shape[0]

    same_x_diff_y = 0
    for _, sub in df.groupby(INPUT_COLUMNS, dropna=False):
        if len(sub) <= 1:
            continue
        if _n_unique_target_rows(sub) > 1:
            same_x_diff_y += len(sub)

    vpi_nonpositive = int((df["V_pi"] <= 0).sum())
    vpi_series = df["V_pi"].astype(float)
    v_pi_out_of_range_count = int(((vpi_series < v_pi_min) | (vpi_series > v_pi_max)).sum())

    target_extremes = [_extreme_report(df[c], c) for c in TARGET_COLUMNS]

    report = {
        "n_rows": n,
        "duplicate_row_mask_count": n_dup_rows,
        "rows_removed_if_drop_duplicates": rows_removed_if_dedupe,
        "rows_with_same_inputs_differing_outputs": same_x_diff_y,
        "v_pi_gate_inclusive_range": {"min": v_pi_min, "max": v_pi_max},
        "v_pi_out_of_range_count": v_pi_out_of_range_count,
        "v_pi_nonpositive_count": vpi_nonpositive,
        "target_extreme_z5": target_extremes,
    }
    return report


def summarize_for_console(df: pd.DataFrame, q: dict) -> str:
    """简短人类可读摘要。"""
    gate = q.get("v_pi_gate_inclusive_range", {})
    lo, hi = gate.get("min", 0.0), gate.get("max", 500.0)
    lines = [
        f"行数={len(df)}",
        f"去重可删除行数={q['rows_removed_if_drop_duplicates']}",
        f"同输入异输出涉及行数={q['rows_with_same_inputs_differing_outputs']}",
        f"V_pi 越界 [ {lo}, {hi} ] 行数={q.get('v_pi_out_of_range_count', 'n/a')}",
        f"V_pi<=0 行数={q['v_pi_nonpositive_count']}",
    ]
    return "; ".join(lines)
