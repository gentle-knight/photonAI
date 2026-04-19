"""统一 CLI：train / eval / infer。"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import torch

from src.config import load_config
from src.data import load_raw_txt, quality_report_before_clean, summarize_for_console
from src.evaluate import export_test_predictions_csv, run_full_evaluation
from src.model import MLPRegressor
from src.plots import generate_all_figures
from src.preprocess import prepare_training_data, rebuild_bundle_for_eval
from src.trainer import fit
from src.utils import (
    find_latest_run,
    make_run_dir,
    resolve_path,
    set_global_seed,
    setup_logging,
)

logger = logging.getLogger("mzm")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_cfg_path(path: str) -> Path:
    p = Path(path)
    if p.is_file():
        return p.resolve()
    cand = _project_root() / path
    if cand.is_file():
        return cand.resolve()
    raise FileNotFoundError(f"找不到配置文件: {path}")


def _resolve_run_dir(cfg_path: Path, explicit: str | None, output_dir: str | None) -> Path:
    if explicit:
        rd = Path(explicit).resolve()
        if not rd.is_dir():
            raise FileNotFoundError(f"run_dir 不存在: {rd}")
        return rd
    base = Path(output_dir) if output_dir else None
    if base is None:
        cfg = load_config(cfg_path)
        base = Path(cfg.output_dir)
    return find_latest_run(base)


def _write_summary_md(run_dir: Path, summary: dict) -> None:
    lines = ["# 评估摘要", ""]
    for split, block in summary.get("splits", {}).items():
        lines.append(f"## {split}")
        lines.append("")
        lines.append(
            f"- **损失（标准化输出空间 Huber/MSE 准则）**: {block['loss']:.6f}"
        )
        for space, label in ("normalized", "标准化空间"), ("physical", "物理量空间"):
            sub = block[space]
            lines.append(f"- **{label}** — 平均 MAE: {sub['overall_mae']:.6f}；"
                         f"平均 RMSE: {sub['overall_rmse']:.6f}；整体 R²: {sub['overall_r2']:.6f}")
            for t, vals in sub["per_target"].items():
                lines.append(
                    f"  - `{t}`: MAE={vals['mae']:.6f}, RMSE={vals['rmse']:.6f}, R²={vals['r2']:.6f}"
                )
        lines.append("")
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def cmd_train(args: argparse.Namespace) -> None:
    cfg_path = _resolve_cfg_path(args.config)
    cfg = load_config(cfg_path)
    set_global_seed(cfg.random_seed)

    run_dir = make_run_dir(resolve_path(cfg.output_dir, _project_root()))
    shutil.copy2(cfg_path, run_dir / "config_snapshot.yaml")
    setup_logging(run_dir / "training.log")

    data_path = resolve_path(cfg.data_path, _project_root())
    df = load_raw_txt(data_path)
    q = quality_report_before_clean(df, cfg.v_pi_min, cfg.v_pi_max)
    logger.info("数据质量(清洗前): %s", summarize_for_console(df, q))

    bundle = prepare_training_data(df, cfg, run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPRegressor(
        input_dim=cfg.model.input_dim,
        hidden_dims=cfg.model.hidden_dims,
        output_dim=cfg.model.output_dim,
        batchnorm=cfg.model.batchnorm,
        dropout=cfg.model.dropout,
        residual=cfg.model.residual,
    ).to(device)

    history = fit(model, cfg, bundle.train_loader, bundle.val_loader, run_dir, device)
    model_eval, summary = run_full_evaluation(cfg, bundle, run_dir, device)
    export_test_predictions_csv(
        bundle, model_eval, device, run_dir / "test_predictions.csv"
    )
    _write_summary_md(run_dir, summary)
    generate_all_figures(cfg, bundle, run_dir, history, device)
    logger.info("训练与评估完成，结果目录: %s", run_dir)


def cmd_eval(args: argparse.Namespace) -> None:
    cfg_path = _resolve_cfg_path(args.config)
    run_dir = _resolve_run_dir(cfg_path, args.run_dir, args.output_dir)
    snap = run_dir / "config_snapshot.yaml"
    cfg = load_config(snap if snap.is_file() else cfg_path)
    set_global_seed(cfg.random_seed)
    setup_logging(run_dir / "eval.log")

    data_path = resolve_path(cfg.data_path, _project_root())
    df = load_raw_txt(data_path)
    bundle = rebuild_bundle_for_eval(df, cfg, run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_eval, summary = run_full_evaluation(cfg, bundle, run_dir, device)
    export_test_predictions_csv(
        bundle, model_eval, device, run_dir / "test_predictions.csv"
    )
    _write_summary_md(run_dir, summary)
    logger.info("评估完成，已更新 %s 下 metrics/summary/test_predictions", run_dir)


def cmd_infer(args: argparse.Namespace) -> None:
    cfg_path = _resolve_cfg_path(args.config)
    run_dir = _resolve_run_dir(cfg_path, args.run_dir, args.output_dir)
    snap = run_dir / "config_snapshot.yaml"
    cfg = load_config(snap if snap.is_file() else cfg_path)
    set_global_seed(cfg.random_seed)
    setup_logging(None)

    from src.infer import run_inference

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output) if args.output else run_dir / "inference_output.csv"
    run_inference(cfg, run_dir, Path(args.input), out, device)
    logger.info("推理完成: %s", out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MZM MLP 训练 / 评估 / 推理")
    sub = p.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("train", help="训练模型")
    pt.add_argument("--config", type=str, default="configs/default.yaml")
    pt.set_defaults(func=cmd_train)

    pe = sub.add_parser("eval", help="在 train/val/test 上重新评估并导出 CSV")
    pe.add_argument("--config", type=str, default="configs/default.yaml")
    pe.add_argument("--run-dir", type=str, default=None, help="指定某次训练输出目录")
    pe.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="在未指定 run-dir 时用于搜索最新 run_* 的根目录",
    )
    pe.set_defaults(func=cmd_eval)

    pi = sub.add_parser("infer", help="批量推理")
    pi.add_argument("--config", type=str, default="configs/default.yaml")
    pi.add_argument("--input", type=str, required=True, help="8 列输入 csv/txt")
    pi.add_argument("--output", type=str, default=None, help="输出预测 csv 路径")
    pi.add_argument("--run-dir", type=str, default=None)
    pi.add_argument("--output-dir", type=str, default=None)
    pi.set_defaults(func=cmd_infer)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
