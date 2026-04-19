# MZM MoE PINN（PyTorch）

本仓库已从原先的 MLP 基线迁移为 **Mixture-of-Experts + Physics-Informed Neural Network** 训练流程，目标是尽量对齐 `MZM_MoE_PINN_Model.ipynb` 的训练行为，同时保留仓库化的命令行入口、结果目录和复现产物。

## 当前训练管线

- **输入**：8 个器件/偏置参数
- **输出**：`BW_3dB`、`IL`、`V_pi`
- **模型**：MoE，包含多个专家网络与一个 gating 网络
- **数据清洗**：保留 `V_pi < 500`
- **数据划分**：`train_test_split(test_size=0.1, random_state=123)`
- **标准化**：
  - `X` 使用一个 `StandardScaler`
  - `Y` 的三个目标分别使用独立的 `StandardScaler`
- **损失**：
  - 数据项：标准化空间 `MSE`
  - 物理项：`dBW/dL <= 0`、`dIL/dL >= 0`、`d(V_pi*L)/dL ~= 0`、`d2BW/dL2` 平滑项
- **优化器**：`AdamW(lr=1e-3, weight_decay=0.05, betas=(0.9, 0.999))`
- **训练方式**：固定 `100` epoch，无早停；每轮计算全量 train/test MSE

默认物理约束权重来自 `best_hyperparams.json`：

```json
{
  "lambda_bw_mon": 0.0,
  "lambda_IL_mon": 0.3,
  "lambda_vpiL": 0.005,
  "lambda_smooth": 0.1
}
```

## 数据格式

数据文件为 11 列逗号分隔浮点数，列含义如下：

| 顺序 | 列名 | 作为 |
| --- | --- | --- |
| 1 | `PN_offset` | 输入 |
| 2 | `Bias_V` | 输入 |
| 3 | `Core_width` | 输入 |
| 4 | `P+_width` | 输入 |
| 5 | `N+_width` | 输入 |
| 6 | `P_width` | 输入 |
| 7 | `N_width` | 输入 |
| 8 | `Phase_length` | 输入 |
| 9 | `BW_3dB` | 输出 |
| 10 | `IL` | 输出 |
| 11 | `V_pi` | 输出 |

支持两种文本格式：

```text
a,b,c,...,k
[a, b, c, ..., k]
```

## 安装

```bash
cd /path/to/photonAI
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 训练

```bash
python -m src.main train --config configs/default.yaml
```

训练完成后会在 `results/run_时间戳/` 下生成：

- `config_snapshot.yaml`
- `data_report.md`
- `data_stats.csv`
- `cleaning_meta.json`
- `split_indices.json`
- `x_scaler.pkl`
- `y_scalers.pkl`
- `train_log.csv`
- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `metrics.csv`
- `summary.json`
- `summary.md`
- `test_predictions.csv`
- `figures/*.png`

说明：

- `train_log.csv` 记录每轮的全量 `train_loss` / `test_loss`
- `summary.*` 与 `metrics.csv` 中的 `loss` 为**标准化空间 MSE**
- 物理空间指标仍输出 `MAE / RMSE / R²`

## 评估

按训练时保存的切分索引与 scaler 重算 train/test 指标：

```bash
python -m src.main eval --config configs/default.yaml --run-dir results/run_YYYYMMDD_HHMMSS
```

## 推理

输入文件需包含 8 个输入列（csv 带表头，或 8 列 txt）：

```bash
python -m src.main infer --config configs/default.yaml --input path/to/inputs.csv --output path/to/preds.csv
```

输出列为原始 8 个输入 + `pred_BW_3dB`、`pred_IL`、`pred_V_pi`。

## 默认配置

`configs/default.yaml` 目前对应 notebook 风格的默认 MoE PINN 参数：

- `data.test_size: 0.1`
- `data.random_state: 123`
- `data.filter_v_pi_max: 500.0`
- `model.hidden_dims: [64, 128, 64]`
- `model.n_experts: 60`
- `model.gating_hidden: 8`
- `model.dropout_rate: 0.0`
- `model.use_bn: true`
- `optimizer.lr: 0.001`
- `optimizer.weight_decay: 0.05`
- `training.batch_size: 128`
- `training.epochs: 100`
- `physics.*` 默认由 `best_hyperparams.json` 提供，再由 YAML 显式值覆盖

## 测试

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_smoke.py
```

## 说明

- 现在的主流程优先保证与 notebook 的 **数据切分、标准化、模型结构、物理损失和训练循环** 一致。
- 为了适配仓库化使用，仍保留了 `train / eval / infer` CLI 与 `run_*` 结果目录结构。
- 旧的 MLP baseline 文档与配置已不再是当前默认路径。
├── scripts
│   ├── train.sh
│   ├── eval.sh
│   └── infer.sh
├── src
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── data.py
│   ├── preprocess.py
│   ├── model.py
│   ├── losses.py
│   ├── metrics.py
│   ├── trainer.py
│   ├── evaluate.py
│   ├── infer.py
│   ├── plots.py
│   └── main.py
└── tests
    └── test_smoke.py
```

## 后续可扩展方向

- **Physics-informed loss**：在标准化空间外叠加与器件物理相关的软约束。
- **PINN / 解析近似混合**：将部分输出与简化解析模型对齐。
- **结构搜索**：在 `hidden_dims`、残差块、Bayesian 优化超参等方向扩展。
- **不确定度**：深度集成、MC Dropout、浅层高斯过程等。

## 许可证与引用

若用于论文，请在方法部分说明数据处理、划分方式与随机种子；并引用本仓库或内部项目号（自行补充）。
