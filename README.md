# MZM 器件性能 MLP 回归基线（PyTorch）

本项目实现一个**多输出回归**基线模型：用 **8 个器件/偏置参数**预测 **3 个射频性能指标**。代码面向科研复现：可配置 YAML、固定随机种子、训练集拟合标准化器、完整日志与可视化产物。

## 项目简介

- **任务类型**：监督学习，多输出回归（非分类）。
- **输入（8 维）**：工艺与偏置相关参数。
- **输出（3 维）**：`BW_3dB`、`IL`、`V_pi`。
- **模型**：原生 PyTorch MLP，可选 BatchNorm / Dropout / 残差（同维时相加）。
- **损失**：默认在**标准化后的输出空间**使用加权 `SmoothL1Loss`（Huber）；可选加权 MSE。
- **v1 目标**：先把数据清洗、划分、训练、评估、日志与可视化流程跑通；**不引入 physics loss**。

## 数据格式

数据为 **txt 或 csv**，每行 **11 个逗号分隔的浮点数**，无表头（txt）或表头与下列字段一致（csv）。

| 顺序 | 列名 | 含义 | 作为 |
| --- | --- | --- | --- |
| 1 | `PN_offset` | PN 偏移 | 输入 |
| 2 | `Bias_V` | 偏置电压 | 输入 |
| 3 | `Core_width` | 芯区宽度 | 输入 |
| 4 | `P+_width` | P+ 区宽度 | 输入 |
| 5 | `N+_width` | N+ 区宽度 | 输入 |
| 6 | `P_width` | P 区宽度 | 输入 |
| 7 | `N_width` | N 区宽度 | 输入 |
| 8 | `Phase_length` | 相位区长度 | 输入 |
| 9 | `BW_3dB` | 3 dB 带宽 | 目标 |
| 10 | `IL` | 插入损耗 | 目标 |
| 11 | `V_pi` | 半波电压 | 目标 |

- 自动忽略空行与行首行尾空格。
- 每行必须恰好 **11 列**，否则整文件解析失败并给出错误行号提示。

## TXT 数据清洗流程（以 V_pi 为准）

本仓库约定：**txt 每行从左到右第 11 个逗号分隔浮点数**即半波电压 **`V_pi`**（与表头列名一致）。清洗时以该列为**物理可信区间**的主门控，避免异常仿真/标注污染训练。

建议按以下顺序理解流水线（与 `src/preprocess.py` 中 `clean_dataframe` 实现一致）：

1. **解析与建表**：读取 txt → 校验每行 11 列 → 转为 `float` → 构建 `DataFrame`（最后一列为 `V_pi`）。
2. **（可选）去重**：`remove_duplicate_rows: true` 时删除 11 列完全相同的重复行。
3. **V_pi 区间门控（主清洗）**：默认启用 `filter_v_pi_range: true`，仅保留  
   `v_pi_min <= V_pi <= v_pi_max`（默认 **`[0, 500]`**）。**区间之外整行剔除**。  
   该步骤专门针对「以最后一列 `V_pi` 为正常范围」的需求。
4. **（可选）严格正电压**：`remove_nonpositive_vpi: true` 时，在区间过滤之后再删除 `V_pi <= 0`（若需保留 `V_pi = 0` 且仍在 `[0,500]` 内，请保持为 `false`）。
5. **后续步骤**：train/val/test 划分、（可选）训练集离群策略、仅在训练集上拟合 `StandardScaler` 等，与原先一致。

清洗前会在日志与 `data_report.md` 中报告：给定 `[v_pi_min, v_pi_max]` 下 **`V_pi` 越界行数**、重复样本、同输入异输出等统计，便于核对。

## 环境要求

- Python **3.10+**（已在 3.13 下通过冒烟测试）。
- 推荐使用虚拟环境。

### 安装依赖

```bash
cd /path/to/photonAI
python -m venv .venv
source .venv/bin/activate   # Windows 使用 .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

## 放置数据

1. 将原始 txt（例如仓库根目录下的 `Sim_MZM_dataset.txt`）复制或软链接到 `data/dataset.txt`。
2. 或在 `configs/default.yaml` 中修改 `data_path` 为绝对路径或相对项目根目录的路径。

若路径不存在，程序会给出明确报错，不会静默失败。

## 训练

```bash
python -m src.main train --config configs/default.yaml
```

或使用脚本：

```bash
bash scripts/train.sh
```

训练会在 `results/run_时间戳/` 下生成：

- `config_snapshot.yaml`：本次运行配置快照。
- `split_indices.json`：对**清洗后**样本行的 train/val/test 索引，便于 `eval` 完全复现划分。
- `x_scaler.pkl` / `y_scaler.pkl`：`StandardScaler`，推理阶段用于反标准化。
- `data_report.md` / `data_stats.csv`：数据统计与清洗说明。
- `cleaning_meta.json`：清洗与划分元信息。
- `train_log.csv`：逐 epoch 的 train/val loss 与学习率。
- `checkpoints/best.pt`、`checkpoints/last.pt`：最优与最后一轮权重。
- 训练结束后：`metrics.csv`、`summary.json`、`summary.md`、`test_predictions.csv`、`figures/*.png`。

**说明（损失列）**：`metrics.csv` / `summary.*` 中的 `loss` 与 `*_loss` 均在**标准化输出空间**按训练准则（Huber / 加权 MSE）计算；物理量空间以 **MAE / RMSE / R²** 为主指标。

## 评估（复现划分与 scaler）

在**同一数据文件**与 `config_snapshot.yaml` 前提下，可仅运行评估：

```bash
python -m src.main eval --config configs/default.yaml --run-dir results/run_YYYYMMDD_HHMMSS
```

若不指定 `--run-dir`，将在 `configs/default.yaml` 的 `output_dir`（默认 `results`）下自动选择**最近修改时间**的 `run_*` 目录。

```bash
bash scripts/eval.sh --run-dir results/run_某次训练
```

## 推理

输入文件需包含上述 **8 个输入列**（csv 带表头，或 8 列无表头 txt）。

```bash
python -m src.main infer --config configs/default.yaml --input path/to/inputs.csv --output path/to/preds.csv
```

脚本封装：

```bash
bash scripts/infer.sh path/to/inputs.csv --run-dir results/run_某次训练 --output preds.csv
```

输出列为 8 个输入 + `pred_BW_3dB`、`pred_IL`、`pred_V_pi`（**物理量空间**，已反标准化）。

## 测试

```bash
pip install pytest
pytest -q tests/test_smoke.py
```

## 配置说明（`configs/default.yaml`）

主要字段：

- **数据与清洗**：`data_path`、`remove_duplicate_rows`、**`filter_v_pi_range` / `v_pi_min` / `v_pi_max`**（默认按 **`V_pi ∈ [0, 500]`** 剔除越界行，对应 txt **第 11 列**）、`remove_nonpositive_vpi`、`outlier_strategy`（`none` / `iqr` / `zscore` / `quantile_clip`）及 `outlier_apply_to`（`targets` / `all`）。
- **划分**：`split_ratios`、`random_seed`；先 shuffle 再切分；**仅在训练子集**上拟合标准化器；离群阈值（若启用）也在训练子集上统计。
- **模型**：`hidden_dims`、`batchnorm`、`dropout`、`residual`。
- **训练**：`AdamW`、`lr`、`weight_decay`、`batch_size`、`epochs`、早停 `early_stopping_patience`。
- **调度器**：`cosine`（默认）或 `plateau`。
- **损失**：`huber`（默认）或 `weighted_mse`，`target_weights` 长度须为 3。

默认策略刻意**不删除**仅因统计极端的样本（`outlier_strategy: none`），但在报告中给出极端值计数；**默认以 `V_pi` 物理区间 `[0,500]` 删除越界行**；`remove_nonpositive_vpi` 默认为 `false`，以便与「0 属于合法下界」一致，需要时可改为 `true`。

## 项目结构

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── configs
│   └── default.yaml
├── data
├── reports
├── results
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
