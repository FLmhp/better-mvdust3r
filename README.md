# Better MV-DUSt3R

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](./requirements.txt)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-76B900?style=flat-square&logo=nvidia&logoColor=white)](./Dockerfile)
[![PyTorch 2.4.1](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](./requirements.txt)
[![Checkpoint best.pth](https://img.shields.io/badge/Checkpoint-best.pth-6f42c1?style=flat-square)](./checkpoints/README.md)
[![License](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey?style=flat-square)](./LICENSE)

<!-- README-I18N:START -->

[English](./README.en.md) | **汉语**

<!-- README-I18N:END -->

> 面向低显存推理、质量自适应重建和 corruption-aware 训练的 MV-DUSt3R 工程优化分支。

`better-mvdust3r` 是基于 [MV-DUSt3R / MV-DUSt3R+](https://arxiv.org/abs/2412.06974) 的工程优化分支，目标是在**不改动模型主体结构与原有训练/推理入口兼容性**的前提下，改善低显存推理、低质量输入稳健性，以及带噪监督下的训练稳定性。

这个仓库保留了原始的训练、评测和 Demo 工作流，同时补上了更适合当前分支的运行时配置、Docker 路径、多语言说明，以及新的稳健训练脚本。

> [!NOTE]
> 当前仓库已经内置 `checkpoints/best.pth`，它是本分支新的训练方式产出的推荐默认权重。

**快速链接：** [论文](https://arxiv.org/abs/2412.06974) · [上游 MV-DUSt3R](https://github.com/facebookresearch/mvdust3r) · [快速启动命令](./command.txt) · [Docker 运行脚本](./run.ps1)

## 这个分支做了什么

| 方向 | 当前优化 |
| --- | --- |
| 低显存推理 | AMP 推理、OOM 自动降 batch、统一 runtime profile |
| 低质量输入稳健性 | 图像质量评分、`lite_robust_v1` 轻量预处理、质量自适应置信度阈值 |
| 大视角场景效率 | `anchorlocal` scene graph，减少完整图带来的 pair 数爆炸 |
| Global Optimization | 最小迭代数 + 检查周期 + 相对改善阈值 + `patience` 早停 |
| Demo / CLI 一致性 | `demo.py` 与 `inference_global_optimization.py` 共用运行时配置思想 |
| 稳健训练 | corruption-aware 数据采样、RGB/几何噪声注入、`geometry_loss_weight` 降权 |
| 复现与部署 | Dockerfile、`run.ps1`、`command.txt`、保留原始训练/测试脚本并新增 robust fine-tune 脚本 |

## 关键改动概览

### 推理期

- `dust3r/runtime_utils.py`：抽离 OOM 检测、batch 候选、图像质量评分、质量自适应阈值和 scene graph 策略。
- `dust3r/inference.py`：支持 `use_amp` 与 `oom_retry`。
- `dust3r/utils/image.py`：加载阶段写入 `raw_quality_score` / `quality_score`，并支持 `preprocess_profile`。
- `dust3r/image_pairs.py`：新增 `anchorlocal`，优先选高质量图作为 anchor。
- `dust3r/cloud_opt/base_opt.py`：为 global alignment 增加早停参数，并记录 `last_alignment_iterations`。
- `demo.py` / `inference_global_optimization.py`：接入统一运行时选项。

### 训练期

- `dust3r/datasets/utils/corruptions.py`：新增 RGB corruption 与 geometry noise 注入逻辑。
- `dust3r/datasets/mvdataset.py`：按 sample 注入 corruption metadata。
- `dust3r/losses.py`：对 geometry-corrupted 样本仅下调依赖 GT geometry 的 loss 权重。
- `scripts/train_mvd_phase2_robust.sh`
- `scripts/train_mvdp_stage2_robust.sh`

## 仓库结构

| 路径 | 说明 |
| --- | --- |
| `demo.py` | Gradio 多图 / 单视频重建 Demo |
| `inference_global_optimization.py` | 带 runtime profile 的 CLI 推理入口 |
| `train.py` | 训练主入口 |
| `dust3r/` | 主要模型、推理、loss、数据集与运行时工具 |
| `scripts/` | 训练、测试与轨迹生成脚本 |
| `checkpoints/` | 权重放置位置，见 `checkpoints/README.md` |
| `data/` | 原始数据放置位置，见 `data/README.md` |
| `trajectories/` | 训练/评测轨迹放置位置，见 `trajectories/README.md` |
| `command.txt` | 基于 `uv` 的 Linux 快速启动命令清单 |
| `run.ps1` | Windows / WSL 下的 Docker Demo 启动脚本 |

## 环境准备

当前依赖配置以 **Python 3.12 + CUDA 12.4 + PyTorch 2.4.1** 为主，与 `Dockerfile` 和 `requirements.txt` 保持一致。

### 方式 1：本地安装（Linux / WSL）

```bash
git clone <your-fork-url>
cd better-mvdust3r
./install.sh
```

`install.sh` 会创建 `mvdp` conda 环境，并安装 `requirements.txt` 中的依赖与 `pytorch3d`。

如果你想快速手动搭环境，也可以逐行参考仓库里的 `command.txt`；其中默认直接使用内置的 `checkpoints/best.pth`。

如需更快的运行时，可额外编译 RoPE CUDA kernel：

```bash
cd croco/models/curope
python setup.py build_ext --inplace
cd ../../..
```

### 方式 2：Docker

```bash
docker build -t mvdust3r:cu124 .
```

该镜像的入口是 `python demo.py`，默认 `CMD` 为 `--help`。

### 方式 3：Windows / WSL PowerShell

```powershell
pwsh -File .\run.ps1
```

`run.ps1` 会以 Docker 方式启动 Demo，并默认使用 `checkpoints/best.pth`。

## 权重与数据

### Checkpoints

当前仓库已经内置了新的训练方式产出的最佳权重：

| 文件 | 说明 |
| --- | --- |
| `best.pth` | 当前分支推荐默认权重，位于 `checkpoints/best.pth` |

如果你还需要对照原始公开权重，可再额外下载到 `checkpoints/`。参考链接：

- <https://huggingface.co/Zhenggang/MV-DUSt3R/tree/main/checkpoints>

常见上游权重如下：

| 文件 | 用途 |
| --- | --- |
| `MVD.pth` | MV-DUSt3R |
| `MVDp_s1.pth` | MV-DUSt3R+ Stage 1 |
| `MVDp_s2.pth` | MV-DUSt3R+ Stage 2 |
| `DUSt3R_ViTLarge_BaseDecoder_224_linear.pth` | DUSt3R 预训练权重 |

### 数据与轨迹

- 数据集说明：`data/README.md`
- 轨迹说明：`trajectories/README.md`
- 轨迹生成脚本：`scripts/tuple_gen/`

当前仓库沿用原项目使用的数据来源：ScanNet、ScanNet++、HM3D、Gibson、MP3D。

## 快速开始

### Gradio Demo

```bash
python demo.py \
  --weights ./checkpoints/best.pth \
  --runtime_profile lite_robust_v1 \
  --server_name 0.0.0.0 \
  --server_port 7860
```

说明：

- 输入支持**多张图片**，或**单个视频文件**（由 loader 均匀抽帧）。
- `lite_robust_v1` 默认启用 AMP、轻量预处理和质量自适应阈值。
- Demo 中的 `confidence threshold` 影响低置信度点过滤比例。
- `No. of video frames` 仅在输入视频时生效。

### CLI 推理（global optimization 路径）

```bash
python inference_global_optimization.py \
  --weights ./checkpoints/best.pth \
  --runtime_profile lite_robust_v1 \
  --scene_graph_policy auto \
  --pair_batch_size 4 \
  --oom_retry \
  image1.jpg image2.jpg image3.jpg image4.jpg
```

推荐至少传入多张图像。该入口会组合以下优化：

- AMP 推理
- OOM 自动回退
- 自动 scene graph 策略
- 质量自适应置信度阈值
- GO 早停

## Runtime Profile

### `default`

更接近原始行为：

- 不启用 AMP
- 不启用 OOM 自动回退
- scene graph 使用 `complete`
- 不做额外预处理
- 不做质量自适应阈值

### `lite_robust_v1`

当前分支推荐默认值：

| 参数 | 默认行为 |
| --- | --- |
| `--amp` | 开启 |
| `--pair_batch_size` | `4` |
| `--oom_retry` | 开启 |
| `--scene_graph_policy` | `auto`，视角较多时转为 `anchorlocal-2` |
| `--preprocess_profile` | `lite_robust_v1` |
| `--quality_adaptive` | 开启 |
| `--go_min_iter` | `40` |
| `--go_max_iter` | `300` |
| `--go_check_every` | `5` |
| `--go_rel_tol` | `5e-4` |
| `--go_patience` | `3` |

不同入口支持的运行时参数并不完全相同：

| 入口 | 已接入参数 |
| --- | --- |
| `demo.py` | `--runtime_profile`、`--amp`、`--preprocess_profile`、`--quality_adaptive`、`--focal_conf_percentile` |
| `inference_global_optimization.py` | 上述参数，以及 `--pair_batch_size`、`--oom_retry`、`--scene_graph_policy`、`--conf_percentile`、`--go_*` |

## 训练与评测

### 原始脚本

| 脚本 | 说明 |
| --- | --- |
| `scripts/train_mvd.sh` | MVD 训练 |
| `scripts/train_mvdp_stage1.sh` | MVDp Stage 1 训练 |
| `scripts/train_mvdp_stage2.sh` | MVDp Stage 2 训练 |
| `scripts/test_mvd.sh` | MVD 测试 |
| `scripts/test_mvdp_stage1.sh` | MVDp Stage 1 测试 |
| `scripts/test_mvdp_stage2.sh` | MVDp Stage 2 测试 |

### 新增稳健训练脚本

| 脚本 | 说明 |
| --- | --- |
| `scripts/train_mvd_phase2_robust.sh` | 从 `outputs/MVD/checkpoint-last.pth` 继续做 robust fine-tune |
| `scripts/train_mvdp_stage2_robust.sh` | 从 `outputs/MPDp_stage2/checkpoint-last.pth` 继续做 robust fine-tune |

robust fine-tune 的默认策略包括：

- `corruption_profile='robust_v1'`
- `corruption_severity='moderate'`
- `corruption_prob=0.35`
- `geom_noise_profile='pose_depth_v1'`
- `geom_noise_prob=0.15`
- `geom_loss_weight=0.3`

附加超参数说明见 `scripts/README.md`，其中已经解释了 `n_all`、`num_views`、`num_render_views`、`random_nv_nr`、`n_ref`、`pts_head_config` 和 `m_ref_flag` 等字段。

## 当前状态与边界

这个分支已经完成了工程层面的运行时与训练稳健化接入，但它的定位仍然是**可继续实验的优化基座**，不是已经做完完整 benchmark 封版的 release。使用时建议自行补充：

- Docker 实际构建验证
- 目标 GPU（尤其 12GB 级别）上的端到端推理验证
- clean / degraded benchmark 对照
- robust fine-tune 收敛验证

## 致谢

- [DUSt3R](https://github.com/naver/dust3r)
- [MV-DUSt3R / MV-DUSt3R+](https://arxiv.org/abs/2412.06974)

## License

项目沿用仓库根目录中的 [LICENSE](LICENSE)。
