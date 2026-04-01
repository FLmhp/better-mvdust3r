# MV-DUSt3R 优化改动详细 Changelog

文档日期：2026-04-01  
文档范围：本轮在当前工作区中已经完成的 Phase 1 与 Phase 2 代码改动  
文档目的：记录实际落地的功能、接口、行为变化、验证结果与未完成事项，作为后续训练、测试、交接和复盘依据

## 1. 总览

本轮改动围绕两个目标展开：

1. 在不显著牺牲精度的前提下，降低推理显存占用与无效算力消耗，形成可在单卡 12GB 级别显存上运行的 `lite_robust_v1` 运行时路径。
2. 在输入图像质量较差或几何监督存在噪声时，提升训练与推理链路的稳健性，形成 corruption-aware 的 Phase 2 微调基础设施。

本轮改动中，真正已经实现的是：

- 容器化构建路径重构
- 推理运行时低显存优化
- 输入图像轻量预处理与质量自适应阈值
- 自适应 scene graph 策略
- global optimization 早停
- Gradio / CLI 推理入口统一接入新运行时选项
- corruption-aware 数据集采样与样本元信息注入
- 几何噪声样本的 loss 下调策略
- robust fine-tune 训练脚本

本轮没有实现的内容：

- 更小 student 模型结构
- teacher-student 蒸馏训练
- Docker 镜像在目标 GPU 环境上的实际构建验证
- Phase 2 的真实训练收敛验证与 benchmark 回归

## 2. 文件级改动总表

### 2.1 新增文件

- `Dockerfile`
- `.dockerignore`
- `dust3r/runtime_utils.py`
- `dust3r/datasets/utils/corruptions.py`
- `scripts/train_mvd_phase2_robust.sh`
- `scripts/train_mvdp_stage2_robust.sh`

### 2.2 修改文件

- `demo.py`
- `dust3r/cloud_opt/base_opt.py`
- `dust3r/datasets/mvdataset.py`
- `dust3r/image_pairs.py`
- `dust3r/inference.py`
- `dust3r/losses.py`
- `dust3r/utils/image.py`
- `inference_global_optimization.py`

### 2.3 无单独删除文件

没有引入新的物理删除文件操作。  
但 `inference_global_optimization.py` 的旧逻辑已经被整体重构，功能上等价于“保留文件名、替换主要实现”。

## 3. Phase 1 详细改动

## 3.1 Docker 与构建链路

### 新增 `Dockerfile`

关键变化：

- 以 `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` 为基础镜像。
- 统一使用 Python 3.12 虚拟环境。
- 预装 `torch==2.4.1` 与 `torchvision==0.19.1` 的 CUDA 12.4 版本。
- 从 `requirements.txt` 生成 `requirements.docker.txt`，剔除不匹配的 `torch/torchvision` pin。
- 通过源码构建 `pytorch3d==0.7.8`。
- 编译 `croco/models/curope` 下的 RoPE CUDA kernel。
- 将容器入口改为 `CMD ["python", "demo.py", "--help"]`，避免容器启动即阻塞在 Gradio。

行为层面的意义：

- 镜像构建以当前工作树为准，而非重新抓取固定 upstream commit。
- 容器更适合作为“当前分支代码的可复现实验镜像”，而不是“某个历史版本的只读镜像”。
- 依赖版本被压实，减少了 PyTorch / PyTorch3D / CUDA 组合失配风险。

### 新增 `.dockerignore`

排除内容包括：

- `.git`
- `outputs/`
- `data/`
- `trajectories/`
- `checkpoints/`
- 各类 cache / pyc

行为层面的意义：

- 避免 build context 被大体积数据、训练产物和版本库元数据污染。
- 强制采用“镜像只带代码，数据和权重运行时挂载”的工作模式。

## 3.2 运行时公共工具抽离

### 新增 `dust3r/runtime_utils.py`

新增能力：

- OOM 错误识别
- 自动 batch size 候选生成
- 基于百分位的置信度阈值计算
- 置信度掩码生成
- 图像质量评分
- 质量自适应阈值调整
- `lite_robust_v1` 预处理
- scene graph policy 自动解析
- 场景级最小保留点数估计

设计意义：

- 将“低显存运行策略”和“低质量输入鲁棒性策略”从业务入口代码中抽离。
- 避免 `demo.py`、`inference_global_optimization.py`、`losses.py` 分别复制同一套阈值逻辑。

## 3.3 图像加载与轻量预处理

### 修改 `dust3r/utils/image.py`

新增行为：

- `load_images()` 支持 `preprocess_profile` 参数。
- 每张图在加载时计算 `raw_quality_score`。
- 根据 profile 执行轻量预处理后，再记录 `quality_score`。

当前 `lite_robust_v1` 预处理内容：

- 轻量去噪
- CLAHE 风格的亮度归一化
- 极低质量图像上的轻微锐化

行为层面的影响：

- 推理链路对低曝光、低对比度、轻度噪声输入更稳。
- 预处理只在图像加载时发生，不侵入模型本体。

## 3.4 推理路径低显存化

### 修改 `dust3r/inference.py`

新增能力：

- `loss_of_one_batch_mv()`、`loss_of_one_batch()` 支持 `use_amp`
- `inference_mv()` 支持 `use_amp`
- `inference()` 支持 `use_amp`
- `inference()` 支持 `oom_retry`
- 当输入 shape 可兼容时，自动尝试 `batch_size` 递减回退

运行时行为变化：

- 在允许 AMP 时，可使用 autocast 减少显存占用。
- 在开启 `oom_retry` 时，pairwise 推理会自动尝试更小 micro-batch，而不是直接失败。

已知边界：

- 当前回退策略是保守式递减，不是动态 profile 学习。
- 如果样本 shape 不一致，则不会强行拼 batch 做更复杂回退。

## 3.5 场景图构建优化

### 修改 `dust3r/image_pairs.py`

新增 `anchorlocal` scene graph：

- 自动选择高质量图像作为 anchor，或显式指定 anchor
- anchor 与所有视角连边
- 同时保留局部邻域边

该策略主要针对：

- 视角数较多时，完整图带来的 pair 数爆炸
- 低质量输入中，局部邻接仍然重要，但全图全连边代价过高

## 3.6 Global Optimization 早停

### 修改 `dust3r/cloud_opt/base_opt.py`

为 `global_alignment_loop()` 增加如下参数：

- `min_iter`
- `check_every`
- `rel_tol`
- `patience`

同时新增：

- `net.last_alignment_iterations`

行为变化：

- GO 不再一律跑满固定迭代数。
- 当达到最小迭代次数后，若若干次检查点之间相对改善不足，则提前停止。

收益预期：

- 在易收敛场景中减少无效迭代。
- 将运行时优化从“只减前向显存”扩展到“也减后端优化时间”。

## 3.7 `inference_global_optimization.py` 全面重构

### 主要变化

- 抽象出默认 runtime option
- 支持命令行配置 runtime profile
- 接入预处理 profile
- 接入质量自适应阈值
- 接入新的 scene graph policy
- 接入 GO 早停参数
- 清理旧版坏掉的调用残留
- 重写为更接近“单文件推理工具”的结构

实际效果：

- 该文件不再只是旧逻辑堆叠，而成为一个可独立承载 batch / GO / 导出职责的统一入口。

## 3.8 Gradio Demo 接入新运行时

### 修改 `demo.py`

新增 CLI / 运行时参数：

- `--runtime_profile`
- `--amp`
- `--preprocess_profile`
- `--quality_adaptive`
- `--focal_conf_percentile`

功能变化：

- Gradio 推理链路接入 `lite_robust_v1`
- 质量较差图像可自动放宽置信度丢弃比例
- 焦距估计前的点选择不再是固定阈值
- 移除阻塞式 `input()`，避免 demo 入口卡死
- `main_demo()` 正确使用传入的 `server_name/server_port`

对使用者的影响：

- 现在 Gradio 侧的行为与 CLI 推理逻辑更加一致。
- demo 更适合直接放在容器内或远程机器上运行。

## 4. Phase 2 详细改动

## 4.1 corruption helper 新增

### 新增 `dust3r/datasets/utils/corruptions.py`

新增两类扰动：

### RGB corruption

- Gaussian blur
- Motion blur
- Gaussian noise
- JPEG compression
- Downsample-Upsample
- Exposure shift
- Contrast shift
- Random occlusion

### Geometry noise

- Depth dropout
- Depth quantization
- Focal perturbation
- Pose perturbation

并实现：

- `build_corruption_state()`：按样本决定 clean / rgb / geometry 三种模式
- `apply_rgb_corruption()`：执行 RGB 退化
- `apply_geometry_corruption()`：执行几何噪声

设计特点：

- 扰动决定是“按样本轨迹统一采样”，不是每张图各自独立采样。
- 这保证了多视图任务不会因为视角间随机退化完全失配而变成无意义训练。

## 4.2 数据集接入 corruption-aware 训练

### 修改 `dust3r/datasets/mvdataset.py`

新增数据集参数：

- `corruption_profile`
- `corruption_severity`
- `corruption_prob`
- `geom_noise_profile`
- `geom_noise_prob`
- `geom_loss_weight`

采样期新增行为：

- 每个 datapoint 在进入 view 构造前先确定 corruption 模式
- 若为 RGB corruption，则在 crop / resize 后执行图像退化
- 若为 geometry corruption，则对 `depth`、`intrinsics`、`camera_pose` 注入噪声
- 为每个 view 写入训练元信息

新增元信息字段包括：

- `corruption_mode`
- `corruption_severity`
- `corruption_profile`
- `geom_noise_profile`
- `rgb_corrupted`
- `geom_corrupted`
- `rgb_corruption_name`
- `geom_noise_name`
- `geometry_loss_weight`

直接效果：

- 后续 loss 层不需要知道数据增强细节，只依赖 metadata 即可。

## 4.3 几何损失的样本级权重控制

### 修改 `dust3r/losses.py`

新增工具函数：

- `_view_sample_weights`
- `_apply_sample_weights`
- `_masked_sample_weights`
- `_masked_weighted_criterion`

### 作用于 `Regr3D`

`Regr3D` 在单视图与多视图路径上都接入了 `geometry_loss_weight`：

- clean / 仅 RGB corruption 样本：权重默认 1.0
- geometry-corrupted 样本：权重可按数据集配置下调，当前默认 0.3

这意味着：

- GT geometry 已被主动加噪时，不再让几何回归 loss 以原权重强行拟合被污染的 supervision。

## 4.4 `GSRenderLoss` 的稳健化

### 关键调整

- `local_loss()` 支持按 sample 返回结果
- `local_lap_loss()` 支持按 view 返回结果
- 在 `compute_loss_mv()` 中，对依赖 GT 几何的 local / lap 项施加 `geometry_loss_weight`
- RGB / LPIPS / render consistency 主链保持原权重

这一设计刻意遵循的原则是：

- noisy geometry 只应削弱对“几何真值”的强制拟合
- 不应同步削弱由多视图一致性与重建渲染带来的学习信号

## 4.5 置信度阈值逻辑的稳健化

### 在 `dust3r/losses.py` 中的调整

将原本固定 3% cutoff 的若干位置替换为：

- `confidence_keep_mask()`
- `min_keep_points_from_hw()`

相对于原实现的变化：

- 不再单纯依赖固定百分位
- 增加最小保留点下界，避免在极差样本上把有效点全部裁掉

## 4.6 新增 robust fine-tune 脚本

### 新增 `scripts/train_mvd_phase2_robust.sh`

特点：

- 从 `outputs/MVD/checkpoint-last.pth` 继续训练
- 默认训练 20 epoch
- 使用 `corruption_profile='robust_v1'`
- 使用 `corruption_prob=0.35`
- 使用 `geom_noise_profile='pose_depth_v1'`
- 使用 `geom_noise_prob=0.15`
- 使用 `geom_loss_weight=0.3`

### 新增 `scripts/train_mvdp_stage2_robust.sh`

特点：

- 从 `outputs/MPDp_stage2/checkpoint-last.pth` 继续训练
- 其余 corruption-aware 参数与上面保持一致

设计上的考虑：

- 不覆盖原有训练脚本
- 原 clean 训练流与 robust fine-tune 流解耦
- 便于做 A/B 对照和回归

## 5. 行为变化总结

## 5.1 用户可见变化

- Docker 镜像默认不再直接运行 demo，而是先输出帮助
- Gradio / CLI 现在有更清晰的 runtime profile 行为
- 单张视频文件也能作为 demo 输入，由 loader 抽帧处理
- 低质量图像的预处理与阈值策略会自动影响推理输出

## 5.2 训练行为变化

- 数据集现在可以在不改模型结构的前提下混合 clean / RGB corruption / geometry corruption
- loss 对 geometry-corrupted 样本更宽容
- 训练脚本现在支持从 clean 模型继续进行 robust fine-tune

## 5.3 推理资源行为变化

- pairwise inference 支持 AMP
- batch 可在 OOM 时自动递减
- GO 支持早停
- 多视角 scene graph 在大视角数条件下可避免完整图爆炸

## 6. 验证记录

本轮已完成的验证：

- 对修改后的 Python 文件执行了 `python -m py_compile`
- 编译检查通过，说明语法层面无错误

本轮未完成的验证：

- Docker 镜像实际构建
- 单卡 12GB GPU 真实推理验证
- Phase 2 训练收敛验证
- clean benchmark 与 corruption benchmark 定量对照

额外说明：

- 曾尝试做本地 Python smoke test，但当前 shell 环境中的 `python` 缺失 `numpy`，因此无法作为训练运行时验证环境。

## 7. 已知风险与技术债

- `inference_global_optimization.py` 虽已重构，但仍需真实样本做端到端 smoke test。
- corruption 设计目前是启发式实现，尚未经过 benchmark 网格搜索。
- `geom_loss_weight=0.3` 是工程先验，不是实验最优值。
- robust fine-tune 脚本默认依赖指定输出目录下已有 checkpoint，若目录命名不一致需手工调整。
- 当前并未实现 student 模型或蒸馏，Phase 2 仍然依赖原模型规模。

## 8. 结论

这轮改动已经把“低显存推理优化”和“低质量输入稳健化训练基础设施”两条线打通，但仍属于“工程可运行基础版”，不属于“实验闭环已完成版”。

如果后续进入正式 benchmark 阶段，建议优先执行：

1. Docker 构建与单卡 12GB 推理验证
2. clean / degraded benchmark 回归
3. `corruption_prob`、`geom_noise_prob`、`geom_loss_weight` 三个超参数网格搜索
4. 是否有必要引入 student/distillation 路径的再评估
