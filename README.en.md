# Better MV-DUSt3R

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](./requirements.txt)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-76B900?style=flat-square&logo=nvidia&logoColor=white)](./Dockerfile)
[![PyTorch 2.4.1](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](./requirements.txt)
[![Checkpoint best.pth](https://img.shields.io/badge/Checkpoint-best.pth-6f42c1?style=flat-square)](./checkpoints/README.md)
[![License](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey?style=flat-square)](./LICENSE)

<!-- README-I18N:START -->

**English** | [汉语](./README.md)

<!-- README-I18N:END -->

> An engineering-focused MV-DUSt3R fork for lower-memory inference, quality-adaptive reconstruction, and corruption-aware training.

`better-mvdust3r` is an optimized fork of [MV-DUSt3R / MV-DUSt3R+](https://arxiv.org/abs/2412.06974). It aims to improve low-memory inference, robustness to low-quality inputs, and training stability under noisy supervision **without changing the core model structure or breaking the original training and inference entry points**.

This repository keeps the original training, evaluation, and demo workflows while adding runtime profiles, Docker-friendly paths, multilingual documentation, and new robust fine-tuning scripts.

> [!NOTE]
> The repository already includes `checkpoints/best.pth`, which is the recommended default checkpoint produced by the new training pipeline in this fork.

**Quick links:** [Paper](https://arxiv.org/abs/2412.06974) · [Upstream MV-DUSt3R](https://github.com/facebookresearch/mvdust3r) · [Bootstrap commands](./command.txt) · [Docker run script](./run.ps1)

## What this fork improves

| Area | Current improvement |
| --- | --- |
| Low-memory inference | AMP inference, OOM batch fallback, unified runtime profiles |
| Robustness to low-quality inputs | Image quality scoring, `lite_robust_v1` preprocessing, quality-adaptive confidence thresholds |
| Efficiency for many views | `anchorlocal` scene graph to avoid pair explosion from complete graphs |
| Global Optimization | Early stopping with minimum iterations, check intervals, relative tolerance, and `patience` |
| Demo / CLI consistency | `demo.py` and `inference_global_optimization.py` share the same runtime-profile philosophy |
| Robust training | corruption-aware sampling, RGB / geometry corruption injection, `geometry_loss_weight` reweighting |
| Reproducibility and deployment | Dockerfile, `run.ps1`, `command.txt`, original train/test scripts, and added robust fine-tune scripts |

## Key implementation changes

### Inference path

- `dust3r/runtime_utils.py`: shared OOM detection, batch-size fallback candidates, image quality scoring, quality-adaptive thresholds, and scene-graph policy helpers.
- `dust3r/inference.py`: support for `use_amp` and `oom_retry`.
- `dust3r/utils/image.py`: records `raw_quality_score` / `quality_score` during loading and supports `preprocess_profile`.
- `dust3r/image_pairs.py`: adds `anchorlocal` and prefers a high-quality image as the anchor.
- `dust3r/cloud_opt/base_opt.py`: adds early-stopping controls for global alignment and stores `last_alignment_iterations`.
- `demo.py` / `inference_global_optimization.py`: wired into the new runtime options.

### Training path

- `dust3r/datasets/utils/corruptions.py`: RGB corruption and geometry-noise injection.
- `dust3r/datasets/mvdataset.py`: injects corruption metadata per sample.
- `dust3r/losses.py`: downweights only GT-geometry-dependent losses for geometry-corrupted samples.
- `scripts/train_mvd_phase2_robust.sh`
- `scripts/train_mvdp_stage2_robust.sh`

## Repository layout

| Path | Description |
| --- | --- |
| `demo.py` | Gradio demo for multi-image or single-video reconstruction |
| `inference_global_optimization.py` | CLI inference entry point with runtime profiles |
| `train.py` | Main training entry point |
| `dust3r/` | Core model, inference, loss, dataset, and runtime utilities |
| `scripts/` | Training, testing, and trajectory-generation scripts |
| `checkpoints/` | Checkpoint directory, see `checkpoints/README.md` |
| `data/` | Raw dataset directory, see `data/README.md` |
| `trajectories/` | Training / evaluation trajectory directory, see `trajectories/README.md` |
| `command.txt` | Linux bootstrap command list based on `uv` |
| `run.ps1` | Docker demo launcher for Windows / WSL |

## Environment setup

The current dependency stack is centered on **Python 3.12 + CUDA 12.4 + PyTorch 2.4.1**, matching `Dockerfile` and `requirements.txt`.

### Option 1: local setup (Linux / WSL)

```bash
git clone <your-fork-url>
cd better-mvdust3r
./install.sh
```

`install.sh` creates the `mvdp` conda environment and installs the dependencies from `requirements.txt` together with `pytorch3d`.

If you prefer a more manual bootstrap flow, you can also follow `command.txt` line by line; it already defaults to the bundled `checkpoints/best.pth`.

For faster runtime, you can optionally compile the RoPE CUDA kernel:

```bash
cd croco/models/curope
python setup.py build_ext --inplace
cd ../../..
```

### Option 2: Docker

```bash
docker build -t mvdust3r:cu124 .
```

The image entrypoint is `python demo.py`, and the default `CMD` is `--help`.

### Option 3: Windows / WSL PowerShell

```powershell
pwsh -File .\run.ps1
```

`run.ps1` starts the Docker demo and defaults to `checkpoints/best.pth`.

## Checkpoints and data

### Checkpoints

The repository already ships with the best checkpoint produced by the new training method:

| File | Description |
| --- | --- |
| `best.pth` | Recommended default checkpoint for this fork, located at `checkpoints/best.pth` |

If you still want the original public weights for comparison, you can download them separately into `checkpoints/`:

- <https://huggingface.co/Zhenggang/MV-DUSt3R/tree/main/checkpoints>

Common upstream checkpoints:

| File | Purpose |
| --- | --- |
| `MVD.pth` | MV-DUSt3R |
| `MVDp_s1.pth` | MV-DUSt3R+ Stage 1 |
| `MVDp_s2.pth` | MV-DUSt3R+ Stage 2 |
| `DUSt3R_ViTLarge_BaseDecoder_224_linear.pth` | DUSt3R pretraining checkpoint |

### Data and trajectories

- Dataset instructions: `data/README.md`
- Trajectory instructions: `trajectories/README.md`
- Trajectory-generation scripts: `scripts/tuple_gen/`

This fork keeps the original data sources used by the upstream project: ScanNet, ScanNet++, HM3D, Gibson, and MP3D.

## Quick start

### Gradio demo

```bash
python demo.py \
  --weights ./checkpoints/best.pth \
  --runtime_profile lite_robust_v1 \
  --server_name 0.0.0.0 \
  --server_port 7860
```

Notes:

- Input can be **multiple images** or a **single video file** sampled uniformly by the loader.
- `lite_robust_v1` enables AMP, lightweight preprocessing, and quality-adaptive thresholds by default.
- The `confidence threshold` in the UI controls low-confidence point filtering.
- `No. of video frames` only applies to video input.

### CLI inference (global optimization path)

```bash
python inference_global_optimization.py \
  --weights ./checkpoints/best.pth \
  --runtime_profile lite_robust_v1 \
  --scene_graph_policy auto \
  --pair_batch_size 4 \
  --oom_retry \
  image1.jpg image2.jpg image3.jpg image4.jpg
```

Using multiple images is recommended. This entry point combines:

- AMP inference
- OOM batch fallback
- automatic scene-graph policy
- quality-adaptive confidence thresholds
- GO early stopping

## Runtime profiles

### `default`

Closer to the original behavior:

- no AMP
- no OOM fallback
- `complete` scene graph
- no extra preprocessing
- no quality-adaptive thresholding

### `lite_robust_v1`

Recommended defaults in this fork:

| Argument | Default behavior |
| --- | --- |
| `--amp` | enabled |
| `--pair_batch_size` | `4` |
| `--oom_retry` | enabled |
| `--scene_graph_policy` | `auto`, switching to `anchorlocal-2` for larger view counts |
| `--preprocess_profile` | `lite_robust_v1` |
| `--quality_adaptive` | enabled |
| `--go_min_iter` | `40` |
| `--go_max_iter` | `300` |
| `--go_check_every` | `5` |
| `--go_rel_tol` | `5e-4` |
| `--go_patience` | `3` |

The supported runtime flags are not identical across entry points:

| Entry point | Wired runtime options |
| --- | --- |
| `demo.py` | `--runtime_profile`, `--amp`, `--preprocess_profile`, `--quality_adaptive`, `--focal_conf_percentile` |
| `inference_global_optimization.py` | everything above plus `--pair_batch_size`, `--oom_retry`, `--scene_graph_policy`, `--conf_percentile`, and `--go_*` |

## Training and evaluation

### Original scripts

| Script | Description |
| --- | --- |
| `scripts/train_mvd.sh` | MVD training |
| `scripts/train_mvdp_stage1.sh` | MVDp Stage 1 training |
| `scripts/train_mvdp_stage2.sh` | MVDp Stage 2 training |
| `scripts/test_mvd.sh` | MVD evaluation |
| `scripts/test_mvdp_stage1.sh` | MVDp Stage 1 evaluation |
| `scripts/test_mvdp_stage2.sh` | MVDp Stage 2 evaluation |

### Added robust fine-tuning scripts

| Script | Description |
| --- | --- |
| `scripts/train_mvd_phase2_robust.sh` | Continues robust fine-tuning from `outputs/MVD/checkpoint-last.pth` |
| `scripts/train_mvdp_stage2_robust.sh` | Continues robust fine-tuning from `outputs/MPDp_stage2/checkpoint-last.pth` |

Default robust fine-tuning strategy:

- `corruption_profile='robust_v1'`
- `corruption_severity='moderate'`
- `corruption_prob=0.35`
- `geom_noise_profile='pose_depth_v1'`
- `geom_noise_prob=0.15`
- `geom_loss_weight=0.3`

Additional hyperparameter notes are documented in `scripts/README.md`, including `n_all`, `num_views`, `num_render_views`, `random_nv_nr`, `n_ref`, `pts_head_config`, and `m_ref_flag`.

## Current status and scope

This fork already wires in the engineering-side runtime and training robustness improvements, but it should still be treated as **an optimization base for continued experiments**, not as a fully benchmark-closed release. You should still validate:

- actual Docker builds
- end-to-end inference on the target GPU, especially 12 GB-class cards
- clean vs. degraded benchmark comparisons
- robust fine-tune convergence

## Acknowledgements

- [DUSt3R](https://github.com/naver/dust3r)
- [MV-DUSt3R / MV-DUSt3R+](https://arxiv.org/abs/2412.06974)

## License

This project follows the repository root [LICENSE](LICENSE).
