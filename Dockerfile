FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG UV_VERSION=0.11.2
ARG MVDUST3R_COMMIT=18872ea
ARG PYTORCH3D_TAG=v0.7.8

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Etc/UTC \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    CMAKE_PREFIX_PATH=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/root/.local/bin:${PATH} \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    ninja-build \
    pkg-config \
 && rm -rf /var/lib/apt/lists/*

# pin uv
RUN curl --proto '=https' --tlsv1.2 -LsSf \
    https://releases.astral.sh/github/uv/releases/download/${UV_VERSION}/uv-installer.sh | sh

WORKDIR /workspace/mvdust3r

# pin mvdust3r source
COPY . /workspace/mvdust3r

# uv-managed Python 3.12 virtualenv
RUN uv venv ${VIRTUAL_ENV} --python 3.12

# install CUDA 12.4 PyTorch first
RUN uv pip install \
    --python ${VIRTUAL_ENV}/bin/python \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    torch==2.4.1 \
    torchvision==0.19.1

# reuse upstream requirements, but drop the mismatched torch/torchvision pins
RUN python - <<'PY'
from pathlib import Path

items = Path("requirements.txt").read_text(encoding="utf-8").split()
drop_prefixes = ("torch==", "torchvision==")
keep = [x for x in items if not x.startswith(drop_prefixes)]

# pytorch3d runtime dependency
keep.append("iopath")

Path("requirements.docker.txt").write_text(
    "\n".join(keep) + "\n",
    encoding="utf-8",
)
PY

RUN uv pip install --python ${VIRTUAL_ENV}/bin/python -r requirements.docker.txt

# build pytorch3d from source against the already-installed torch
RUN uv pip install \
    --python ${VIRTUAL_ENV}/bin/python \
    --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"

# compile RoPE CUDA kernels for faster runtime
RUN cd croco/models/curope \
 && python setup.py build_ext --inplace

EXPOSE 7860

CMD ["python", "demo.py", "--help"]