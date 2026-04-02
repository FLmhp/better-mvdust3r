FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG UV_VERSION=0.11.3
ARG PYTORCH3D_TAG=V0.7.8

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Etc/UTC \
    DEBIAN_FRONTEND=noninteractive \
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

# 安装系统依赖
RUN sed -i 's@http://archive.ubuntu.com/ubuntu/@https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' /etc/apt/sources.list \
 && sed -i 's@http://security.ubuntu.com/ubuntu/@https://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' /etc/apt/sources.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
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

# 安装 UV
RUN curl --proto '=https' --tlsv1.2 -LsSf \
    https://releases.astral.sh/github/uv/releases/download/${UV_VERSION}/uv-installer.sh | sh

# 复制源码
WORKDIR /workspace/mvdust3r
COPY requirements.txt /workspace/mvdust3r

# 创建 Python 3.12 虚拟环境
RUN uv venv ${VIRTUAL_ENV} --python 3.12
# 安装 Python 依赖
RUN uv pip install \
    --python ${VIRTUAL_ENV}/bin/python \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    -r requirements.txt

# 构建并安装 pytorch3d
RUN uv pip install \
    --python ${VIRTUAL_ENV}/bin/python \
    --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@${PYTORCH3D_TAG}"

COPY . /workspace/mvdust3r

# compile RoPE CUDA kernels for faster runtime
RUN cd croco/models/curope \
 && python setup.py build_ext --inplace

EXPOSE 7860
ENTRYPOINT ["python", "demo.py"]
CMD ["--help"]