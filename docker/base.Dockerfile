#------------------------------------------------------------
# Dockerfile: PyTorch + TensorFlow for RTX 40/50 Series GPUs
# Base: Ubuntu 22.04, CUDA 12.8, Python 3.10
# Strategy: Use nightly builds with CUDA 12.8 for Blackwell (sm_120) support
# Adapted from: https://github.com/dconsorte/pytorch-tensorflow-gpu
#------------------------------------------------------------

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

LABEL maintainer="Dennis Consorte (adapted)"
LABEL description="PyTorch and TensorFlow with GPU support for modern NVIDIA GPUs including Blackwell"

# Create workspace
RUN mkdir /workspace
WORKDIR /workspace

# Install base dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        gpg-agent \
        ca-certificates \
        wget \
        curl \
        git \
        lsb-release \
        build-essential \
        pkg-config \
        libopenblas-dev \
        libjpeg-dev \
        libpng-dev \
        libhdf5-dev \
        libgl1-mesa-glx \
        libgl1-mesa-dri \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install pip for system Python 3.10
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip python3-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Set CUDA environment
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV CUDA_CACHE_DISABLE=0

# Install PyTorch NIGHTLY with CUDA 12.8 for Blackwell support
RUN pip3 install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128/

# Install TensorFlow nightly
RUN pip3 install tf-nightly

# Install additional ML packages
RUN pip3 install \
    "numpy<2" \
    scipy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    scikit-image \
    opencv-python \
    tqdm \
    pillow \
    h5py \
    pyyaml \
    tensorboard

# Copy test utilities from original repo
COPY test_gpu.py /workspace/test_gpu.py

# Test GPU support during build
RUN python3 /workspace/test_gpu.py

CMD ["/bin/bash"]