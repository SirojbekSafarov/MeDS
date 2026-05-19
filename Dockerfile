# MeDS — reproducible CUDA 11.8 / PyTorch 2.1.2 image.
#
# Build:   docker build -t meds:latest .
# Run:     docker run --gpus all -it --rm \
#              -v /path/to/datasets:/data \
#              -v /path/to/outputs:/outputs \
#              meds:latest

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# System libs needed by opencv-python, matplotlib, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        ca-certificates \
        git \
        libpq-dev \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Miniconda (Python 3.9 base — matches the reference env)
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py39_23.1.0-1-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /root/miniconda3 && \
    rm -f /tmp/miniconda.sh && \
    conda --version

RUN conda init bash && \
    echo "source activate base" >> ~/.bashrc

# Install Python deps via the pinned-version script.
# We copy only install_packages.sh first so this Docker layer is cached
# even when the source code changes.
COPY install_packages.sh /tmp/install_packages.sh
RUN chmod +x /tmp/install_packages.sh && \
    /tmp/install_packages.sh

# Copy the MeDS source tree.
WORKDIR /workspace/MeDS
COPY . /workspace/MeDS

# Conventional mount points (override with `-v` at `docker run`).
# /data    — host datasets (MVTec-AD, VisA, Real-IAD, noisy variants)
# /outputs — host output directory for memory scores / checkpoints / metrics
VOLUME ["/data", "/outputs"]

CMD ["/bin/bash"]
