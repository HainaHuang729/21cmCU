FROM nvidia/cuda:12.6.2-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_ARCHS="80,86,89,90"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3-pip \
    python3-venv \
    gcc \
    g++ \
    libgsl-dev \
    libfftw3-dev \
    pkg-config \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/21cmcu
COPY . /opt/21cmcu

RUN python3.12 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && USE_CUDA=1 CUDA_ARCHS="${CUDA_ARCHS}" /opt/venv/bin/pip install .

ENV PATH="/opt/venv/bin:${PATH}" \
    P21_USE_CUDA=1 \
    P21_CUDA_MODE=fast \
    OPENBLAS_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1

CMD ["21cmcu", "doctor"]
