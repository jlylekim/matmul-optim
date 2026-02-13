FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
        torch torchvision torchaudio && \
    python3 -m pip install -e .[test,baselines]

ENV PYTHONUNBUFFERED=1
CMD ["bash"]
