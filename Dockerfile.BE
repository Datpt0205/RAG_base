FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    gcc \
    libpq-dev \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 && \
    python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .

RUN if [ -f ./auth_config_real.yaml ]; then cp ./auth_config_real.yaml ./src/settings/auth_config.yaml; fi
RUN if [ -f ./database_config_real.yaml ]; then cp ./database_config_real.yaml ./src/settings/database_config.yaml; fi

CMD ["bash"]