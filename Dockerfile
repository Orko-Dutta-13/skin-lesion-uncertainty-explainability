# Reproducible environment for the whole project. CPU image by default (small, runs anywhere):
#
#   docker build -t reliable-derm .
#   docker run --rm reliable-derm                      # whole pipeline on fake data (`make smoke-all`), a few minutes
#   docker run --rm reliable-derm make test            # unit tests
#
# GPU (NVIDIA Container Toolkit / Docker Desktop with WSL2) for real training:
#   docker build -t reliable-derm-gpu --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 .
#   docker run --rm --gpus all -v "%cd%\data:/app/data" -v "%cd%\runs:/app/runs" -v "%cd%\reports:/app/reports" reliable-derm-gpu make train
# The ISIC images are NOT baked into the image (licence + size); mount them as shown above.
FROM python:3.11-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends make && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# 1) torch/torchvision from the chosen index first, so requirements.txt does not pull the huge default CUDA wheels
RUN pip install torch torchvision --index-url ${TORCH_INDEX}
# 2) everything else (cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install -r requirements.txt
# 3) the project
COPY . .

CMD ["make", "smoke-all"]
