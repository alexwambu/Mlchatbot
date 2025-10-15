FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# system dependencies needed by pip wheels and runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements list (will be used after torch install)
COPY requirements.txt /app/requirements.txt

# Install pinned torch CPU wheel first (uses PyTorch CPU index)
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir torch==2.3.1+cpu torchvision==0.18.1+cpu torchaudio==2.3.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining Python deps
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app code
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
