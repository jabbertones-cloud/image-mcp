# ImageTools remote-generation server.
#
# Build:
#   docker build -t imagetools-remotegen:0.1 -f Dockerfile .
#
# Run (local smoke):
#   docker run --rm --gpus all -p 8443:8443 \
#     -v /path/to/server-secrets:/etc/remote-gen:ro \
#     -v /path/to/qwen-weights:/var/remote-gen/models:ro \
#     --tmpfs /run/remote-gen/transfer:size=4g,mode=1700 \
#     -e REMOTE_MODEL=qwen-image \
#     -e REMOTE_HOST=0.0.0.0 -e REMOTE_PORT=8443 \
#     imagetools-remotegen:0.1
#
# Pin notes:
#   * CUDA 12.4 + torch 2.6.0+cu124 — same combo verified for Qwen training in
#     QwenCharLoRA. Avoids the torch 2.11+cu128 fp8 ACCESS_VIOLATION on Ada.
#   * Diffusers 0.38+ is required for QwenImagePipeline + QwenImageEditPipeline.
#   * Transformers pinned <5.0 — 5.x changed Qwen2-VL state-dict shape and the
#     server's loader can't read those weights anyway (per local notes).

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── system deps + tailscale repo ────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl gnupg \
        git tini iptables iproute2 jq && \
    add-apt-repository ppa:deadsnakes/ppa && \
    curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/jammy.noarmor.gpg \
        | tee /usr/share/keyrings/tailscale-archive-keyring.gpg >/dev/null && \
    curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/jammy.tailscale-keyring.list \
        | tee /etc/apt/sources.list.d/tailscale.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        tailscale && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python && \
    python -m ensurepip --upgrade && \
    python -m pip install --upgrade pip wheel

# ── torch (cu124) + diffusers + transformers ────────────────────────
RUN python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0 torchvision==0.21.0 && \
    python -m pip install \
        diffusers==0.38.0 \
        "transformers>=4.45.0,<5.0" \
        accelerate \
        "safetensors>=0.4.0,<0.8" \
        bitsandbytes==0.49.2 \
        huggingface_hub[hf_xet] \
        Pillow

# ── server package + runtime deps ───────────────────────────────────
RUN python -m pip install fastapi 'uvicorn[standard]' httpx PyNaCl python-multipart cryptography

WORKDIR /opt/remote-server
COPY remote_server /opt/remote-server/remote_server
COPY scripts/entrypoint.sh /usr/local/bin/remote-gen-entrypoint
RUN chmod +x /usr/local/bin/remote-gen-entrypoint

# ── runtime dirs (paths the settings.py defaults expect) ───────────
RUN mkdir -p /etc/remote-gen /var/remote-gen/models /run/remote-gen/transfer

ENV REMOTE_SECRETS_DIR=/etc/remote-gen \
    REMOTE_MODELS_ROOT=/var/remote-gen/models \
    REMOTE_TRANSFER_DIR=/run/remote-gen/transfer \
    REMOTE_HOST=0.0.0.0 \
    REMOTE_PORT=8443 \
    REMOTE_MODEL=qwen-image

EXPOSE 8443

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/remote-gen-entrypoint"]
CMD []
