# Stage 1: Build the Rust topological verifier CLI
FROM --platform=linux/amd64 rust:1.83-slim as builder
WORKDIR /build
COPY rust /build/rust
WORKDIR /build/rust
RUN cargo build --release --bin bioknot_cli

# Stage 2: Python runtime image
FROM --platform=linux/amd64 python:3.10-slim

# Copy compiled Rust CLI binary from builder
COPY --from=builder /build/rust/target/release/bioknot_cli /usr/local/bin/bioknot_cli
RUN chmod +x /usr/local/bin/bioknot_cli

# Hugging Face Spaces runs containers as UID 1000; matching it keeps the HF cache
# and the trained checkpoint writable at runtime.
RUN useradd -m -u 1000 user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create /app owned by `user` rather than letting WORKDIR create it as root.
# COPY --chown fixes the ownership of copied contents but not of the directory
# itself, and the training step below writes a NEW file (bioknot_native.pt) into
# it, which needs write permission on the directory.
RUN install -d -o user -g user /app
WORKDIR /app

COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . /app

USER user

# Preload the ESM-2 backbone weights into the container image layer so cold starts
# do not download them. Pre-trained classifier weights (bioknot_native.pt) are copied
# directly, avoiding slow boot-time CPU training and timeout risks.
RUN python training/preload.py

EXPOSE 7860

# --timeout 300 sits above the server's own FOLD_DEADLINE_SECONDS=240 budget, so
# a slow ESMFold tiling job returns a partial result instead of being killed.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 2 --timeout 300 --graceful-timeout 30 --access-logfile - app.server:app"]
