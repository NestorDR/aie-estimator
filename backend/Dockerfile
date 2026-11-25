# docker/Dockerfile

# Stage 1 — Builder
# Multi-stage build isolates uv, build backend (uv_build), and cache artifacts
# from the final runtime image, minimizing image size and attack surface.
FROM python:3.14-slim-trixie AS builder

# Pin uv version explicitly (avoid mutable :latest tag per best practices).
# Matches uv_build>=0.13.1,<0.14.0 required in pyproject.toml.
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

# Configure uv for deterministic, compiled container builds:
# - UV_COMPILE_BYTECODE=1: Pre-compile .pyc files for faster container startup
# - UV_LINK_MODE=copy: Copy files from cache instead of hardlinking across mounts
# - UV_PYTHON_DOWNLOADS=never: Use the container's system Python 3.14 interpreter
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Step 1: Copy only dependency manifests first to maximize Docker layer caching.
# Dependencies are only re-resolved when pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock ./

# Install production third-party dependencies without installing the project yet.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# Step 2: Copy project metadata and application source required by uv_build.
COPY README.md ./
COPY src/ ./src/

# Install the aie_estimator package itself (--no-editable embeds it cleanly in
# site-packages so importlib.metadata.version("aie_estimator") and the
# `aie-estimator` console script resolve without needing raw source on PYTHONPATH).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable --no-dev




# Stage 2 — Runtime
# Clean minimal runtime image with the same Debian Trixie libc ecosystem.
FROM python:3.14-slim-trixie  AS runtime

# OCI-standard image provenance labels
LABEL org.opencontainers.image.title="aie-estimator" \
    org.opencontainers.image.description="FastAPI backend service for automated software engineering effort estimation" \
    org.opencontainers.image.version="0.1.0"

# Create a dedicated unprivileged system group and user with a fixed UID/GID (10001).
RUN groupadd --system --gid 10001 appgroup && \
    useradd --system --uid 10001 --gid appgroup --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

# Copy only the pre-built virtual environment (which already contains the
# non-editable `aie_estimator` package and its compiled bytecode) with non-root
# ownership set at copy time (avoids an expensive recursive `RUN chown -R` layer).
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv

# Runtime environment configuration:
# - PATH: Resolve `python`, `uvicorn`, and `aie-estimator` from the venv
# - PYTHONUNBUFFERED=1: Stream stdout/stderr logs immediately without buffering
# - PYTHONDONTWRITEBYTECODE=1: Prevent runtime .pyc writes (supports read-only rootfs)
# - APP_ENV=prod: Default to production mode (JSON logging, reload disabled)
# - SERVER_HOST=0.0.0.0: Listen on all container interfaces instead of localhost
# - SERVER_PORT=8000: Default listening port
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=prod \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=8000

# Switch to the unprivileged user before runtime.
USER appuser

EXPOSE 8000

# Docker-native health check probing GET /health via Python stdlib (no curl in slim).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"]

# Exec-form entrypoint invoking the package console script (`aie_estimator.__main__:main`),
# which initializes unified structlog formatting and launches Uvicorn with `log_config=None`.
CMD ["aie-estimator"]