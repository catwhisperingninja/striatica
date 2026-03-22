# striatica pipeline — CPU variant
# GPU variant: change base image to nvidia/cuda:12.x-runtime-ubuntu22.04
# and install torch with CUDA support
#
# Usage:
#   docker build -t striatica-pipeline .
#   docker run -v $(pwd)/output:/app/output striatica-pipeline discover --sae-types res
#   docker run -v $(pwd)/output:/app/output striatica-pipeline model --np-id gpt2-small/6-res-jb
#   docker run -v $(pwd)/output:/app/output striatica-pipeline batch --np-ids '...'

FROM python:3.12-slim

# System deps for numpy/scipy builds (if needed) + git for pip installs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml poetry.lock* ./

# Install dependencies only (--no-root skips installing the project itself,
# which would fail because the source code isn't copied yet)
# CRITICAL: Do NOT run `poetry lock` here. The lockfile pins exact versions
# that produce reproducible UMAP embeddings. Re-locking can pull newer patch
# versions of umap-learn/numpy/scipy/numba/pynndescent, which changes the
# 3D positions even with the same random_state=42.
RUN poetry config virtualenvs.create false && \
    poetry install --extras ml --without dev --no-interaction --no-root

# Copy pipeline code
COPY pipeline/ pipeline/
COPY scripts/ scripts/

# Now install the project itself (the striat entry point)
RUN poetry install --extras ml --without dev --no-interaction

# Output directory
RUN mkdir -p /app/output /app/data
ENV STRIATICA_OUTPUT_DIR=/app/output
ENV STRIATICA_DATA_DIR=/app/data

ENTRYPOINT ["python", "-m", "pipeline.cli"]
CMD ["--help"]
