# striatica pipeline — CPU variant
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

WORKDIR /app

# === UMAP REPRODUCIBILITY CHAIN — exact pins ===
# These versions produced the Feb 27 2026 dataset positions.
# Changing ANY of these will produce different 3D embeddings
# even with the same random_state. Do not widen these ranges.
RUN pip install --no-cache-dir \
    "numpy==2.4.2" \
    "scipy==1.17.1" \
    "scikit-learn==1.8.0" \
    "umap-learn==0.5.11" \
    "hdbscan==0.8.41"

# Other pipeline deps
RUN pip install --no-cache-dir \
    "requests>=2.32.5,<3.0.0" \
    "huggingface-hub>=0.28.0,<1.0.0" \
    "safetensors>=0.5.0,<1.0.0" \
    "python-dotenv>=1.1.0,<2.0.0"

# ML optional deps (SAELens, TransformerLens, transformers)
# CPU torch pulled automatically as a dep
RUN pip install --no-cache-dir \
    "sae-lens>=6.37.0" \
    "transformer-lens>=2.12.0" \
    "transformers>=4.40.0,<5.0.0"

# Copy source and install project (striat entrypoint)
COPY pyproject.toml ./
COPY pipeline/ pipeline/
COPY scripts/ scripts/
RUN pip install --no-cache-dir -e .

# Output directory
RUN mkdir -p /app/output /app/data
ENV STRIATICA_OUTPUT_DIR=/app/output
ENV STRIATICA_DATA_DIR=/app/data

ENTRYPOINT ["python", "-m", "pipeline.cli"]
CMD ["--help"]
