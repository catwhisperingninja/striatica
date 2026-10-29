# striatica pipeline — CPU variant
#
# Usage:
#   docker build -t striatica-pipeline .
#   docker run -v $(pwd)/output:/app/output striatica-pipeline discover --sae-types res
#   docker run -v $(pwd)/output:/app/output striatica-pipeline model --np-id gpt2-small/6-res-jb
#   docker run -v $(pwd)/output:/app/output striatica-pipeline batch --np-ids '...'

FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159

# System deps for numpy/scipy builds (if needed) + git for pip installs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Stream CLI and child-process output immediately, including without a TTY.
ENV PYTHONUNBUFFERED=1

# Keep Poetry's tools separate from the locked application environment.
RUN pip install --no-cache-dir "poetry==2.5.1"
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV PATH="/app/.venv/bin:$PATH"

# Install every application dependency (including transitives) from the lock.
# Never resolve new versions or regenerate the lockfile during a build.
COPY pyproject.toml poetry.lock ./
RUN poetry install --extras ml --no-root --no-interaction --no-ansi

# Copy source and install only the project (striat entrypoint).
COPY pipeline/ pipeline/
COPY scripts/ scripts/
RUN poetry install --only-root --no-interaction --no-ansi

# Output directory
RUN mkdir -p /app/output /app/data
ENV STRIATICA_OUTPUT_DIR=/app/output
ENV STRIATICA_DATA_DIR=/app/data

ENTRYPOINT ["python", "-m", "pipeline.cli"]
CMD ["--help"]
