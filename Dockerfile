FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml ./
COPY src/ ./src/
COPY config/ ./config/

# Install production dependencies (no dev extras)
RUN uv pip install --system --no-cache .

# /app/src exposes the `normarag` package; /app exposes the root `config` package.
ENV PYTHONPATH=/app/src:/app

CMD ["uvicorn", "normarag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
