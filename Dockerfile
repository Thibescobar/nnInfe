FROM python:3.10-slim

WORKDIR /app

# Install dependencies (SimpleITK might need some system packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY nninfe/ nninfe/

RUN pip install --no-cache-dir ".[cpu]"

CMD ["nninfe-det", "--help"]