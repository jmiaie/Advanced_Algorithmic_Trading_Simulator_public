# Public portfolio quickstart image — synthetic offline demo only (scripts/main.py).
# Matches pyproject requires-python (>=3.11). Not a live-trading container.
FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the project before editable install (requirements.txt is `-e .[live]`).
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

# Core + live extras available; default CMD needs only the research package.
RUN pip install --no-cache-dir -e ".[live]"

CMD ["python", "scripts/main.py"]
