FROM python:3.11-slim

# ── System deps ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────
WORKDIR /app

# ── Python deps (layer-cached) ─────────────────────────────────────────────
COPY requirements.txt .
COPY pyproject.toml  .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────────────────
COPY server.py              ./server.py
COPY openenv.yaml           ./openenv.yaml
COPY inference.py           ./inference.py
COPY test_server.py         ./test_server.py
COPY validate-submission.sh ./validate-submission.sh
COPY graders/               ./graders/

# ── Non-root user (HF Spaces requirement) ─────────────────────────────────
RUN useradd -m -u 1000 appuser
USER appuser

# ── Runtime ────────────────────────────────────────────────────────────────
EXPOSE 7860

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:7860/health || exit 1

CMD ["uvicorn", "server:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info"]
