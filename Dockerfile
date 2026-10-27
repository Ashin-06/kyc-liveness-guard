# Runtime image: dependencies from the pinned lock, source copied in, runs as a non-root user.
FROM python:3.12-slim-bookworm AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.lock /tmp/
RUN python -m venv /opt/venv && /opt/venv/bin/pip install -r /tmp/requirements.lock

FROM python:3.12-slim-bookworm
ENV PATH=/opt/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    LIVENESS_ENV=production LIVENESS_HOST=0.0.0.0 LIVENESS_PORT=8001 \
    LIVENESS_DATABASE_URL=sqlite:////data/liveness.db
RUN useradd --system --uid 10001 --no-create-home app && mkdir /data && chown app /data
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY liveness liveness
COPY web web
# only the ASVspoof-trained 16 kHz model is in the repository; 48 kHz audio uses the rules
COPY models/voice_model_asvspoof.joblib models/
USER app
VOLUME /data
EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ['LIVENESS_PORT'], timeout=2)"]
CMD ["python", "-m", "liveness", "serve"]
