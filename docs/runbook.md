# Runbook

How to deploy, configure and operate the liveness service. Commands assume the repository root.

## Deploy

```bash
docker build -t kyc-liveness-guard:1.0.0 .
python -m liveness.ops keygen          # one API key per client, plus the challenge key
docker run -d --name liveness -p 8001:8001 -v liveness-data:/data \
  -e LIVENESS_KEYS_FILE=/run/secrets/liveness-keys.json \
  -e LIVENESS_CHALLENGE_KEY=... \
  kyc-liveness-guard:1.0.0
curl -fsS http://127.0.0.1:8001/readyz
```

The image runs in production mode as a non-root user and keeps its SQLite database (redeemed
challenge ids only) in the `/data` volume. It refuses to start, and logs why, when anything
required is missing. Terminate TLS in front of it.

The image contains the ASVspoof-trained 16 kHz voice model only. Audio above 16 kHz (the
browser capture) is scored by the rules, which is also what a fresh clone of the repository
does. To use a model of your own for wideband audio, mount it and set `LIVENESS_MODEL`.

For more than one replica, use Postgres (`LIVENESS_DATABASE_URL=postgresql://user:pass@host/db`)
so a token redeemed on one replica is refused on the others, give every replica the same
challenge key, and put a shared rate limiter at the ingress.

Verification is CPU-bound (video decoding, face detection, pulse extraction) and has not been
load-tested. Measure it on your own capture lengths before sizing replicas.

Without Docker: `pip install -r requirements.lock && pip install -e .`, then `liveness serve`.

## Configure

| Variable | Required in production | Meaning |
|---|---|---|
| `LIVENESS_ENV` | — | `development` (default) or `production` |
| `LIVENESS_API_KEYS` / `LIVENESS_KEYS_FILE` | yes | JSON `{"sha256:<hex>": {"user": "...", "role": "..."}}`. Roles: `kyc_app` (challenge and verify), `analyst` (also `/v1/analyze/*`), `monitor` (`/metrics`), `admin` |
| `LIVENESS_CHALLENGE_KEY` | yes | base64, at least 32 bytes; signs challenge tokens (replaces the old hex `LIVENESS_SECRET`) |
| `LIVENESS_DATABASE_URL` | yes | `sqlite:////data/liveness.db` or `postgresql://...` (default in development: in memory) |
| `LIVENESS_HOST`, `LIVENESS_PORT` | — | default `127.0.0.1:8001`; any non-loopback address needs API keys |
| `LIVENESS_RATE_LIMIT` | — | requests per minute per client (default 120) |
| `LIVENESS_MAX_BODY_BYTES` | — | default 50,000,000 (one capture); each uploaded file is also capped at 40 MB |
| `LIVENESS_CORS_ORIGINS` | — | comma-separated origins allowed to call from a browser; none by default |
| `LIVENESS_MODEL`, `LIVENESS_MODEL_16K` | — | voice model files (defaults under `models/`) |
| `LIVENESS_LOG_LEVEL` | — | default `INFO` |
| `FORWARDED_ALLOW_IPS` | — | addresses of proxies whose `X-Forwarded-For` is trusted (uvicorn) |

## Watch

- `GET /healthz`: the process is up (used by the container health check).
- `GET /readyz`: database reachable and the 16 kHz voice model loaded; 503 otherwise.
- `GET /metrics` (role `monitor`): `http_requests_total`, `http_request_duration_seconds`,
  `errors_total`, `liveness_decisions_total{decision="PASS|STEP_UP|REJECT|challenge_rejected"}`.
- Logs are one JSON object per line on stdout, each with a `request_id` that is also returned in
  the `X-Request-ID` response header. Each verification logs the caller, challenge id, decision
  and risk; never the recordings.

Suggested alerts: `/readyz` failing for 2 minutes; `errors_total{kind="unhandled"}` increasing;
5xx share above 1% for 10 minutes; p95 of `/v1/verify` latency above your SLO; the share of
`REJECT` or `challenge_rejected` moving sharply away from its usual level (a new attack tool, or
a broken client release).

## Rotate keys

**API keys.** Add the new hash to the key file, restart, give the client the new key, then remove
the old hash and restart again.

**Challenge key.** Set `LIVENESS_CHALLENGE_KEY_PREVIOUS` to the current key and
`LIVENESS_CHALLENGE_KEY` to a new one, then restart. Tokens already issued stay redeemable.
After 3 minutes (the token lifetime) remove `_PREVIOUS` and restart.

## Back up and restore

The database holds only redeemed challenge ids that expire within 3 minutes; there is nothing
worth restoring. If it is lost, tokens issued in the last 3 minutes could be redeemed a second
time; rotating the challenge key at the same moment closes that window.

Migrations are applied automatically on start, or with `python -m liveness migrate`.

## Incidents

1. **Leaked API key.** Remove its hash from the key file and restart. Search the logs for that
   user. Issue a new key.
2. **Leaked challenge key.** Rotate it without `_PREVIOUS` and restart: every outstanding token
   becomes invalid and customers in the middle of a check start again.
3. **A new attack passes.** Find the sessions by challenge id in the logs and get the recordings
   from the onboarding app (the service keeps none). Route the affected channel to STEP_UP while
   you add a check and a regression case to the benchmark.
4. **Real customers rejected in numbers.** Compare the decision mix per app release and device
   type. Re-calibrate the thresholds listed in the README (Calibration) on recordings from that
   channel; until then, route REJECT to manual review.
5. **Decoder crash or hang on an upload.** The request fails with a 500 and a request ID; find the
   upload in the app, and check for a patched OpenCV / libsndfile wheel.
