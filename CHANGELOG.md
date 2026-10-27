# Changelog

All notable changes to this project. Versions follow [semantic versioning](https://semver.org/).

## 1.0.0 — 2026-10-27

### Added
- `pyproject.toml` with a `liveness` console command, and pinned `requirements.lock` /
  `requirements-dev.lock` (generated with `uv pip compile --universal`). scikit-learn is held at
  1.7 (the shipped voice models were pickled with it) and OpenCV at 4.x (the real-face numbers
  were measured with it).
- Container image: slim Python 3.12 base, `opencv-python-headless`, non-root user, health check
  on `/healthz`. It ships the ASVspoof-trained 16 kHz model; 48 kHz audio is scored by the rules.
- CI on every push: ruff, bandit, pip-audit, pytest with a coverage report.
- Settings from `LIVENESS_*` environment variables, checked at startup. `LIVENESS_ENV=production`
  refuses to start without hashed API keys, `LIVENESS_CHALLENGE_KEY` and
  `LIVENESS_DATABASE_URL`. No built-in secrets in any mode.
- Roles on API keys: `kyc_app` issues challenges and verifies, `analyst` also scores single
  recordings, `monitor` scrapes metrics, `admin` does all three. Keys can be stored as
  `sha256:<hex>`; `python -m liveness.ops keygen` makes new ones.
- Per-client rate limit, a 50 MB request body cap, security headers, CORS only for listed origins.
- JSON logs with request IDs; `/healthz`, `/readyz` and Prometheus `/metrics` with a
  `liveness_decisions_total` counter.
- Redeemed challenge ids in a database: SQLite by default, Postgres through
  `LIVENESS_DATABASE_URL`, numbered SQL migrations (`python -m liveness migrate`).
- Challenge-key rotation: tokens signed with `LIVENESS_CHALLENGE_KEY_PREVIOUS` still redeem.
- `SECURITY.md`, `docs/threat-model.md`, `docs/runbook.md`, `docs/model-card.md`.

### Changed
- Single-use tracking moved from process memory to the `used_challenges` table, so a replayed
  token is refused after a restart and on every replica.
- `LIVENESS_SECRET` (hex) is replaced by `LIVENESS_CHALLENGE_KEY` (base64).
- `/healthz` is a bare liveness check; the voice-engine details moved to the startup log and the
  16 kHz model check to `/readyz`.
- Validation and decoder errors no longer echo the submitted input or temporary file paths.
- The capture page sends an API key when the service has keys.
- Without API keys the service refuses to listen on anything but loopback.
- `synth.voice` no longer crashes when a pulse lands exactly on the last sample (a rounding case
  that depended on the random cue schedule and could fail a benchmark run or a test).
- Lint fixes (named lambdas, unused loop variables) with no change in behaviour.

### Benchmarks and real data
- The attack benchmark was re-run three times with the locked dependencies. Its counts vary
  between runs because the cue schedule is random; the README table and text now show a fresh run
  and the range (bona fide BPCER 0%, `faceswap_clone` APCER 15–20%) instead of the older sample
  (BPCER 5.0%, APCER 10.0%).
- Scoring code is unchanged. The ASVspoof, MCD-rPPG and GRID numbers in `results/` were not re-run
  (the datasets are not on this machine) and are unchanged.
