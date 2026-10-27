"""Settings, auth, limits, headers, metrics, health, persistence and migrations."""
import base64
import hashlib
import importlib
import json
import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from liveness import db, ops
from liveness.challenge import ChallengeError, ChallengeService

C32 = base64.b64encode(b"c" * 32).decode()
KEY = "liveness_" + "x" * 40
HASHED = {f"sha256:{hashlib.sha256(KEY.encode()).hexdigest()}": {"user": "onboarding", "role": "kyc_app"}}


def prod_env(**over):
    env = {"LIVENESS_ENV": "production", "LIVENESS_API_KEYS": json.dumps(HASHED), "LIVENESS_CHALLENGE_KEY": C32,
           "LIVENESS_DATABASE_URL": "sqlite://"}
    env.update(over)
    return {k: v for k, v in env.items() if v is not None}


def boot(monkeypatch, **env):
    for k in [k for k in os.environ if k.startswith("LIVENESS_")]:
        monkeypatch.delenv(k)
    for k, v in env.items():
        monkeypatch.setenv(f"LIVENESS_{k}", v)
    import liveness.api as api

    return importlib.reload(api)


def test_production_settings_accepted():
    s = ops.load_settings(prod_env())
    assert s.production and not s.demo and s.secret["CHALLENGE_KEY"] == b"c" * 32 and not s.ephemeral


@pytest.mark.parametrize("change, message", [
    ({"LIVENESS_API_KEYS": None}, "API keys"),
    ({"LIVENESS_CHALLENGE_KEY": None}, "CHALLENGE_KEY"),
    ({"LIVENESS_DATABASE_URL": None}, "DATABASE_URL"),
    ({"LIVENESS_API_KEYS": json.dumps({KEY: {"user": "onboarding", "role": "kyc_app"}})}, "sha256"),
    ({"LIVENESS_API_KEYS": json.dumps({"sha256:" + "0" * 64: {"user": "x", "role": "root"}})}, "roles"),
    ({"LIVENESS_CHALLENGE_KEY": base64.b64encode(b"short").decode()}, "at least 32"),
    ({"LIVENESS_CHALLENGE_KEY": "not base64!"}, "base64"),
    ({"LIVENESS_CORS_ORIGINS": "*"}, "CORS"),
    ({"LIVENESS_ENV": "prod"}, "development or production"),
    ({"LIVENESS_DATABASE_URL": "mysql://x"}, "postgresql"),
])
def test_production_refuses_to_start(change, message):
    with pytest.raises(ops.ConfigError, match=message):
        ops.load_settings(prod_env(**change))


def test_no_public_bind_without_keys():
    with pytest.raises(ops.ConfigError, match="without API keys"):
        ops.load_settings({"LIVENESS_HOST": "0.0.0.0"})
    assert ops.load_settings({"LIVENESS_HOST": "0.0.0.0", "LIVENESS_API_KEYS": json.dumps(HASHED)}).host == "0.0.0.0"
    dev = ops.load_settings({})
    assert dev.demo and dev.ephemeral == ("CHALLENGE_KEY",) and dev.database_url == "sqlite://"


def test_serve_exits_with_a_readable_error():
    env = {k: v for k, v in os.environ.items() if not k.startswith("LIVENESS_")}
    r = subprocess.run([sys.executable, "-m", "liveness", "serve", "--host", "0.0.0.0"], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "without API keys" in r.stderr


def test_roles_decide_what_a_key_may_do(monkeypatch):
    keys = {"k-app": {"user": "onboarding", "role": "kyc_app"}, "k-an": {"user": "asha", "role": "analyst"},
            "k-mon": {"user": "prom", "role": "monitor"}}
    c = TestClient(boot(monkeypatch, API_KEYS=json.dumps(keys)).app)
    assert c.post("/v1/challenge").status_code == 401
    assert c.post("/v1/challenge", headers={"x-api-key": "wrong"}).status_code == 401
    assert c.post("/v1/challenge", headers={"x-api-key": "k-app"}).status_code == 200
    assert c.post("/v1/challenge", headers={"authorization": "Bearer k-mon"}).status_code == 403
    wav = {"audio": ("a.wav", b"RIFF", "audio/wav")}
    assert c.post("/v1/analyze/audio", files=wav, headers={"x-api-key": "k-app"}).status_code == 403
    assert c.post("/v1/analyze/audio", files=wav, headers={"x-api-key": "k-an"}).status_code == 422  # allowed, bad audio
    assert c.get("/metrics", headers={"x-api-key": "k-app"}).status_code == 403
    m = c.get("/metrics", headers={"x-api-key": "k-mon"})
    assert m.status_code == 200 and 'http_requests_total{method="POST",route="/v1/challenge",status="200"}' in m.text
    assert "http_request_duration_seconds_bucket" in m.text and "errors_total" in m.text
    assert c.get("/").status_code == 200  # the capture page itself is static


def test_rate_limit(monkeypatch):
    c = TestClient(boot(monkeypatch, RATE_LIMIT="3").app)
    codes = [c.post("/v1/challenge").status_code for _ in range(5)]
    assert codes[:3] == [200] * 3 and codes[3:] == [429, 429]
    assert int(c.post("/v1/challenge").headers["retry-after"]) >= 1
    assert c.get("/healthz").status_code == 200  # probes are never limited


def test_body_size_cap(monkeypatch):
    c = TestClient(boot(monkeypatch, MAX_BODY_BYTES="1000").app)
    assert c.post("/v1/analyze/audio", files={"audio": ("a.wav", b"a" * 2000, "audio/wav")}).status_code == 413

    def chunks():  # no content-length: the cap still holds
        for _ in range(20):
            yield b"a" * 100

    assert c.post("/v1/analyze/audio", content=chunks(), headers={"content-type": "application/octet-stream"}).status_code == 413
    assert c.post("/v1/challenge").status_code == 200


def test_headers_and_request_ids(monkeypatch):
    c = TestClient(boot(monkeypatch).app)
    r = c.post("/v1/challenge", headers={"x-request-id": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123"
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["x-frame-options"] == "DENY"
    assert r.headers["content-security-policy"] == ops.API_CSP and r.headers["cache-control"] == "no-store"
    assert "script-src 'self' 'unsafe-inline' blob:" in c.get("/").headers["content-security-policy"]  # audio worklet
    assert len(c.get("/healthz", headers={"x-request-id": "bad id\n"}).headers["x-request-id"]) == 32
    assert "access-control-allow-origin" not in c.get("/healthz", headers={"origin": "https://evil.example"}).headers


def test_cors_only_for_listed_origins(monkeypatch):
    c = TestClient(boot(monkeypatch, CORS_ORIGINS="https://kyc.bank.example").app)
    ok = c.get("/healthz", headers={"origin": "https://kyc.bank.example"})
    assert ok.headers["access-control-allow-origin"] == "https://kyc.bank.example"
    assert "access-control-allow-origin" not in c.get("/healthz", headers={"origin": "https://evil.example"}).headers


def test_errors_do_not_leak(monkeypatch):
    api = boot(monkeypatch)

    @api.app.get("/boom")
    def boom():
        raise RuntimeError("secret detail: aadhaar 2345 6789 0123")

    c = TestClient(api.app)
    r = c.get("/boom")
    assert r.status_code == 500 and "2345" not in r.text and r.json()["request_id"] == r.headers["x-request-id"]
    bad = c.post("/v1/analyze/video", data={"duration": "aadhaar-234567890123"}, files={"video": ("v.mp4", b"x", "video/mp4")})
    assert bad.status_code == 422 and "234567890123" not in bad.text
    unreadable = c.post("/v1/analyze/video", files={"video": ("v.mp4", b"not a video", "video/mp4")})
    assert unreadable.status_code == 422 and "tmp" not in unreadable.text.lower()


def test_health_and_readiness(monkeypatch):
    api = boot(monkeypatch)
    c = TestClient(api.app)
    assert c.get("/healthz").json() == {"status": "ok"}
    r = c.get("/readyz")
    assert r.status_code == 200 and r.json()["checks"] == {"database": True, "voice_model_16k": True}
    api.store.dispose()
    monkeypatch.setattr(api.store, "connect", lambda: (_ for _ in ()).throw(RuntimeError("db down")))
    r = c.get("/readyz")
    assert r.status_code == 503 and r.json()["checks"]["database"] is False and "db down" not in r.text


def test_json_logs_carry_request_ids(capsys):
    ops.setup_logging("INFO")
    token = ops.request_id.set("rid-1")
    ops.log.info("hello", extra={"fields": {"n": 1}})
    ops.request_id.reset(token)
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["msg"] == "hello" and line["request_id"] == "rid-1" and line["n"] == 1 and line["level"] == "INFO"


def _url(tmp_path):
    return os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path / 'state.db'}"


def test_used_tokens_survive_a_restart(tmp_path):
    url, key = _url(tmp_path), b"k" * 32
    first = ChallengeService(key, engine=db.connect(url))
    token, _ = first.issue()
    first.redeem(token)
    second = ChallengeService(key, engine=db.connect(url))  # a new process or replica, same database and key
    with pytest.raises(ChallengeError, match="replay"):
        second.redeem(token)

    old, _ = first.issue()
    rotated = ChallengeService(b"n" * 32, engine=db.connect(url), previous_keys=(key,))
    with pytest.raises(ChallengeError, match="signature"):
        ChallengeService(b"n" * 32, engine=db.connect(url)).redeem(old)
    assert rotated.redeem(old)["id"]  # tokens issued before a rotation still redeem


def test_migrations_apply_once(tmp_path):
    url = f"sqlite:///{tmp_path / 'm.db'}"
    eng = db.engine(url)
    assert db.migrate(eng) == ["001_init.sql"]
    assert db.migrate(eng) == []
    with eng.connect() as c:
        assert [r[0] for r in c.execute(text("SELECT version FROM schema_migrations"))] == [1]
        tables = {r[0] for r in c.execute(text("SELECT name FROM sqlite_master WHERE type = 'table'"))}
    assert "used_challenges" in tables
    env = {**{k: v for k, v in os.environ.items() if not k.startswith("LIVENESS_")}, "LIVENESS_DATABASE_URL": url}
    out = subprocess.run([sys.executable, "-m", "liveness", "migrate"], env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "up to date" in out.stdout
