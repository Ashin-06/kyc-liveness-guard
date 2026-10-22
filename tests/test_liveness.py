import io
import time

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from liveness import synth
from liveness.audio import AudioAnalyzer, speech_onsets
from liveness.challenge import ChallengeError, ChallengeService, check_responses
from liveness.lipsync import envelope_per_frame, sync_score
from liveness.redteam import run
from liveness.rppg import analyze_traces


def test_voice_rules_separate_genuine_from_synthetic():
    a = AudioAnalyzer()
    genuine = [a.analyze(synth.voice(seed=s), 48000).risk for s in range(3)]
    tts = [a.analyze(synth.tts(seed=s), 48000).risk for s in range(3)]
    masked = [a.analyze(synth.tts(seed=s, add_noise=True), 48000).risk for s in range(3)]
    assert max(genuine) < 0.3
    assert min(tts) > 0.7
    assert min(masked) > 0.6  # adaptive attacker still caught by spectral edge + jitter


def test_pulse_found_on_live_face_only():
    live = analyze_traces(synth.face_traces("live", seed=1, bpm=72), 30.0)
    assert abs(live.bpm - 72) < 6 and live.risk < 0.3
    for kind in ("photo", "deepfake"):
        assert analyze_traces(synth.face_traces(kind, seed=1), 30.0).risk > 0.7


def test_lipsync_detects_dubbing():
    x = synth.voice(48000, 6.0, seed=4)
    n = 180
    own = synth.mouth_from_audio(x, 48000, 30.0, n, seed=1)
    other = synth.mouth_from_audio(synth.voice(48000, 6.0, seed=5), 48000, 30.0, n, seed=1)
    env = envelope_per_frame(x, 48000, 30.0, n)
    assert sync_score(env, own, 30.0)[0] > 0.5
    assert sync_score(env, other, 30.0)[0] < 0.35


def test_challenge_token_is_single_use_and_signed():
    svc = ChallengeService(b"k")
    token, _ = svc.issue()
    svc.redeem(token)
    with pytest.raises(ChallengeError, match="replay"):
        svc.redeem(token)
    t2, _ = svc.issue()
    body, sig = t2.split(".")
    with pytest.raises(ChallengeError, match="signature"):
        svc.redeem(body[:-2] + "AA." + sig)
    t3, _ = svc.issue(now=time.time() - 1000)
    with pytest.raises(ChallengeError, match="expired"):
        svc.redeem(t3)


def test_speech_must_follow_cues():
    _, payload = ChallengeService(b"k").issue(n_speak=3, n_turn=0)
    speak = [c["t"] + 0.5 for c in payload["cues"]]
    on_cue = synth.voice(48000, payload["duration"], 1, word_times=speak)
    off_cue = synth.voice(48000, payload["duration"], 1, word_times=[t - 1.2 for t in speak])
    assert check_responses(payload, speech_onsets(on_cue, 48000), None, None)["speech_timing_ok"]
    assert not check_responses(payload, speech_onsets(off_cue, 48000), None, None)["speech_timing_ok"]


def test_benchmark_rates():
    res = run(trials=4, seed=1)
    assert res["bona_fide"]["bpcer"] <= 0.25
    assert all(r["apcer"] == 0 for k, r in res.items() if k != "bona_fide")


def test_api_round_trip():
    from liveness.api import app

    c = TestClient(app)
    ch = c.post("/v1/challenge").json()
    speak = [q["t"] + 0.5 for q in ch["cues"] if q["kind"] == "speak"]
    buf = io.BytesIO()
    sf.write(buf, synth.voice(48000, ch["duration"], 2, word_times=speak), 48000, format="WAV")
    files = {"audio": ("a.wav", buf.getvalue(), "audio/wav")}
    r = c.post("/v1/verify", data={"token": ch["token"]}, files=files)
    assert r.status_code == 200
    names = {x["name"] for x in r.json()["checks"]}
    assert {"voice", "challenge"} <= names
    replay = c.post("/v1/verify", data={"token": ch["token"]}, files=files)
    assert replay.status_code == 401
