"""Red-team benchmark with ISO/IEC 30107-3 style metrics.

APCER: share of attack presentations that were accepted (decision PASS).
BPCER: share of bona fide presentations that were not accepted.
"""
from __future__ import annotations

import numpy as np

from . import synth
from .challenge import USER_TO_IMAGE, ChallengeService
from .pipeline import evaluate
from .rppg import VideoTraces

SR, FPS = 48000, 30.0

SCENARIOS = {
    "bona_fide": "live person following the cues",
    "photo_tts": "printed photo held to camera + TTS clip played from a laptop",
    "screen_replay": "replay of a previous genuine session on a phone screen",
    "faceswap_clone": "real-time face swap + voice clone, attacker follows cues live",
    "clone_noise_masked": "attacker's real face + cloned voice with added room noise",
    "virtual_cam_old_session": "injected stream of an older session (bypasses camera)",
}


def _yaw(cues, n, rng, follow: bool) -> list[str]:
    yaw = ["front"] * n
    for c in cues:
        if c["kind"] != "turn":
            continue
        direction = USER_TO_IMAGE[c["value"]] if follow else rng.choice(["left", "right"])
        start = c["t"] + (rng.uniform(0.4, 0.9) if follow else rng.uniform(-2.0, 2.0))
        for i in range(int(start * FPS), int((start + rng.uniform(0.6, 1.0)) * FPS)):
            if 0 <= i < n:
                yaw[i] = direction if rng.random() < 0.55 else "none"  # profile cascade misses frames
    return yaw


def session(kind: str, payload: dict, seed: int):
    rng = np.random.default_rng(seed)
    dur = payload["duration"]
    n = int(dur * FPS)
    speak_t = [c["t"] + rng.uniform(0.35, 0.9) for c in payload["cues"] if c["kind"] == "speak"]
    stale_t = sorted(rng.uniform(0.8, dur - 1.0, size=len(speak_t)))  # timing of some other session

    if kind == "bona_fide":
        audio = synth.voice(SR, dur, seed, word_times=speak_t)
        face, follow, sync = "live", True, True
    elif kind == "photo_tts":
        audio = synth.tts(SR, dur, seed, word_times=stale_t)
        face, follow, sync = "photo", False, False
    elif kind == "screen_replay":
        audio = synth.replay(SR, dur, seed, word_times=stale_t)
        face, follow, sync = "screen", False, True
    elif kind == "faceswap_clone":
        audio = synth.tts(SR, dur, seed, native_sr=22050, add_noise=True, word_times=speak_t)
        face, follow, sync = "deepfake", True, True
    elif kind == "clone_noise_masked":
        audio = synth.tts(SR, dur, seed, add_noise=True, word_times=speak_t)
        face, follow, sync = "live", True, False
    elif kind == "virtual_cam_old_session":
        audio = synth.voice(SR, dur, seed, word_times=stale_t)
        face, follow, sync = "live", False, True
    else:
        raise ValueError(kind)

    regions = synth.face_traces(face, FPS, dur, seed)
    src = audio if sync else synth.voice(SR, dur, seed + 1000)
    mouth = synth.mouth_from_audio(src, SR, FPS, n, seed)
    video = VideoTraces(FPS, {k: v[:n] for k, v in regions.items()}, mouth, 1.0, _yaw(payload["cues"], n, rng, follow))
    return (audio, SR), video


def run(trials: int = 20, seed: int = 0) -> dict:
    svc = ChallengeService(b"redteam")
    results = {}
    for kind in SCENARIOS:
        decisions, risks, reasons = [], [], {}
        for i in range(trials):
            _, payload = svc.issue()
            audio, video = session(kind, payload, seed * 10_000 + i)
            d = evaluate(audio, video, payload)
            decisions.append(d.decision)
            risks.append(d.risk)
            for c in d.checks:
                for r in c.reasons:
                    key = c.name
                    reasons[key] = reasons.get(key, 0) + 1
        accepted = decisions.count("PASS") / trials
        results[kind] = {
            "description": SCENARIOS[kind],
            "trials": trials,
            "pass": decisions.count("PASS"),
            "step_up": decisions.count("STEP_UP"),
            "reject": decisions.count("REJECT"),
            "mean_risk": float(np.mean(risks)),
            ("bpcer" if kind == "bona_fide" else "apcer"): (1 - accepted) if kind == "bona_fide" else accepted,
            "triggered_checks": reasons,
            "risks": risks,
        }
    return results


def table(results: dict) -> str:
    lines = [f"{'scenario':26} {'pass':>5} {'step':>5} {'rej':>5} {'risk':>6}  metric", "-" * 64]
    for k, r in results.items():
        metric = f"BPCER {r['bpcer']:.1%}" if "bpcer" in r else f"APCER {r['apcer']:.1%}"
        lines.append(f"{k:26} {r['pass']:5d} {r['step_up']:5d} {r['reject']:5d} {r['mean_risk']:6.2f}  {metric}")
    return "\n".join(lines)
