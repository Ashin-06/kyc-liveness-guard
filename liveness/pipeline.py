"""Combine voice, pulse, lip-sync and challenge evidence into one decision."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .audio import AudioAnalyzer, speech_onsets
from .challenge import check_responses
from .lipsync import envelope_per_frame, lipsync_risk, sync_score
from .rppg import VideoTraces, analyze_traces

WEIGHTS = {"voice": 0.35, "pulse": 0.35, "lipsync": 0.30}
PASS_BELOW = 0.35
REJECT_ABOVE = 0.60


@dataclass
class Check:
    name: str
    risk: float | None
    status: str
    reasons: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)


@dataclass
class Decision:
    decision: str  # PASS | STEP_UP | REJECT
    risk: float
    checks: list[Check]

    def as_dict(self) -> dict:
        return {
            "decision": self.decision,
            "risk": round(self.risk, 3),
            "checks": [
                {"name": c.name, "risk": None if c.risk is None else round(c.risk, 3), "status": c.status,
                 "reasons": c.reasons, "data": c.data}
                for c in self.checks
            ],
        }


def _status(r: float) -> str:
    return "pass" if r < PASS_BELOW else ("warn" if r < REJECT_ABOVE else "fail")


def evaluate(
    audio: tuple[np.ndarray, int] | None = None,
    video: VideoTraces | None = None,
    challenge: dict | None = None,
    analyzer: AudioAnalyzer | None = None,
) -> Decision:
    checks: list[Check] = []
    hard_fail = False

    if audio is not None:
        x, sr = audio
        rep = (analyzer or AudioAnalyzer()).analyze(x, sr)
        keep = ("noise_floor_db", "digital_silence_ratio", "hf_cliff_db", "hf_edge_hz", "offset_decay_db", "jitter", "pitch_std_st", "voiced_frames")
        checks.append(Check("voice", rep.risk, _status(rep.risk), rep.reasons,
                            {k: rep.features[k] for k in keep} | {"engine": rep.source}))

    if video is not None:
        if video.face_ratio < 0.5:
            hard_fail = True
            checks.append(Check("face", 1.0, "fail", [f"face visible in only {video.face_ratio:.0%} of frames"]))
        pr = analyze_traces(video.regions, video.fps)
        checks.append(Check("pulse", pr.risk, _status(pr.risk), pr.reasons,
                            {"bpm": round(pr.bpm, 1), "snr_db": round(pr.snr_db, 2), "region_bpm": {k: round(v, 1) for k, v in pr.region_bpm.items()},
                             "bpm_spread": round(pr.bpm_spread, 1)}))
        if audio is not None:
            env = envelope_per_frame(audio[0], audio[1], video.fps, len(video.mouth_motion))
            corr, lag = sync_score(env, video.mouth_motion, video.fps)
            lr, reasons = lipsync_risk(corr)
            checks.append(Check("lipsync", lr, _status(lr), reasons, {"corr": round(corr, 3), "lag_s": round(lag, 3)}))

    if challenge is not None:
        onsets = speech_onsets(*audio) if audio is not None else []
        res = check_responses(challenge, onsets, video.yaw if video else None, video.fps if video else None)
        reasons = []
        if not res["speech_timing_ok"]:
            missed = [c["value"] for c in res["cues"] if c["kind"] == "speak" and not c["passed"]]
            reasons.append(f"speech did not start on cue for: {', '.join(missed)}")
        if res["head_turn_ok"] is False:
            reasons.append("head turns did not match the requested directions")
        if res["stray_onsets"] > 5:
            reasons.append(f"{res['stray_onsets']} speech bursts outside cue windows (continuous playback?)")
        hard_fail |= not res["passed"]
        checks.append(Check("challenge", 0.0 if res["passed"] else 1.0, "pass" if res["passed"] else "fail", reasons, res))

    scored = [c for c in checks if c.name in WEIGHTS]
    total_w = sum(WEIGHTS[c.name] for c in scored) or 1.0
    risk = sum(WEIGHTS[c.name] * c.risk for c in scored) / total_w if scored else 1.0
    # one strongly failing signal should not be averaged away by two clean ones
    worst = max((c.risk for c in scored), default=0.0)
    risk = max(risk, worst - 0.25)

    if hard_fail or risk >= REJECT_ABOVE:
        decision = "REJECT"
    elif risk >= PASS_BELOW or not scored:
        decision = "STEP_UP"
    else:
        decision = "PASS"
    return Decision(decision, float(risk), checks)
