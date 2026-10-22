"""Voice anti-spoofing features.

Every feature here targets a concrete artifact of synthetic or replayed speech:

* digital silence / missing noise floor  -> TTS and vocoder output rendered offline
* missing high band (low roll-off)       -> audio generated at 16/22 kHz and upsampled
* missing low band                       -> playback through a phone or laptop speaker
* unnaturally stable pitch (low jitter)  -> parametric and many neural TTS systems
* flat pitch contour                     -> monotone synthesis
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.fft import dct, rfft

ANALYSIS_SR = 16000
EPS = 1e-12


def load_audio(src: str | Path | bytes) -> tuple[np.ndarray, int]:
    """Read wav/flac/ogg from a path or raw bytes, return mono float32 and sample rate."""
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    x, sr = sf.read(src, dtype="float32", always_2d=True)
    return x.mean(axis=1), int(sr)


def frame_signal(x: np.ndarray, sr: int, win: float = 0.025, hop: float = 0.010) -> np.ndarray:
    n, h = int(win * sr), int(hop * sr)
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    count = 1 + (len(x) - n) // h
    idx = np.arange(n)[None, :] + h * np.arange(count)[:, None]
    return x[idx]


def frame_db(x: np.ndarray, sr: int, win: float = 0.025, hop: float = 0.010) -> np.ndarray:
    rms = np.sqrt(np.mean(frame_signal(x, sr, win, hop) ** 2, axis=1))
    return 20 * np.log10(rms + 1e-10)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 40, fmin: float = 20.0, fmax: float | None = None) -> np.ndarray:
    fmax = fmax or sr / 2
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = hz(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        lo, c, hi = bins[i - 1], bins[i], bins[i + 1]
        if c > lo:
            fb[i - 1, lo:c] = (np.arange(lo, c) - lo) / (c - lo)
        if hi > c:
            fb[i - 1, c:hi] = (hi - np.arange(c, hi)) / (hi - c)
    return fb


def mfcc(x: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    fr = frame_signal(x, sr) * np.hamming(int(0.025 * sr))
    n_fft = 512
    power = np.abs(rfft(fr, n=n_fft)) ** 2
    mel = np.log(power @ mel_filterbank(sr, n_fft).T + EPS)
    return dct(mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]


def f0_track(x: np.ndarray, sr: int, fmin: float = 70, fmax: float = 400) -> np.ndarray:
    """Per-frame F0 via normalised autocorrelation; 0 for unvoiced frames."""
    fr = frame_signal(x, sr, win=0.040, hop=0.010)
    fr = fr - fr.mean(axis=1, keepdims=True)
    energy = np.sum(fr**2, axis=1)
    active = energy > np.percentile(energy, 30) * 2 + 1e-9
    n = fr.shape[1]
    spec = rfft(fr * np.hanning(n), n=2 * n)
    ac = np.fft.irfft(np.abs(spec) ** 2)[:, :n]
    ac /= ac[:, :1] + EPS
    lo, hi = int(sr / fmax), min(int(sr / fmin), n - 1)
    lag = lo + np.argmax(ac[:, lo:hi], axis=1)
    peak = ac[np.arange(len(lag)), lag]
    # parabolic interpolation for sub-sample lag precision; jitter lives in that precision
    l = np.clip(lag, 1, n - 2)
    a, b, c = ac[np.arange(len(l)), l - 1], ac[np.arange(len(l)), l], ac[np.arange(len(l)), l + 1]
    shift = 0.5 * (a - c) / (a - 2 * b + c + EPS)
    f0 = sr / (l + np.clip(shift, -0.5, 0.5))
    return np.where(active & (peak > 0.45), f0, 0.0)


def pitch_stats(f0: np.ndarray) -> tuple[float, float, int]:
    """(jitter, pitch std in semitones, voiced frame count) over contiguous voiced runs."""
    voiced = f0 > 0
    rel = []
    for run in np.split(np.arange(len(f0)), np.where(np.diff(voiced.astype(int)) != 0)[0] + 1):
        if len(run) >= 4 and voiced[run[0]]:
            p = 1.0 / f0[run]
            rel.extend(np.abs(np.diff(p)) / p[:-1])
    v = f0[voiced]
    if len(v) < 10:
        return float("nan"), float("nan"), int(len(v))
    semis = 12 * np.log2(v / np.median(v))
    return float(np.median(rel)) if rel else float("nan"), float(np.std(semis)), int(len(v))


def hf_cliff(x: np.ndarray, sr: int) -> tuple[float, float]:
    """Largest drop (dB) between a 1.5 kHz band and *everything* above it.

    A live microphone always has broadband self-noise up to Nyquist, and speech fades
    out gradually. Audio synthesized at 16/22 kHz and upsampled has a brick-wall edge
    with nothing above it. Returns (cliff dB, edge frequency Hz)."""
    f, p = signal.welch(x, sr, nperseg=min(len(x), 4096))
    level = 10 * np.log10(p + 1e-20)
    top = 0.9 * sr / 2  # ignore the capture device's own anti-alias roll-off
    best, edge = 0.0, top
    for fc in np.arange(3000.0, top - 1000, 250.0):
        below = level[(f >= fc - 1500) & (f < fc)].mean()
        above = level[(f >= fc + 600) & (f < top)].max()
        if below - above > best:
            best, edge = below - above, fc
    return float(best), float(edge)


def offset_decay(x: np.ndarray, sr: int) -> float:
    """Median level drop (dB) in the 120 ms after each word ends. Close-talk speech
    falls to the noise floor fast; playback through a speaker adds a second room's
    reverberation tail."""
    db = frame_db(x, sr)
    act = voice_activity(x, sr)
    drops = []
    for i in np.flatnonzero(act[:-1] & ~act[1:]):
        seg = db[i : i + 13]
        if len(seg) == 13:
            drops.append(db[max(i - 3, 0) : i + 1].max() - seg.min())
    return float(np.median(drops)) if drops else float("nan")


@dataclass
class AudioReport:
    features: dict
    risk: float
    reasons: list[str] = field(default_factory=list)
    source: str = "rules"


FEATURE_ORDER = [
    "noise_floor_db", "dynamic_range_db", "digital_silence_ratio", "hf_cliff_db", "hf_ratio_db",
    "lf_ratio_db", "offset_decay_db", "jitter", "pitch_std_st", "flatness_mean", "flatness_std",
]


def extract_features(x: np.ndarray, sr: int) -> dict:
    x = x.astype(np.float64)
    x = x - x.mean()
    peak = np.max(np.abs(x)) + EPS
    x = x / peak  # level independent
    db = frame_db(x, sr)
    rms = 10 ** (db / 20)

    f, p = signal.welch(x, sr, nperseg=min(len(x), 2048))
    nyq = sr / 2
    cliff, edge = hf_cliff(x, sr)
    hf_lo = min(7000.0, 0.85 * nyq)
    speech = p[(f >= 300) & (f < hf_lo)].sum() + EPS
    hf = p[f >= hf_lo].sum() + EPS
    lf = p[(f >= 20) & (f < 150)].sum() + EPS

    x16 = signal.resample_poly(x, ANALYSIS_SR, sr) if sr != ANALYSIS_SR else x
    jit, pstd, nvoiced = pitch_stats(f0_track(x16, ANALYSIS_SR))

    fr = frame_signal(x16, ANALYSIS_SR) * np.hamming(int(0.025 * ANALYSIS_SR))
    spec = np.abs(rfft(fr, n=512)) ** 2 + EPS
    flat = np.exp(np.mean(np.log(spec), axis=1)) / np.mean(spec, axis=1)
    db16 = frame_db(x16, ANALYSIS_SR)
    loud = db16 > np.percentile(db16, 50)

    mf = mfcc(x16, ANALYSIS_SR)
    feats = {
        "duration_s": len(x) / sr,
        "sample_rate": sr,
        "noise_floor_db": float(np.percentile(db, 10)),
        "dynamic_range_db": float(np.percentile(db, 95) - np.percentile(db, 10)),
        "digital_silence_ratio": float(np.mean(rms < 1e-5)),
        "hf_cliff_db": cliff,
        "hf_edge_hz": edge,
        "hf_ratio_db": float(10 * np.log10(hf / speech)),
        "lf_ratio_db": float(10 * np.log10(lf / speech)),
        "offset_decay_db": offset_decay(x, sr),
        "jitter": jit,
        "pitch_std_st": pstd,
        "voiced_frames": nvoiced,
        "flatness_mean": float(np.mean(flat[loud])) if loud.any() else float("nan"),
        "flatness_std": float(np.std(flat[loud])) if loud.any() else float("nan"),
    }
    for i in range(mf.shape[1]):
        feats[f"mfcc{i}_mean"] = float(mf[:, i].mean())
        feats[f"mfcc{i}_std"] = float(mf[:, i].std())
    return feats


def feature_vector(feats: dict) -> np.ndarray:
    keys = FEATURE_ORDER + [k for k in sorted(feats) if k.startswith("mfcc")]
    return np.nan_to_num(np.array([feats[k] for k in keys], dtype=float), nan=0.0)


# Rule thresholds. These are calibration knobs: tune them on recordings from your own
# capture channel (browser + mic) before trusting the rule path in production.
RULES = {
    "digital_silence_ratio": 0.03,  # share of frames below -100 dBFS
    "noise_floor_db": -82.0,        # quieter than any real microphone chain
    "hf_cliff_db": 15.0,            # brick-wall spectral edge
    "jitter": 0.0025,               # frame-level period perturbation
    "pitch_std_st": 0.8,            # semitones; monotone synthesis
    "offset_decay_db": 18.0,        # dB lost in 120 ms after a word ends
}


def rule_risk(feats: dict) -> tuple[float, list[str]]:
    score, reasons = -2.5, []
    if feats["digital_silence_ratio"] > RULES["digital_silence_ratio"]:
        score += 3.0
        reasons.append(f"{feats['digital_silence_ratio']:.0%} of frames are digital silence (no microphone noise floor)")
    elif feats["noise_floor_db"] < RULES["noise_floor_db"]:
        score += 2.0
        reasons.append(f"noise floor {feats['noise_floor_db']:.0f} dBFS is unrealistically clean")
    if feats["hf_cliff_db"] > RULES["hf_cliff_db"]:
        score += 2.5
        reasons.append(
            f"spectrum is cut off sharply at {feats['hf_edge_hz'] / 1000:.1f} kHz "
            f"({feats['hf_cliff_db']:.0f} dB wall): audio was produced at a lower rate and upsampled, or came through a narrowband headset"
        )
    if feats["voiced_frames"] >= 40 and not np.isnan(feats["jitter"]):
        if feats["jitter"] < RULES["jitter"]:
            score += 2.0
            reasons.append(f"pitch jitter {feats['jitter'] * 100:.3f}% is below the natural voice range")
        if feats["pitch_std_st"] < RULES["pitch_std_st"]:
            score += 0.8
            reasons.append(f"pitch varies only {feats['pitch_std_st']:.2f} semitones (monotone)")
    elif feats["voiced_frames"] < 40:
        score += 1.5
        reasons.append("too little voiced speech to analyse")
    if not np.isnan(feats["offset_decay_db"]) and feats["offset_decay_db"] < RULES["offset_decay_db"]:
        score += 2.0
        reasons.append(f"words ring out with a long reverberation tail ({feats['offset_decay_db']:.0f} dB/120 ms): playback through a speaker")
    return float(1 / (1 + np.exp(-score))), reasons


class AudioAnalyzer:
    """Rule-based by default; uses a trained classifier when a model file is present.

    `narrowband_model_path` is a model trained on real 16 kHz speech (ASVspoof 2019 LA, see
    realdata.py). It is used for audio at 16 kHz or below, the phone and app channel it was
    validated on; wideband browser capture keeps `model_path`."""

    def __init__(self, model_path: str | Path | None = None, narrowband_model_path: str | Path | None = None):
        import joblib

        self.model = joblib.load(model_path) if model_path and Path(model_path).exists() else None
        nb = narrowband_model_path
        self.narrowband = joblib.load(nb) if nb and Path(nb).exists() else None

    def analyze(self, x: np.ndarray, sr: int) -> AudioReport:
        feats = extract_features(x, sr)
        risk, reasons = rule_risk(feats)
        model, source = (self.narrowband, "16k model+rules") if sr <= ANALYSIS_SR and self.narrowband else (self.model, "model+rules")
        if model is None:
            return AudioReport(feats, risk, reasons, "rules")
        p = float(model.predict_proba(feature_vector(feats)[None, :])[0, 1])
        # keep rule reasons as explanation; blend so hard artifacts still count
        return AudioReport(feats, 0.7 * p + 0.3 * risk, reasons, source)


def voice_activity(x: np.ndarray, sr: int, hop: float = 0.010) -> np.ndarray:
    """Boolean per-hop voice activity, adaptive to the recording's own noise floor."""
    db = frame_db(x / (np.max(np.abs(x)) + EPS), sr, hop=hop)
    floor = np.percentile(db, 10)
    act = db > max(floor + 12.0, -50.0)
    # close short gaps (<120 ms) and drop blips (<80 ms)
    k = int(0.12 / hop)
    act = np.convolve(act, np.ones(k), "same") > 0
    act = np.convolve(act, np.ones(k), "same") >= k
    m = int(0.08 / hop)
    return np.convolve(act, np.ones(m), "same") >= m * 0.6


def speech_onsets(x: np.ndarray, sr: int, hop: float = 0.010, min_silence: float = 0.2) -> list[float]:
    act = voice_activity(x, sr, hop)
    onsets, silent_run = [], int(min_silence / hop)
    quiet = silent_run
    for i, a in enumerate(act):
        if a and quiet >= silent_run:
            onsets.append(i * hop)
        quiet = 0 if a else quiet + 1
    return onsets
