"""Synthetic bona fide sessions and presentation attacks for red-teaming.

Voices are produced with a source-filter model (glottal pulse train with jitter and
shimmer through vowel formant resonators). Faces are mean-RGB traces per skin region
with a pulse component. Each attack reproduces the artifact a real attack leaves,
not a toy label.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

VOWELS = [(730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240), (530, 1840, 2480),
          (570, 840, 2410), (440, 1020, 2240), (660, 1720, 2410)]
BANDWIDTHS = (80, 110, 150)


def _resonate(x: np.ndarray, f: float, bw: float, sr: int) -> np.ndarray:
    r = np.exp(-np.pi * bw / sr)
    a = [1, -2 * r * np.cos(2 * np.pi * f / sr), r * r]
    return signal.lfilter([1 - r], a, x)


def _segments(rng, dur: float, word_times: list[float] | None) -> list[tuple[float, float]]:
    if word_times is not None:
        return [(t, min(t + rng.uniform(0.45, 0.75), dur)) for t in word_times]
    segs, t = [], rng.uniform(0.2, 0.5)
    while t < dur - 0.3:
        length = rng.uniform(0.15, 0.35)
        segs.append((t, min(t + length, dur)))
        t += length + rng.uniform(0.08, 0.35)
    return segs


def voice(
    sr: int = 48000,
    dur: float = 4.0,
    seed: int = 0,
    f0: float | None = None,
    jitter: float = 0.012,
    shimmer: float = 0.06,
    intonation: float = 0.08,
    noise_db: float | None = -58.0,
    word_times: list[float] | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sr * dur)
    f0 = f0 or rng.uniform(95, 220)
    phase = rng.uniform(0, 6)

    exc, t = np.zeros(n), 0.0
    while t < dur:
        drift = 1 + intonation * np.sin(2 * np.pi * 0.35 * t + phase) + 0.4 * intonation * np.sin(2 * np.pi * 1.7 * t)
        idx = int(t * sr)
        exc[idx] = 1 + shimmer * rng.standard_normal()
        t += (1 / (f0 * drift)) * (1 + jitter * rng.standard_normal())
    g = signal.lfilter([1], [1, -0.96], signal.lfilter([1], [1, -0.96], exc))
    src = np.diff(g, prepend=0.0) + 0.015 * rng.standard_normal(n)  # lip radiation + aspiration

    out = np.zeros(n)
    for s, e in _segments(rng, dur, word_times):
        i0, i1 = int(s * sr), int(e * sr)
        seg = src[i0:i1]
        for f, bw in zip(VOWELS[rng.integers(len(VOWELS))], BANDWIDTHS):
            seg = _resonate(seg, f * rng.uniform(0.95, 1.05), bw, sr)
        seg = seg / (np.max(np.abs(seg)) + 1e-12)
        out[i0:i1] += seg * _ramp(len(seg), sr)
        if rng.random() < 0.6:  # fricative ("s", "sh", "f") before the vowel
            m = int(rng.uniform(0.05, 0.11) * sr)
            j0 = max(i0 - m, 0)
            hi = min(rng.uniform(9000, 12000), 0.45 * sr)
            fric = signal.sosfilt(signal.butter(2, [rng.uniform(2500, 4500), hi], btype="band", fs=sr, output="sos"), rng.standard_normal(i0 - j0))
            out[j0:i0] += 0.35 * fric / (np.max(np.abs(fric)) + 1e-12) * _ramp(i0 - j0, sr)
    out = 0.5 * out / (np.max(np.abs(out)) + 1e-12)
    if noise_db is not None:
        out += room_noise(n, sr, noise_db, rng)
    return out.astype(np.float32)


def _ramp(n: int, sr: int) -> np.ndarray:
    r = min(int(0.03 * sr), n // 2)
    env = np.ones(n)
    if r:
        env[:r] = np.sin(np.linspace(0, np.pi / 2, r)) ** 2
        env[n - r :] = env[:r][::-1]
    return env


def room_noise(n: int, sr: int, level_db: float, rng) -> np.ndarray:
    white = rng.standard_normal(n)
    pink = signal.lfilter([0.049922035, -0.095993537, 0.050612699, -0.004408786], [1, -2.494956002, 2.017265875, -0.522189400], white)
    hum = signal.lfilter(*signal.butter(2, 120 / (sr / 2)), rng.standard_normal(n)) * 4
    mix = 0.6 * white / white.std() + 0.4 * pink / pink.std() + 0.3 * hum / (hum.std() + 1e-12)
    return mix / mix.std() * 10 ** (level_db / 20)


def tts(sr: int = 48000, dur: float = 4.0, seed: int = 0, native_sr: int = 16000, add_noise: bool = False, word_times=None) -> np.ndarray:
    """Offline text-to-speech: stable pitch, flat intonation, generated at a low rate
    and upsampled. With add_noise the attacker masks the digital silence."""
    x = voice(native_sr, dur, seed, jitter=0.0008, shimmer=0.008, intonation=0.015, noise_db=None, word_times=word_times)
    x = signal.resample_poly(x, sr, native_sr).astype(np.float64)[: int(sr * dur)]
    if add_noise:
        x += room_noise(len(x), sr, -60.0, np.random.default_rng(seed + 99))
    return x.astype(np.float32)


def replay(sr: int = 48000, dur: float = 4.0, seed: int = 0, word_times=None) -> np.ndarray:
    """A genuine recording played through a phone speaker into the microphone."""
    rng = np.random.default_rng(seed + 7)
    x = voice(sr, dur, seed, word_times=word_times).astype(np.float64)
    x = signal.sosfilt(signal.butter(4, [280, 7000], btype="band", fs=sr, output="sos"), x)
    t = np.arange(int(0.45 * sr)) / sr
    ir = rng.standard_normal(len(t)) * np.exp(-6.9 * t / 0.45)  # RT60 ~ 0.45 s
    ir[0] = 4.0
    x = signal.fftconvolve(x, ir)[: len(x)]
    x = 0.5 * x / np.max(np.abs(x))
    return (x + room_noise(len(x), sr, -56.0, rng)).astype(np.float32)


# ------------------------------------------------------------------ faces

SKIN = np.array([182.0, 128.0, 108.0])
PULSE_WEIGHTS = np.array([0.33, 0.77, 0.53])  # relative pulsatile strength per channel


def face_traces(kind: str = "live", fps: float = 30.0, dur: float = 10.0, seed: int = 0, bpm: float | None = None) -> dict[str, np.ndarray]:
    """kind: live | photo | screen | deepfake"""
    rng = np.random.default_rng(seed)
    n = int(fps * dur)
    t = np.arange(n) / fps
    bpm = bpm or rng.uniform(58, 95)
    hr = bpm / 60 * (1 + 0.03 * np.sin(2 * np.pi * 0.1 * t))  # heart-rate variability
    pulse = np.sin(2 * np.pi * np.cumsum(hr) / fps) + 0.35 * np.sin(4 * np.pi * np.cumsum(hr) / fps + 0.8)
    light = 1 + 0.004 * np.cumsum(rng.standard_normal(n)) / np.sqrt(n) + 0.002 * np.sin(2 * np.pi * 0.07 * t)
    out = {}
    for i, region in enumerate(("forehead", "cheek_l", "cheek_r")):
        base = SKIN * rng.uniform(0.95, 1.05)
        amp = {"live": 0.0035, "photo": 0.0, "screen": 0.0009, "deepfake": 0.0}[kind]
        p = pulse
        if kind == "deepfake":
            # generator re-renders texture per frame: region-local pseudo-periodic flicker
            f = rng.uniform(0.8, 3.0)
            p = np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) + 0.8 * rng.standard_normal(n)
            amp = 0.0012
        trace = base[None, :] * light[:, None] * (1 + amp * PULSE_WEIGHTS[None, :] * p[:, None])
        if kind == "screen":
            trace *= 1 + 0.004 * np.sin(2 * np.pi * (fps / 2 - 0.9) * t)[:, None]  # aliased refresh beat
        out[region] = trace + rng.normal(0, 0.12, size=(n, 3))  # sensor noise
    return out


def mouth_from_audio(x: np.ndarray, sr: int, fps: float, n: int, seed: int = 0, lag_s: float = 0.04) -> np.ndarray:
    from .lipsync import envelope_per_frame

    rng = np.random.default_rng(seed)
    env = envelope_per_frame(x, sr, fps, n)
    shift = int(lag_s * fps)
    m = np.roll(env, shift)
    m[:shift] = m[shift] if shift < n else 0
    m = np.abs(np.diff(m, prepend=m[0])) * 6 + m  # opening and closing both move pixels
    return m / (m.max() + 1e-9) + rng.normal(0, 0.08, n)
