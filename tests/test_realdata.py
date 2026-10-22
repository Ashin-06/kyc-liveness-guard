import io

import numpy as np
import soundfile as sf

from liveness.audio import AudioAnalyzer
from liveness.realdata import features_from_flac


class _Fixed:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, x):
        return np.c_[1 - np.full(len(x), self.p), np.full(len(x), self.p)]


def test_flac_bytes_become_a_feature_vector_and_garbage_is_skipped():
    sr = 16000
    t = np.arange(sr * 2) / sr
    x = (0.3 * np.sin(2 * np.pi * 180 * t) * (1 + 0.1 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="FLAC")
    v = features_from_flac(buf.getvalue())
    assert v is not None and np.isfinite(v).all()
    assert features_from_flac(b"not audio") is None


def test_narrowband_audio_uses_the_16k_model():
    a = AudioAnalyzer()
    a.model, a.narrowband = _Fixed(0.0), _Fixed(1.0)
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.1, 16000 * 2).astype(np.float32)
    assert a.analyze(x, 16000).source == "16k model+rules"
    assert a.analyze(np.repeat(x, 3), 48000).source == "model+rules"
