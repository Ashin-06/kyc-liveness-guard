# KYC Liveness Guard

Liveness detection for video KYC and call-centre voice verification. It is built to stop the attacks that now beat single-signal checks: voice clones, face swaps, replayed sessions and injected camera streams.

Most liveness products check one thing, like a blink or a voice match. A deepfake only needs to fake that one thing. This project checks four independent signals, each tied to an artifact attackers find hard to remove, and fuses them into a decision.

| Signal | What it measures | What it catches |
|---|---|---|
| **Voice** | Microphone noise floor, spectral brick-wall edge, pitch jitter, reverberation tail after words | TTS and voice clones (offline or upsampled), loudspeaker replays |
| **Pulse (rPPG)** | Skin colour pulse from forehead and both cheeks, using the POS algorithm | Printed photos, screen replays, face swaps (the pulse is missing or the regions disagree) |
| **Lip sync** | Lagged correlation between the speech envelope and mouth motion | Dubbed video, a cloned voice over a real face |
| **Timed challenge** | Speech onsets and head turns must land right after randomly timed cues | Pre-recorded clips, injected virtual-camera streams |

The challenge needs no speech recognition. The server picks *when* each cue appears, and a recording made earlier cannot know that schedule. Tokens are HMAC-signed, expire after 3 minutes and can be used only once.

## Results on the attack benchmark

`python -m liveness redteam --trials 20`:

```
scenario                    pass  step   rej   risk  metric
----------------------------------------------------------------
bona_fide                     20     0     0   0.11  BPCER 0.0%
photo_tts                      0     0    20   0.94  APCER 0.0%
screen_replay                  0     0    20   0.27  APCER 0.0%
faceswap_clone                 0     1    19   0.68  APCER 0.0%
clone_noise_masked             0     1    19   0.65  APCER 0.0%
virtual_cam_old_session        0     0    20   0.08  APCER 0.0%
```

APCER and BPCER follow ISO/IEC 30107-3. APCER is the share of attacks accepted; BPCER is the share of genuine users not accepted.

Two attackers in the benchmark adapt to the defence:
- **`clone_noise_masked`** adds fake room noise to hide digital silence. The spectral edge and pitch jitter still catch it.
- **`faceswap_clone`** follows the cues live. Timing alone can't catch it, so the missing pulse and the voice artifacts do.

The benchmark is synthetic: voices come from a source-filter model and faces are RGB traces. Treat these numbers as a regression suite, not a field accuracy claim. Before deploying, measure on your own capture channel (see Calibration).

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                       # 7 tests
python -m liveness redteam --trials 20    # attack benchmark
python -m liveness serve --port 8001      # API + capture page
```

Open http://127.0.0.1:8001 for the browser capture flow. It records about 15 seconds of webcam video and raw microphone audio while it shows the cues, then posts both for verification.

Dashboard (benchmark, upload a recording, pulse explorer):

```bash
streamlit run app.py
```

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/v1/challenge` | – | `token`, `cues[]` (`t`, `kind`, `value`), `duration` |
| POST | `/v1/verify` | multipart: `token`, `audio` (wav), `video` (webm/mp4, optional), `duration` | `decision` (PASS / STEP_UP / REJECT), `risk`, per-check reasons |
| POST | `/v1/analyze/audio` | multipart: `audio` | voice risk and features |
| POST | `/v1/analyze/video` | multipart: `video` | pulse, heart rate, face coverage |
| GET | `/healthz` | – | status, active voice engine |

Example response:

```json
{
  "decision": "REJECT",
  "risk": 0.91,
  "checks": [
    {"name": "voice", "status": "fail", "risk": 0.94,
     "reasons": ["spectrum is cut off sharply at 7.2 kHz (19 dB wall): audio was produced at a lower rate and upsampled, or came through a narrowband headset",
                 "pitch jitter 0.132% is below the natural voice range"]},
    {"name": "pulse", "status": "fail", "risk": 0.97, "reasons": ["no clear skin pulse (SNR -4.6 dB)"]},
    {"name": "challenge", "status": "fail", "reasons": ["speech did not start on cue for: harbor, velvet"]}
  ]
}
```

Set `LIVENESS_SECRET` (hex) so tokens survive restarts and work across replicas. Single-use tracking is in memory; with more than one replica, move the used-token set to Redis.

## Training the voice model

The rule engine works without any model. A trained classifier is used automatically when `models/voice_model.joblib` exists.

```bash
# ASVspoof 2019 LA (or 2021 LA/DF) protocol + flac folder
python -m liveness train --asvspoof ASVspoof2019.LA.cm.train.trn.txt LA/ASVspoof2019_LA_train/flac

# or a synthetic corpus to exercise the pipeline
python -m liveness train --synthetic 600
```

The training script reports AUC and EER on a held-out split.

## Calibration

The thresholds are physical quantities, and every capture channel shifts them. Before going live, record 50+ genuine sessions through your real app and adjust:

- `liveness/audio.py` → `RULES` (noise floor, spectral edge, jitter, decay)
- `liveness/rppg.py` → `SNR_LIVE_DB`, `BPM_AGREEMENT`, `PROFILE_CASCADE_LOOKS`
- `liveness/lipsync.py` → `SYNC_MIN_CORR`
- `liveness/pipeline.py` → fusion weights and the PASS / REJECT cut-offs

Known false-positive sources:
- **Bluetooth headsets** (16 kHz mic links show a spectral edge).
- **Browsers with noise suppression left on.** The capture page turns it off on purpose.
- **Very dark skin under poor light.** rPPG SNR drops, so route these sessions to STEP_UP rather than REJECT.

## Layout

```
liveness/
  audio.py      voice features, rule engine, VAD/onsets
  rppg.py       POS pulse extraction, multi-region check, video -> traces (OpenCV)
  lipsync.py    audio envelope vs mouth motion
  challenge.py  signed single-use timed challenges, cue matching
  pipeline.py   fusion into PASS / STEP_UP / REJECT
  synth.py      synthetic genuine sessions and attacks
  redteam.py    benchmark with APCER/BPCER
  train.py      classifier training (ASVspoof or synthetic)
  api.py        FastAPI service
web/capture.html  browser capture flow
app.py            Streamlit dashboard
```

## Limits

- Face tracking uses Haar cascades so it runs anywhere. For production, swap in a landmark model (MediaPipe Face Mesh) for tighter ROIs and real yaw angles. `extract_video` is the only function that changes.
- A strong real-time voice converter running at 48 kHz with natural jitter weakens the voice signal. Pulse and lip sync still have to be beaten at the same time.
- Injection attacks that replace the camera feed are best stopped at the device: attest the app and camera. The timed challenge raises the bar but is not a substitute for that.
