# KYC Liveness Guard

Liveness checks for video KYC and call-centre voice verification. It is meant to stop voice clones, face swaps, replayed sessions and injected camera streams.

A check that looks at one thing, such as a blink or a voice match, can be beaten by faking that one thing. This service checks four independent signals and combines them into a single decision: PASS, STEP_UP (send to review) or REJECT.

| Signal | What it measures | What it catches |
|---|---|---|
| Voice | microphone noise floor, spectral cut-off, pitch jitter, room reverberation | text-to-speech, voice clones, loudspeaker replays |
| Pulse | the faint colour pulse in the skin of the forehead and cheeks | printed photos, screen replays, face swaps |
| Lip sync | how closely mouth movement follows the loudness of the speech | dubbed video, a cloned voice over a real face |
| Timed challenge | speech and head turns must follow cues shown at random times | pre-recorded clips, virtual cameras |

The challenge needs no speech recognition. The server decides when each cue appears, and a recording made earlier cannot know that schedule. Challenge tokens are signed, expire after 3 minutes and can be used once.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                       # 33 tests
python -m liveness redteam --trials 20    # attack benchmark
python -m liveness serve                  # API and capture page on http://127.0.0.1:8001
streamlit run app.py                      # dashboard
```

The capture page records about 15 seconds of webcam video and microphone audio while it shows the cues, then sends both for verification. Without API keys the service runs in demo mode and only listens on localhost.

To run it in a container:

```bash
docker build -t kyc-liveness-guard .
python -m liveness.ops keygen             # prints an API key, its hash and a challenge key
docker run -p 8001:8001 -v liveness-data:/data \
  -e LIVENESS_API_KEYS='{"sha256:...": {"user": "onboarding", "role": "kyc_app"}}' \
  -e LIVENESS_CHALLENGE_KEY=... kyc-liveness-guard
```

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/v1/challenge` | – | `token`, `cues`, `duration` |
| POST | `/v1/verify` | `token`, `audio` (wav), `video` (webm/mp4, optional), `duration` | `decision`, `risk`, reasons per check |
| POST | `/v1/analyze/audio` | `audio` | voice risk and features |
| POST | `/v1/analyze/video` | `video` | pulse, heart rate, face coverage |
| GET | `/healthz`, `/readyz` | – | liveness and readiness |
| GET | `/metrics` | – | Prometheus metrics |

Send the API key as `x-api-key` or `Authorization: Bearer`. Roles: `kyc_app` issues challenges and verifies, `analyst` can also use `/v1/analyze/*`, `monitor` reads metrics, `admin` can do everything.

Example response:

```json
{
  "decision": "REJECT",
  "risk": 0.91,
  "checks": [
    {"name": "voice", "status": "fail", "risk": 0.94,
     "reasons": ["spectrum is cut off sharply at 7.2 kHz (19 dB wall): audio was produced at a lower rate and upsampled, or came through a narrowband headset"]},
    {"name": "pulse", "status": "fail", "risk": 0.97, "reasons": ["no clear skin pulse (SNR -4.6 dB)"]},
    {"name": "challenge", "status": "fail", "reasons": ["speech did not start on cue for: harbor, velvet"]}
  ]
}
```

## Results

| Test | Result |
|---|---|
| Synthetic attack benchmark (20 sessions per scenario) | all photo, replay, cloned-voice and virtual-camera attacks rejected; 15–20% of live face swaps accepted; no genuine users rejected |
| Real speech, ASVspoof 2019 LA, unseen attacks | voice equal error rate 15.0% (published baselines 8.1–9.6%) |
| Real webcam faces, MCD-rPPG | heart rate within 0.9 bpm of a finger sensor (median); 60% of photo attacks rejected, 15% of real faces rejected |
| Lip sync, GRID corpus | real speakers always pass, but so does audio of a different sentence |

Details, including what changed after testing on real data, are in [docs/evaluation.md](docs/evaluation.md).

## Configuration

| Variable | Purpose |
|---|---|
| `LIVENESS_ENV=production` | refuse to start without API keys, a challenge key and a database |
| `LIVENESS_API_KEYS` / `LIVENESS_KEYS_FILE` | hashed API keys with the user and role for each |
| `LIVENESS_CHALLENGE_KEY` | key that signs challenge tokens |
| `LIVENESS_DATABASE_URL` | `sqlite:///...` or `postgresql://...`; used tokens are stored here so they can't be replayed |
| `LIVENESS_MODEL_16K` | voice model for 16 kHz audio |

All settings, key rotation and backups: [docs/runbook.md](docs/runbook.md).

## Calibration

The thresholds are physical quantities and every camera and microphone shifts them. Before going live, record at least 50 genuine sessions through the real app and adjust:

- `liveness/audio.py`: `RULES` (noise floor, spectral edge, jitter, decay)
- `liveness/rppg.py`: `SNR_LIVE_DB`, `BPM_AGREEMENT`, `BG_CORR_MAX`
- `liveness/lipsync.py`: `SYNC_MIN_CORR`
- `liveness/pipeline.py`: signal weights and the PASS / REJECT cut-offs

Common causes of false rejections are Bluetooth headsets (their 16 kHz audio looks upsampled), browsers with noise suppression switched on, and dark skin in poor light, where the pulse signal is weak. Send these to STEP_UP rather than REJECT.

## Training the voice model

The rule engine works without a model. A trained classifier is used automatically when `models/voice_model.joblib` exists.

```bash
python -m liveness train --asvspoof ASVspoof2019.LA.cm.train.trn.txt LA/ASVspoof2019_LA_train/flac
python -m liveness realdata --train train.parquet --eval test-00000.parquet test-00001.parquet
```

## Project layout

```
liveness/
  audio.py      voice features and rules
  rppg.py       pulse extraction from video
  lipsync.py    speech loudness vs mouth movement
  challenge.py  signed timed challenges
  pipeline.py   combines the checks into a decision
  redteam.py    attack benchmark
  realdata.py   ASVspoof training and testing
  realface.py   pulse and lip-sync tests on public video
  api.py        FastAPI service
web/capture.html  browser capture page
app.py            Streamlit dashboard
```

## Limitations

- Face tracking uses OpenCV Haar cascades. A landmark model such as MediaPipe Face Mesh would give tighter regions and real head angles.
- Lip sync can't tell which words were said, so a cloned voice speaking in the same rhythm passes it.
- The pulse thresholds come from one public webcam dataset and must be re-tuned for other cameras, lighting and skin tones.
- Attacks that replace the camera feed are best stopped on the device with app and camera attestation.
- The code has not had an independent security review. Before use in real onboarding it also needs testing on the bank's own capture channel, RBI video-KYC and data protection approval, and the challenge key in an HSM or KMS.

See [SECURITY.md](SECURITY.md), [docs/threat-model.md](docs/threat-model.md) and [docs/model-card.md](docs/model-card.md).

## License

MIT. See [LICENSE](LICENSE).
