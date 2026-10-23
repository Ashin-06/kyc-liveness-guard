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
bona_fide                     19     1     0   0.10  BPCER 5.0%
photo_tts                      0     0    20   0.89  APCER 0.0%
screen_replay                  0     0    20   0.20  APCER 0.0%
faceswap_clone                 2    10     8   0.50  APCER 10.0%
clone_noise_masked             0     2    18   0.64  APCER 0.0%
virtual_cam_old_session        0     0    20   0.09  APCER 0.0%
```

APCER and BPCER follow ISO/IEC 30107-3. APCER is the share of attacks accepted; BPCER is the share of genuine users not accepted.

Two attackers in the benchmark adapt to the defence:
- **`clone_noise_masked`** adds fake room noise to hide digital silence. The spectral edge and pitch jitter still catch it.
- **`faceswap_clone`** follows the cues live. Timing alone can't catch it, so the missing pulse and the voice artifacts have to. With the pulse thresholds calibrated on real webcams (below), 2 of 20 get through and 10 are held for review. Before that calibration all 20 were stopped, but the same settings rejected two thirds of real people.

The benchmark is synthetic: voices come from a source-filter model and faces are RGB traces. Treat these numbers as a regression suite, not a field accuracy claim. Before deploying, measure on your own capture channel (see Calibration).

## On real speech: ASVspoof 2019

The voice check was then measured on real speech and real attacks with [ASVspoof 2019 LA](https://datashare.ed.ac.uk/handle/10283/3336) (ODC-By). Training uses the 25,380-clip train split, whose spoofs come from attacks A01–A06. Testing uses 15,831 clips (2 of the 9 eval shards, about a fifth) whose spoofs come from attacks A07–A19, most of which never appear in training. That is the position a bank is in when a new cloning tool comes out. Command: `python -m liveness realdata`; results in `results/real_data.json`.

| Voice model | Equal error rate | AUC |
|---|---|---|
| Shipped model, trained on the synthetic voices above | **50.5%** (chance) | 0.50 |
| Same 37 features, trained on ASVspoof (`models/voice_model_asvspoof.joblib`) | **15.0%** | 0.93 |
| *Published ASVspoof baselines, full eval set: LFCC-GMM / CQCC-GMM* | *8.1% / 9.6%* | |

What this means:
- **The synthetic voice benchmark says nothing about real deepfakes.** The model trained on synthetic voices is at chance on real ones.
- **Trained on real speech, the hand-crafted features catch most unseen attacks**, but still trail the published spectral baselines. A learned front end (for example LFCC or SSL embeddings) is the next step.
- **The real-speech model is used where it was validated:** `AudioAnalyzer` switches to it for audio at 16 kHz or below, the phone and app channel (`LIVENESS_MODEL_16K`).
- **The 48 kHz browser capture keeps the original model.** Before relying on it, calibrate it on recordings from your own channel.

## On real faces: MCD-rPPG and GRID

`python -m liveness realface`; results in `results/real_face.json`.

**Pulse, [MCD-rPPG](https://huggingface.co/datasets/kyegorov/mcd_rppg)** (Egorov et al., ACM MM 2025; CC BY 4.0): frontal webcam recordings (640×480, 30 fps) of 20 people at rest and after exercise, with a finger PPG synchronised to every frame. It was used for pulse estimation only, which is what the dataset is published for. Two 15-second windows per recording were scored as live; that is about the length of a KYC capture. Three photo attacks were built from a frame of each recording: the still frame with sensor noise, the same under a lamp flickering at 72 bpm, and under a green-tinted flicker aimed at the pulse method.

- **Heart rate is right.** On 80 clips, the median error against the finger PPG is 0.9 bpm, and 76% of clips are within 5 bpm. The mean error, 6.8 bpm, comes from a few clips where the estimate locked onto a harmonic.
- **The shipped thresholds rejected most real people.** Real webcam faces have a pulse SNR around −0.8 dB, against the 1.0 dB the thresholds expected. The forehead and cheeks disagree by a median 12 bpm, because the cheek patches are small at this resolution. With the old settings, 67.5% of live clips on held-out subjects failed.
- **What changed.** `SNR_LIVE_DB` and `BPM_AGREEMENT` were reset on 10 subjects so that at most 10% of their live clips fail. A new check rejects a pulse that also shows up in the background (the two top corners of the frame): blood only colours skin, while a lamp or screen colours the room. All numbers below are on the other 10 subjects.

| Held-out subjects | Pass | Review | Reject |
|---|---|---|---|
| Live faces, old thresholds | 12.5% | 20% | 67.5% |
| Live faces, recalibrated | 57.5% | 27.5% | 15% |
| Photo | 5% | 35% | 60% |
| Photo under a white lamp flickering at 72 bpm | 30% | 5% | 65% |
| Photo under a green-tinted flicker at 72 bpm | 100% → **0%** with the background check | 0% | 0% → **100%** |

On a 15-second webcam clip the pulse is weak evidence: it stops most photos but also holds back a quarter of real people. It cannot be the gate on its own. Against a printed photo, the head-turn cues in the timed challenge carry most of the weight. The live reject rate on held-out subjects (15%) is above the 10% that was targeted on the tuning subjects.

**Lip sync, [GRID](https://zenodo.org/records/3625687)** (Cooke et al., 2006; CC BY 4.0): 200 three-second sentences from two speakers, filmed face-on. Each clip's mouth movement was scored against its own audio, against another sentence by the same speaker (a dub or separate audio over a real face), and against its own audio 0.5 s late (a slow voice changer).

| Audio | Passes the sync check | AUC against genuine |
|---|---|---|
| Its own | 100% | |
| Another sentence, same speaker | **100%** | 0.54 (chance) |
| Its own, 0.5 s late | 76.5% | 0.86 |

- **The lip-sync check never rejected a real speaker, but it cannot tell which words were said.** Every GRID sentence has the same six-word rhythm, and the check compares mouth movement with loudness only. A cloned voice saying other words in the same cadence passes it.
- **It catches gross mismatches only:** silence against a moving mouth, speech against a still one, or audio far out of step. Matching the actual words needs a visual speech model (lip reading), which is out of scope here.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                       # 10 tests
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

# or the Hugging Face parquet mirrors of ASVspoof 2019 LA (train split + eval shards)
python -m liveness realdata --train train-00000-of-00001.parquet --eval test-00000-of-00009.parquet test-00001-of-00009.parquet
```

The training script reports AUC and EER on a held-out split.

## Calibration

The thresholds are physical quantities, and every capture channel shifts them. Before going live, record 50+ genuine sessions through your real app and adjust:

- `liveness/audio.py` → `RULES` (noise floor, spectral edge, jitter, decay)
- `liveness/rppg.py` → `SNR_LIVE_DB`, `BPM_AGREEMENT`, `BG_CORR_MAX`, `PROFILE_CASCADE_LOOKS` (set on MCD-rPPG webcams; re-tune for your camera)
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
  realdata.py   train/test on ASVspoof 2019 LA parquet, writes the 16 kHz model
  realface.py   pulse on MCD-rPPG webcams, lip sync on GRID
  api.py        FastAPI service
web/capture.html  browser capture flow
app.py            Streamlit dashboard
```

## Limits

- Face tracking uses Haar cascades so it runs anywhere. For production, swap in a landmark model (MediaPipe Face Mesh) for tighter ROIs and real yaw angles. `extract_video` is the only function that changes.
- Lip sync compares loudness with mouth movement, so a cloned voice saying other words in the same rhythm passes it (see On real faces).
- The pulse thresholds come from one public webcam set (mostly Russian adults, indoor light). Skin tone, lighting and camera all shift them.
- A strong real-time voice converter running at 48 kHz with natural jitter weakens the voice signal. Pulse and lip sync still have to be beaten at the same time.
- Injection attacks that replace the camera feed are best stopped at the device: attest the app and camera. The timed challenge raises the bar but is not a substitute for that.
