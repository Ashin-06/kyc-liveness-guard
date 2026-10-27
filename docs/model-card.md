# Model card: KYC Liveness Guard 1.0.0

Numbers come from `results/real_data.json`, `results/real_face.json` and the README. Nothing
here has been measured on a bank's own customers.

## What it is

Four checks fused into PASS / STEP_UP / REJECT for one recorded session:

| Check | Method | Learned? |
|---|---|---|
| Voice | 37 hand-crafted features (noise floor, spectral edge, pitch jitter, decay, MFCC stats) and rules; a gradient-boosting classifier on top when a model file is present | 16 kHz model trained on ASVspoof 2019 LA; rules otherwise |
| Pulse (rPPG) | POS algorithm on forehead and both cheeks, region agreement, background-pulse check | thresholds set on 10 MCD-rPPG subjects |
| Lip sync | lagged correlation of speech loudness with mouth motion | threshold only |
| Timed challenge | speech onsets and head turns must follow a random cue schedule | rules |

Fusion weights and cut-offs are in `liveness/pipeline.py` (pass below 0.35, reject above 0.60).

## Intended use

One signal in a video-KYC or voice-verification flow, alongside device attestation, face match
against an ID document and human review of STEP_UP sessions. Not intended as the only gate,
and not for identifying people (it never compares two people).

## Data

| Data | Used for | Licence |
|---|---|---|
| Synthetic voices (source-filter model) and RGB face traces, `liveness/synth.py` | the attack benchmark and the original 48 kHz voice model | — |
| ASVspoof 2019 LA: 25,380 train clips (attacks A01–A06); 15,831 eval clips from 2 of 9 eval shards (attacks A07–A19) | training and testing the 16 kHz voice model | ODC-By |
| MCD-rPPG: frontal webcam (640×480, 30 fps), 20 subjects, rest and after exercise, finger PPG reference | pulse accuracy and threshold calibration (10 tuning, 10 test subjects) | CC BY 4.0 |
| GRID: 200 three-second sentences, speakers s1 and s2 | lip-sync behaviour | CC BY 4.0 |

## Metrics

**Voice, ASVspoof 2019 LA eval (unseen attacks, about a fifth of the eval set):**

| Model | EER | AUC |
|---|---|---|
| 48 kHz model trained on synthetic voices | 50.5% (chance) | 0.50 |
| 16 kHz model trained on ASVspoof train (`models/voice_model_asvspoof.joblib`) | 15.0% | 0.93 |
| Published baselines on the full eval set: LFCC-GMM / CQCC-GMM | 8.1% / 9.6% | |

**Pulse, MCD-rPPG:** heart rate against finger PPG on 80 clips: median error 0.9 bpm, mean error
6.8 bpm (a few clips lock onto a harmonic), 76.2% within 5 bpm, 81.2% within 10 bpm.

Pulse check on the 10 held-out subjects (15-second clips), with the background check:

| Presentation | Pass | Review | Reject | AUC live vs attack |
|---|---|---|---|---|
| Live (40 clips) | 57.5% | 27.5% | 15% | |
| Photo | 5% | 35% | 60% | 0.83 |
| Photo under a white lamp flickering at 72 bpm | 30% | 5% | 65% | 0.75 |
| Photo under a green-tinted flicker at 72 bpm | 0% | 0% | 100% | 0.90 |

Before recalibration, the shipped thresholds rejected 67.5% of live clips on held-out subjects.
Without the background check the green-tinted flicker photo passed 100% of the time.

**Lip sync, GRID:** passes its own audio 100% of the time; passes another sentence by the same
speaker 100% of the time (AUC 0.54, chance); passes its own audio 0.5 s late 76.5% of the time
(AUC 0.86).

**Synthetic attack benchmark** (`python -m liveness redteam --trials 20`): see the README table.
The cue schedule is drawn at random on every run, so counts move between runs; the README lists
the range seen. It is a regression suite, not a field accuracy claim.

## Known failure modes

- The synthetic voice benchmark says nothing about real deepfakes: the model trained on it is at
  chance on real speech. The 48 kHz browser path in the container uses rules only and has no
  real-data measurement at all.
- Hand-crafted voice features trail the published spectral baselines (15.0% vs 8.1% EER).
- On a 15-second webcam clip the pulse is weak evidence: it holds back about a quarter of real
  people and lets some photos through. It cannot be the gate on its own. The live reject rate on
  held-out subjects (15%) is above the 10% targeted on the tuning subjects.
- Lip sync compares loudness with mouth movement, so a cloned voice saying other words in the
  same rhythm passes it.
- Bluetooth headsets (16 kHz links), browsers with noise suppression on, and very dark skin in
  poor light raise false rejects.
- Face tracking uses Haar cascades; yaw is coarse.

## Caveats

- The pulse thresholds come from one public webcam set (mostly Russian adults, indoor light).
  Skin tone, lighting and camera all shift them. Measure per channel and per population.
- The voice evaluation uses 2 of 9 eval shards; the published baselines use the full set.
- The synthetic benchmark's faces are RGB traces, not video; its voices are a source-filter model.
- Re-tune the thresholds listed in the README (Calibration) on 50+ genuine sessions from your own
  capture channel before going live.
