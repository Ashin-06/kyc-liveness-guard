"""Analyst dashboard: red-team results and ad-hoc analysis of uploaded recordings.

    streamlit run app.py
"""
import os
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import spectrogram

from liveness import synth
from liveness.audio import AudioAnalyzer, load_audio
from liveness.pipeline import evaluate
from liveness.redteam import SCENARIOS, run
from liveness.rppg import analyze_traces, extract_video, spectrum

st.set_page_config(page_title="KYC Liveness Guard", layout="wide")
st.title("KYC Liveness Guard")

tab_bench, tab_inspect, tab_pulse = st.tabs(["Attack benchmark", "Inspect a recording", "Pulse explorer"])

with tab_bench:
    st.caption("Each scenario reproduces the artifacts a real presentation attack leaves behind. "
               "APCER = attacks accepted, BPCER = genuine users not accepted (ISO/IEC 30107-3).")
    trials = st.slider("Sessions per scenario", 5, 60, 20, 5)
    if st.button("Run benchmark", type="primary"):
        with st.spinner("Simulating sessions…"):
            st.session_state["bench"] = run(trials)
    res = st.session_state.get("bench")
    if res:
        rows = [{
            "scenario": k, "what it is": v["description"], "pass": v["pass"], "step-up": v["step_up"],
            "reject": v["reject"], "mean risk": round(v["mean_risk"], 3),
            "metric": f"BPCER {v['bpcer']:.1%}" if "bpcer" in v else f"APCER {v['apcer']:.1%}",
            "caught by": ", ".join(f"{c} ({n})" for c, n in sorted(v["triggered_checks"].items(), key=lambda x: -x[1])),
        } for k, v in res.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        fig = go.Figure()
        for k, v in res.items():
            fig.add_trace(go.Box(y=v["risks"], name=k, boxpoints="all", jitter=0.4))
        fig.add_hline(y=0.35, line_dash="dot", annotation_text="pass below")
        fig.add_hline(y=0.60, line_dash="dot", annotation_text="reject above")
        fig.update_layout(height=420, yaxis_title="fused risk", showlegend=False, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")

with tab_inspect:
    c1, c2 = st.columns(2)
    audio_file = c1.file_uploader("Voice recording (wav/flac/ogg)", type=["wav", "flac", "ogg"])
    video_file = c2.file_uploader("Face video (mp4/webm/mov)", type=["mp4", "webm", "mov", "avi"])
    if audio_file or video_file:
        audio = load_audio(audio_file.getvalue()) if audio_file else None
        video = None
        if video_file:
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(video_file.name)[1], delete=False) as tmp:
                tmp.write(video_file.getvalue())
            try:
                video = extract_video(tmp.name)
            except ValueError as e:
                st.error(str(e))
            finally:
                os.unlink(tmp.name)
        d = evaluate(audio, video, analyzer=AudioAnalyzer("models/voice_model.joblib"))
        color = {"PASS": "green", "STEP_UP": "orange", "REJECT": "red"}[d.decision]
        st.markdown(f"### :{color}[{d.decision}] · risk {d.risk:.2f}")
        for c in d.as_dict()["checks"]:
            with st.expander(f"{c['name']} — {c['status']} (risk {c['risk']})", expanded=c["status"] != "pass"):
                for r in c["reasons"]:
                    st.write("•", r)
                st.json(c["data"])
        if audio:
            x, sr = audio
            f, t, S = spectrogram(x, sr, nperseg=1024, noverlap=768)
            fig = go.Figure(go.Heatmap(z=10 * np.log10(S + 1e-12), x=t, y=f, colorscale="Viridis", showscale=False))
            fig.update_layout(height=320, title="Spectrogram — look for a hard ceiling (upsampled audio) or empty gaps (digital silence)",
                              yaxis_title="Hz", xaxis_title="s", margin=dict(t=40))
            st.plotly_chart(fig, width="stretch")

with tab_pulse:
    st.caption("rPPG pulse recovered from synthetic face traces. A live face shows one clean peak shared by all regions.")
    kind = st.radio("Presentation", ["live", "photo", "screen", "deepfake"], horizontal=True)
    seed = st.number_input("Seed", 0, 9999, 3)
    traces = synth.face_traces(kind, seed=int(seed))
    pr = analyze_traces(traces, 30.0)
    m1, m2, m3 = st.columns(3)
    m1.metric("Heart rate", f"{pr.bpm:.0f} bpm")
    m2.metric("Pulse SNR", f"{pr.snr_db:.1f} dB")
    m3.metric("Region spread", f"{pr.bpm_spread:.0f} bpm")
    f, p = spectrum(pr.pulse, 30.0)
    band = (f > 0.5) & (f < 4)
    fig = go.Figure(go.Scatter(x=f[band] * 60, y=p[band], fill="tozeroy"))
    fig.update_layout(height=300, xaxis_title="bpm", yaxis_title="power", margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    for r in pr.reasons:
        st.warning(r)
