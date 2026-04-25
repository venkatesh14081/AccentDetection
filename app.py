import streamlit as st
import tempfile
from scipy.io.wavfile import write
import joblib
import numpy as np
import matplotlib.pyplot as plt
import whisper
from fuzzywuzzy import fuzz
import os
import torch
import soundfile as sf

from sentences import get_random_sentence
from accent_predictor import extract_features
from pronunciation_feedback import pronunciation_feedback

# ===== SAFE IMPORT =====
try:
    import sounddevice as sd
    LOCAL_MIC = True
except:
    LOCAL_MIC = False

# ===== WEBRTC =====
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    WEBRTC = True
except:
    WEBRTC = False

torch.set_num_threads(1)

# ================= SILENCE CHECK =================
def is_audio_too_quiet(audio_path, threshold=0.01):
    try:
        y, sr = sf.read(audio_path)
        if len(y.shape) > 1:
            y = y[:, 0]
        rms = np.sqrt(np.mean(y**2))
        return rms < threshold
    except:
        return False

# ================= SESSION =================
if "audio" not in st.session_state:
    st.session_state.audio = None
if "sentence_data" not in st.session_state:
    st.session_state.sentence_data = None
if "accent" not in st.session_state:
    st.session_state.accent = None

# ================= CONFIG =================
st.set_page_config(page_title="Accent Coach", page_icon="🎧", layout="wide")

# ================= UI =================
st.markdown("""
<style>
html, body, [data-testid="stApp"] {background-color:#000;color:#f8fafc;}
[data-testid="stSidebar"] {background-color:#020617;}
.card {background:#0f172a;border-radius:18px;padding:20px;margin-bottom:20px;}
.metric {font-size:34px;font-weight:900;color:#22c55e;}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🎧 Accent Coach")

# ================= LOAD =================
@st.cache_resource
def load_models():
    clf = joblib.load("model/accent_classifier.pkl")
    le = joblib.load("model/label_encoder.pkl")
    whisper_model = whisper.load_model("tiny")
    return clf, le, whisper_model

classifier, label_encoder, whisper_model = load_models()

# ================= SIDEBAR =================
with st.sidebar:
    st.header("🎯 Settings")
    target_accent = st.selectbox(
        "Target Accent",
        ["american", "british", "australian", "canadian"]
    )

# ================= SENTENCE =================
if st.session_state.sentence_data is None or st.session_state.accent != target_accent:
    st.session_state.sentence_data = get_random_sentence(target_accent)
    st.session_state.accent = target_accent

sentence, audio_file = st.session_state.sentence_data

st.markdown("<div class='card'>", unsafe_allow_html=True)

c1, c2 = st.columns([8,1])
with c1:
    st.subheader("📘 Practice Sentence")
with c2:
    if st.button("🔄 New"):
        st.session_state.sentence_data = get_random_sentence(target_accent)
        st.rerun()

st.info(sentence)

if audio_file:
    path = f"reference_audio/{target_accent}/{audio_file}"
    if os.path.exists(path):
        st.audio(path)

st.markdown("</div>", unsafe_allow_html=True)

# ================= INPUT =================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🎙️ Your Voice")

col1, col2 = st.columns(2)

with col1:
    mode = st.radio("Input Method", ["Microphone", "Upload WAV"])

with col2:

    if mode == "Microphone":

        # LOCAL
        if LOCAL_MIC:
            if st.button("🎤 Record (5 sec)"):
                fs = 16000
                rec = sd.rec(int(5 * fs), samplerate=fs, channels=1)
                sd.wait()

                temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                write(temp.name, fs, rec.squeeze())

                st.session_state.audio = temp.name
                st.success("Recorded")

        # CLOUD
        else:
            st.info("🎤 Using browser microphone")

            webrtc_ctx = webrtc_streamer(
                key="webrtc_audio",
                mode=WebRtcMode.SENDONLY,
                media_stream_constraints={"audio": True, "video": False},
            )

            if st.button("Stop & Save"):

                frames = []

                if webrtc_ctx.audio_receiver:
                    try:
                        while True:
                            frame = webrtc_ctx.audio_receiver.get_frame(timeout=1)
                            if frame is None:
                                break
                            frames.append(frame.to_ndarray())
                    except:
                        pass

                if frames:
                    audio = np.concatenate(frames, axis=0)

                    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    write(temp.name, 16000, audio.astype(np.float32))

                    st.session_state.audio = temp.name
                    st.success("Recorded (browser)")
                else:
                    st.warning("No audio captured. Please try again.")

    else:
        file = st.file_uploader("Upload WAV", type=["wav"])
        if file:
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp.write(file.read())
            st.session_state.audio = temp.name
            st.success("Uploaded")

st.markdown("</div>", unsafe_allow_html=True)

# ================= ANALYSIS =================
analyze_btn = st.button("🚀 Analyze")

if st.session_state.audio and analyze_btn:

    if is_audio_too_quiet(st.session_state.audio):
        st.warning("🔊 Please speak louder.")
        st.stop()

    audio, sr = sf.read(st.session_state.audio)
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    result = whisper_model.transcribe(audio)

    spoken = result["text"].lower().strip()
    similarity = fuzz.ratio(sentence.lower(), spoken)

    if similarity < 80:
        st.error("❌ Sentence mismatch")
        st.write(spoken)
        st.stop()

    st.success(f"Sentence Accuracy: {similarity:.1f}%")

    features = extract_features(st.session_state.audio).reshape(1, -1)
    probs = classifier.predict_proba(features)[0] * 100

    idx = list(label_encoder.classes_).index(target_accent)

    st.markdown(f"<div class='metric'>{probs[idx]:.1f}%</div>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(2.4, 1.6))
    bars = ax.bar(label_encoder.classes_, probs)
    bars[idx].set_color("#22c55e")
    ax.set_ylim(0, 100)

    st.pyplot(fig)

    for tip in pronunciation_feedback(target_accent, spoken, sentence):
        st.markdown(f"• {tip}")