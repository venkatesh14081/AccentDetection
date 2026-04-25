import streamlit as st
import tempfile
import sounddevice as sd
from scipy.io.wavfile import write
import joblib
import numpy as np
import matplotlib.pyplot as plt
import whisper
from fuzzywuzzy import fuzz
import os
import librosa

from sentences import get_random_sentence
from accent_predictor import extract_features
from pronunciation_feedback import pronunciation_feedback

# ================= SILENCE CHECK =================
def is_audio_too_quiet(audio_path, threshold=0.01):
    y, _ = librosa.load(audio_path, sr=16000)
    rms = np.sqrt(np.mean(y**2))
    return rms < threshold

# ================= SESSION STATE =================
if "audio" not in st.session_state:
    st.session_state.audio = None
if "sentence_data" not in st.session_state:
    st.session_state.sentence_data = None
if "accent" not in st.session_state:
    st.session_state.accent = None

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Accent Coach",
    page_icon="🎧",
    layout="wide"
)

# ================= BLACK UI =================
st.markdown("""
<style>
html, body, [data-testid="stApp"] {
    background-color: #000000;
    color: #f8fafc;
}
[data-testid="stSidebar"] {
    background-color: #020617;
}
.card {
    background: #0f172a;
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 0 18px rgba(99,102,241,0.18);
}
h1, h2 { color: #38bdf8 !important; }
h3 { color: #f472b6 !important; }
p, span, label { color: #e5e7eb !important; }

.metric {
    font-size: 34px;
    font-weight: 900;
    color: #22c55e;
}

.stButton>button {
    background: linear-gradient(90deg,#6366f1,#8b5cf6);
    color: white;
    border-radius: 12px;
    border: none;
}

.small-text {
    color: #cbd5f5;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("## 🎧 Accent Coach")
st.markdown("<div class='small-text'>Listen • Speak • Analyze • Improve</div>", unsafe_allow_html=True)

# ================= LOAD MODELS =================
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

# ================= FIXED REFERENCE AUDIO =================
if audio_file:
    ref_path = f"reference_audio/{target_accent}/{audio_file}"
    if os.path.exists(ref_path):
        st.markdown(
            f"<p style='color:#22c55e;'>🎧 Reference Audio ({target_accent.capitalize()})</p>",
            unsafe_allow_html=True
        )
        st.audio(ref_path)
    else:
        st.warning("Reference audio not found")

st.markdown("</div>", unsafe_allow_html=True)

# ================= AUDIO INPUT =================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🎙️ Your Voice")

col1, col2 = st.columns(2)

with col1:
    mode = st.radio("Input Method", ["Microphone", "Upload WAV"])

with col2:
    if mode == "Microphone":
        if st.button("🎤 Record (5 sec)"):
            fs = 16000
            rec = sd.rec(int(5 * fs), samplerate=fs, channels=1)
            sd.wait()
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            write(temp.name, fs, rec)
            st.session_state.audio = temp.name
            st.success("Recorded")
    else:
        file = st.file_uploader("Upload WAV", type=["wav"])
        if file:
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp.write(file.read())
            st.session_state.audio = temp.name
            st.success("Uploaded")

st.markdown("</div>", unsafe_allow_html=True)

# ================= ANALYSIS =================
if st.session_state.audio and st.button("🚀 Analyze"):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Results")

    # Silence check
    if is_audio_too_quiet(st.session_state.audio):
        st.warning("🔊 Please speak louder. Audio too low or silent.")
        st.stop()

    with st.spinner("Analyzing pronunciation..."):
        result = whisper_model.transcribe(st.session_state.audio)

    spoken = result["text"].lower().strip()
    similarity = fuzz.ratio(sentence.lower(), spoken)

    if similarity < 80:
        st.error("❌ Sentence mismatch")
        st.write("Detected:", spoken)
        st.stop()

    st.success(f"Sentence Accuracy: {similarity:.1f}%")

    # Accent prediction
    features = extract_features(st.session_state.audio).reshape(1, -1)
    probs = classifier.predict_proba(features)[0] * 100

    idx = list(label_encoder.classes_).index(target_accent)
    score = probs[idx]

    st.markdown("<h3>🎯 Target Accent Confidence</h3>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric'>{score:.1f}%</div>", unsafe_allow_html=True)

    # ================= COMPACT BEAUTIFUL GRAPH =================
    fig, ax = plt.subplots(figsize=(2.4, 1.6))

    colors = ["#60a5fa", "#a78bfa", "#34d399", "#fb923c"]
    bars = ax.bar(label_encoder.classes_, probs, color=colors, width=0.5)
    bars[idx].set_color("#22c55e")

    ax.set_ylim(0, 100)
    ax.set_title("Accent Match", fontsize=8)
    ax.set_ylabel("%", fontsize=7)

    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            f"{bar.get_height():.0f}%",
            ha="center",
            fontsize=6
        )

    st.pyplot(fig, use_container_width=False)

    # ================= PRONUNCIATION GUIDE =================
    st.markdown("<h3>🗣️ Sentence-Specific Pronunciation Guide</h3>", unsafe_allow_html=True)

    for tip in pronunciation_feedback(target_accent, spoken, sentence):
        st.markdown(f"• {tip}")

    st.success("✔ Practice session completed")
    st.markdown("</div>", unsafe_allow_html=True)