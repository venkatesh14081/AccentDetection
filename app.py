import streamlit as st
import tempfile
from scipy.io.wavfile import write
import joblib
import numpy as np
import matplotlib.pyplot as plt
import whisper
from fuzzywuzzy import fuzz
import os
import librosa
import torch

from sentences import get_random_sentence
from accent_predictor import extract_features
from pronunciation_feedback import pronunciation_feedback

# ===== FIX (ONLY CHANGE) =====
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
    y, _ = librosa.load(audio_path, sr=16000, res_type="kaiser_fast")
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

# ================= UI (UNCHANGED) =================
st.markdown("""<style>
html, body, [data-testid="stApp"] {background-color:#000;color:#f8fafc;}
.metric {font-size:34px;font-weight:900;color:#22c55e;}
</style>""", unsafe_allow_html=True)

st.markdown("## 🎧 Accent Coach")

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    clf = joblib.load("model/accent_classifier.pkl")
    le = joblib.load("model/label_encoder.pkl")
    whisper_model = whisper.load_model("tiny")
    return clf, le, whisper_model

classifier, label_encoder, whisper_model = load_models()

# ================= SIDEBAR =================
target_accent = st.sidebar.selectbox(
    "Target Accent",
    ["american", "british", "australian", "canadian"]
)

# ================= SENTENCE =================
if st.session_state.sentence_data is None or st.session_state.accent != target_accent:
    st.session_state.sentence_data = get_random_sentence(target_accent)
    st.session_state.accent = target_accent

sentence, audio_file = st.session_state.sentence_data
st.info(sentence)

if audio_file:
    ref_path = f"reference_audio/{target_accent}/{audio_file}"
    if os.path.exists(ref_path):
        st.audio(ref_path)

# ================= INPUT =================
mode = st.radio("Input Method", ["Microphone", "Upload WAV"])

if mode == "Microphone":

    # ===== LOCAL =====
    if LOCAL_MIC:
        if st.button("🎤 Record (5 sec)"):
            fs = 16000
            rec = sd.rec(int(5 * fs), samplerate=fs, channels=1)
            sd.wait()

            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            write(temp.name, fs, rec.squeeze())

            st.session_state.audio = temp.name
            st.success("Recorded")

    # ===== CLOUD =====
    else:
        st.info("🎤 Using browser microphone")

        webrtc_ctx = webrtc_streamer(
            key="webrtc_audio",
            mode=WebRtcMode.SENDONLY,
            media_stream_constraints={"audio": True, "video": False},
        )

        if st.button("Stop & Save"):
            if webrtc_ctx.audio_receiver:
                frames = []

                for _ in range(200):
                    frame = webrtc_ctx.audio_receiver.get_frame(timeout=1)
                    if frame is None:
                        break
                    frames.append(frame.to_ndarray())

                if frames:
                    audio = np.concatenate(frames, axis=0)

                    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    write(temp.name, 16000, audio.astype(np.float32))

                    st.session_state.audio = temp.name
                    st.success("Recorded (browser)")

else:
    file = st.file_uploader("Upload WAV", type=["wav"])
    if file:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp.write(file.read())
        st.session_state.audio = temp.name
        st.success("Uploaded")

# ================= ANALYSIS =================
analyze = st.button("🚀 Analyze")

if st.session_state.audio and analyze:

    if is_audio_too_quiet(st.session_state.audio):
        st.warning("Audio too quiet")
        st.stop()

    result = whisper_model.transcribe(st.session_state.audio)
    spoken = result["text"].lower().strip()

    similarity = fuzz.ratio(sentence.lower(), spoken)

    if similarity < 80:
        st.error("Sentence mismatch")
        st.write("Detected:", spoken)
        st.stop()

    st.success(f"Sentence Accuracy: {similarity:.1f}%")

    features = extract_features(st.session_state.audio).reshape(1, -1)
    probs = classifier.predict_proba(features)[0] * 100

    idx = list(label_encoder.classes_).index(target_accent)

    st.markdown(f"<div class='metric'>{probs[idx]:.1f}%</div>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(2.5, 1.5))
    bars = ax.bar(label_encoder.classes_, probs)
    bars[idx].set_color("green")

    ax.set_ylim(0, 100)
    st.pyplot(fig)

    for tip in pronunciation_feedback(target_accent, spoken, sentence):
        st.write("•", tip)