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
import librosa

from sentences import get_random_sentence
from accent_predictor import extract_features
from pronunciation_feedback import pronunciation_feedback

# ===== SAFE IMPORT =====
try:
    import sounddevice as sd
    LOCAL_MIC = True
except:
    LOCAL_MIC = False

from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

torch.set_num_threads(1)

# ================= AUDIO =================
def prepare_audio(path):
    try:
        audio, sr = sf.read(path)
    except:
        audio, sr = librosa.load(path, sr=16000)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)

    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    audio = np.nan_to_num(audio)

    if len(audio) < 16000:
        audio = np.pad(audio, (0, 16000 - len(audio)))

    return audio

def is_audio_too_quiet(path):
    try:
        audio = prepare_audio(path)
        return np.sqrt(np.mean(audio**2)) < 0.01
    except:
        return False

# ================= SESSION =================
if "audio" not in st.session_state:
    st.session_state.audio = None
if "sentence_data" not in st.session_state:
    st.session_state.sentence_data = None
if "accent" not in st.session_state:
    st.session_state.accent = None

st.set_page_config(page_title="Accent Coach", layout="wide")

st.markdown("""
<style>
html,body {background:#000;color:#fff;}
.card {background:#0f172a;padding:20px;border-radius:15px;margin-bottom:20px;}
.metric {font-size:34px;color:#22c55e;}
</style>
""", unsafe_allow_html=True)

st.title("🎧 Accent Coach")

# ================= LOAD =================
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

st.markdown("<div class='card'>", unsafe_allow_html=True)

c1, c2 = st.columns([8,1])
with c1:
    st.subheader("Practice Sentence")
with c2:
    if st.button("New"):
        st.session_state.sentence_data = get_random_sentence(target_accent)
        st.rerun()

st.info(sentence)

if audio_file:
    path = f"reference_audio/{target_accent}/{audio_file}"
    if os.path.exists(path):
        st.audio(path)

st.markdown("</div>", unsafe_allow_html=True)

# ================= INPUT =================
mode = st.radio("Input Method", ["Microphone", "Upload WAV"])

if mode == "Microphone":

    if LOCAL_MIC:
        if st.button("Record (5 sec)"):
            fs = 16000
            rec = sd.rec(int(5 * fs), samplerate=fs, channels=1)
            sd.wait()

            audio = rec.squeeze()
            audio = np.clip(audio, -1, 1)

            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            write(temp.name, fs, (audio * 32767).astype(np.int16))

            st.session_state.audio = temp.name
            st.success("Recorded")

    else:
        rtc_config = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })

        webrtc_ctx = webrtc_streamer(
            key="webrtc_audio",
            mode=WebRtcMode.SENDONLY,
            rtc_configuration=rtc_config,
            media_stream_constraints={"audio": True, "video": False},
        )

        if st.button("Stop & Save"):

            frames = []

            if webrtc_ctx and webrtc_ctx.audio_receiver:
                try:
                    for _ in range(500):
                        frame = webrtc_ctx.audio_receiver.get_frame(timeout=2)
                        if frame is None:
                            break
                        frames.append(frame.to_ndarray())
                except:
                    pass

            if len(frames) < 10:
                st.warning("No proper audio captured. Speak clearly.")
                st.stop()

            audio = np.concatenate(frames, axis=0)

            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            audio = audio.astype(np.float32)

            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))

            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            write(temp.name, 16000, (audio * 32767).astype(np.int16))

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
if st.button("Analyze") and st.session_state.audio:

    audio = prepare_audio(st.session_state.audio)

    result = whisper_model.transcribe(
        audio,
        language="en",
        temperature=0,
        fp16=False
    )

    spoken = result["text"].lower().strip()

    st.write("You said:", spoken)

    similarity = fuzz.partial_ratio(sentence.lower(), spoken)

    if similarity < 60:
        st.error("Sentence mismatch")
        st.stop()

    st.success(f"Accuracy: {similarity:.1f}%")

    features = extract_features(st.session_state.audio).reshape(1, -1)
    probs = classifier.predict_proba(features)[0] * 100

    idx = list(label_encoder.classes_).index(target_accent)

    st.markdown(f"<div class='metric'>{probs[idx]:.1f}%</div>", unsafe_allow_html=True)

    fig, ax = plt.subplots()
    bars = ax.bar(label_encoder.classes_, probs)
    bars[idx].set_color("green")

    st.pyplot(fig)

    for tip in pronunciation_feedback(target_accent, spoken, sentence):
        st.write("•", tip)