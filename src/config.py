# src/config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
AUDIO = DATA / "audio"
TRAIN_DIR = AUDIO / "train"
VAL_CLEAN_DIR = AUDIO / "val_clean"
VAL_AUG_DIR = AUDIO / "val_aug"
TERMS_PATH = DATA / "terms.yaml"
CORPUS_PATH = DATA / "corpus.jsonl"
CHECKPOINTS = ROOT / "checkpoints"

# --- Kokoro (bulk 70% local TTS engine, ONNX) ---
KOKORO_DIR = ROOT / "kokoro_models"
KOKORO_ONNX = KOKORO_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES_BIN = KOKORO_DIR / "voices-v1.0.bin"
KOKORO_TRAIN_VOICES = ["af_heart", "af_bella", "af_nicole", "am_michael", "am_adam", "am_onyx"]
KOKORO_VAL_VOICES = ["af_sarah", "am_puck"]   # held-out speakers
KOKORO_SPEEDS = [0.95, 1.0, 1.1]

# --- Supertonic (superseded as bulk engine; kept for reference) ---
SUPERTONIC_DIR = ROOT / "supertonic"
SUPERTONIC_ONNX = SUPERTONIC_DIR / "assets" / "onnx"
SUPERTONIC_VOICE_DIR = SUPERTONIC_DIR / "assets" / "voice_styles"
SUPERTONIC_TRAIN_VOICES = ["M1", "M2", "M3", "F1", "F2", "F3"]
SUPERTONIC_VAL_VOICES = ["M4", "F4"]
SUPERTONIC_SPEEDS = [0.95, 1.05, 1.15]

SAMPLE_RATE = 16000
MIN_TERM_COVERAGE = 8
ENGINE_RATIO = (0.7, 0.3)  # (kokoro, openai) of TRAIN clips
TARGET_TRAIN_CLIPS = 6000
MODEL_ID = "openai/whisper-base.en"
QC_MODEL = "openai/whisper-small.en"   # reference ASR for back-transcription QC

OPENAI_TRAIN_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
OPENAI_VAL_VOICES = ["ash", "sage"]
OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
SEED = 42
