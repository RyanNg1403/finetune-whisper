# src/config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
AUDIO = DATA / "audio"
TRAIN_DIR = AUDIO / "train"
VAL_CLEAN_DIR = AUDIO / "val_clean"
VAL_AUG_DIR = AUDIO / "val_aug"
TERMS_PATH = DATA / "terms.yaml"
HARD_TERMS_PATH = DATA / "terms_hard.yaml"
TERMS_PATHS = [TERMS_PATH, HARD_TERMS_PATH]   # combined termset (official + hard)
CORPUS_PATH = DATA / "corpus.jsonl"
CHECKPOINTS = ROOT / "checkpoints"

# --- Kokoro (bulk 70% local TTS engine, ONNX) ---
KOKORO_DIR = ROOT / "kokoro_models"
KOKORO_ONNX = KOKORO_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES_BIN = KOKORO_DIR / "voices-v1.0.bin"
KOKORO_TRAIN_VOICES = ["af_heart", "af_bella", "af_nicole", "am_michael", "am_adam", "am_onyx"]
KOKORO_VAL_VOICES = ["af_sarah", "am_puck"]   # held-out speakers
KOKORO_SPEEDS = [0.95, 1.0, 1.1]

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
