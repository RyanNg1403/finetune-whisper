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
SUPERTONIC_DIR = ROOT / "supertonic"
SUPERTONIC_ONNX = SUPERTONIC_DIR / "assets" / "onnx"
SUPERTONIC_VOICE_DIR = SUPERTONIC_DIR / "assets" / "voice_styles"

SAMPLE_RATE = 16000
MIN_TERM_COVERAGE = 8
ENGINE_RATIO = (0.7, 0.3)  # (supertonic, openai) of TRAIN clips
TARGET_TRAIN_CLIPS = 6000
MODEL_ID = "openai/whisper-base.en"

SUPERTONIC_TRAIN_VOICES = ["M1", "M2", "M3", "F1", "F2", "F3"]
SUPERTONIC_VAL_VOICES = ["M4", "F4"]
SUPERTONIC_SPEEDS = [0.95, 1.05, 1.15]
OPENAI_TRAIN_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
OPENAI_VAL_VOICES = ["ash", "sage"]
OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
SEED = 42
