# src/audio_io.py
import numpy as np
import soundfile as sf
import librosa
from src import config


def save_wav_16k(path, wav):
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    sf.write(str(path), wav, config.SAMPLE_RATE)


def load_wav_16k_mono(path):
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != config.SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=config.SAMPLE_RATE)
    return wav.astype(np.float32)
