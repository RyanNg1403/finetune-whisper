# tests/test_audio_io.py
import numpy as np
from src.audio_io import save_wav_16k, load_wav_16k_mono
from src import config


def test_roundtrip_16k_mono(tmp_path):
    wav = (0.1 * np.random.randn(8000)).astype(np.float32)  # 0.5s @16k
    p = tmp_path / "a.wav"
    save_wav_16k(p, wav)
    out = load_wav_16k_mono(p)
    assert out.ndim == 1
    assert len(out) == 8000


def test_resamples_and_downmixes(tmp_path):
    import soundfile as sf
    stereo = (0.1 * np.random.randn(48000, 2)).astype(np.float32)  # 1s @48k stereo
    p = tmp_path / "b.wav"
    sf.write(p, stereo, 48000)
    out = load_wav_16k_mono(p)
    assert out.ndim == 1
    assert abs(len(out) - 16000) <= 1
