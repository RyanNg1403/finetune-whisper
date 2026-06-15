# tests/test_config.py
from src import config


def test_core_constants():
    assert config.SAMPLE_RATE == 16000
    assert abs(sum(config.ENGINE_RATIO) - 1.0) < 1e-9
    assert config.MIN_TERM_COVERAGE >= 1
    assert config.MODEL_ID == "openai/whisper-base.en"
    # train and val voices must be disjoint (no speaker leakage)
    assert not (set(config.KOKORO_TRAIN_VOICES) & set(config.KOKORO_VAL_VOICES))
    assert not (set(config.OPENAI_TRAIN_VOICES) & set(config.OPENAI_VAL_VOICES))
