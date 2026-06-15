# tests/test_eval.py
import json
import numpy as np
import soundfile as sf
from src.eval import evaluate
from src import config


def test_evaluate_returns_report(tmp_path):
    d = tmp_path / "val"
    d.mkdir()
    sf.write(d / "a.wav", (0.01 * np.random.randn(16000)).astype(np.float32), config.SAMPLE_RATE)
    with open(d / "manifest.jsonl", "w") as f:
        f.write(json.dumps({"audio": "a.wav", "text": "claude opus", "terms": ["Claude Opus"]}) + "\n")
    rep = evaluate(config.MODEL_ID, str(d), device="cpu")
    assert "wer" in rep and "term_recall" in rep and rep["wer"] >= 0.0
    assert rep["n"] == 1
