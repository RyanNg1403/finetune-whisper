# tests/test_dataset.py
import json
import numpy as np
import soundfile as sf
from transformers import WhisperProcessor
from src.dataset import WhisperASRDataset, DataCollatorSpeechSeq2SeqWithPadding
from src import config


def _mini(tmp_path):
    d = tmp_path / "train"
    d.mkdir()
    for i in range(2):
        sf.write(d / f"a{i}.wav", (0.1 * np.random.randn(16000)).astype(np.float32), config.SAMPLE_RATE)
    with open(d / "manifest.jsonl", "w") as f:
        for i in range(2):
            f.write(json.dumps({"audio": f"a{i}.wav", "text": "claude opus", "terms": ["Claude Opus"]}) + "\n")
    return d


def test_dataset_yields_features_and_labels(tmp_path):
    d = _mini(tmp_path)
    proc = WhisperProcessor.from_pretrained(config.MODEL_ID)
    ds = WhisperASRDataset([d], proc, augment=False)
    item = ds[0]
    assert "input_features" in item and "labels" in item
    assert len(item["labels"]) > 0


def test_collator_pads_batch(tmp_path):
    d = _mini(tmp_path)
    proc = WhisperProcessor.from_pretrained(config.MODEL_ID)
    ds = WhisperASRDataset([d], proc, augment=False)
    coll = DataCollatorSpeechSeq2SeqWithPadding(proc)
    batch = coll([ds[0], ds[1]])
    assert batch["input_features"].shape[0] == 2
    assert batch["labels"].shape[0] == 2
