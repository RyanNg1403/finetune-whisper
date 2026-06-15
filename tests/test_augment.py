# tests/test_augment.py
import numpy as np
from src.augment import build_train_augmenter, build_fixed_val_augmenter, apply


def test_train_aug_changes_signal():
    rng = np.random.default_rng(0)
    wav = (0.1 * rng.standard_normal(16000)).astype(np.float32)
    aug = build_train_augmenter()
    out = apply(aug, wav)
    assert out.shape[0] >= 1 and not np.array_equal(out, wav)


def test_fixed_val_aug_is_deterministic():
    wav = (0.1 * np.random.default_rng(1).standard_normal(16000)).astype(np.float32)
    aug = build_fixed_val_augmenter()
    a = apply(aug, wav, seed=123)
    b = apply(aug, wav, seed=123)
    assert np.allclose(a, b)
