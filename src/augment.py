# src/augment.py
import random
import numpy as np
from audiomentations import (Compose, AddColorNoise, RoomSimulator, Gain,
                             Mp3Compression, SevenBandParametricEQ)
from src import config


def _train_chain():
    """Full realism chain for training (RoomSimulator reverb included)."""
    return Compose([
        AddColorNoise(min_snr_db=8.0, max_snr_db=30.0, p=0.7),
        RoomSimulator(p=0.4),
        Gain(min_gain_db=-6.0, max_gain_db=6.0, p=0.4),
        SevenBandParametricEQ(p=0.3),
        Mp3Compression(min_bitrate=32, max_bitrate=96, p=0.3),
    ])


def _val_chain():
    """Deterministic-when-seeded subset for the fixed val_aug bake.
    RoomSimulator is excluded because pyroomacoustics is not reproducibly seedable."""
    return Compose([
        AddColorNoise(min_snr_db=8.0, max_snr_db=30.0, p=0.8),
        Gain(min_gain_db=-6.0, max_gain_db=6.0, p=0.5),
        SevenBandParametricEQ(p=0.4),
        Mp3Compression(min_bitrate=32, max_bitrate=96, p=0.3),
    ])


def build_train_augmenter():
    return _train_chain()


def build_fixed_val_augmenter():
    return _val_chain()


def apply(aug, wav, seed=None):
    """Apply an augmenter. Pass a seed to make the (deterministic) chain reproducible;
    audiomentations randomizes parameters at call time, so the seed is set here, not at build."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    return aug(samples=np.asarray(wav, dtype=np.float32), sample_rate=config.SAMPLE_RATE)
