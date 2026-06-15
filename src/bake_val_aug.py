# src/bake_val_aug.py
import json
import glob
from src import config
from src.audio_io import load_wav_16k_mono, save_wav_16k
from src.augment import build_fixed_val_augmenter, apply


def main():
    config.VAL_AUG_DIR.mkdir(parents=True, exist_ok=True)
    aug = build_fixed_val_augmenter()
    out_mf = open(config.VAL_AUG_DIR / "manifest.jsonl", "w")
    n = 0
    for m in sorted(glob.glob(str(config.VAL_CLEAN_DIR / "manifest*.jsonl"))):
        for line in open(m):
            row = json.loads(line)
            wav = load_wav_16k_mono(config.VAL_CLEAN_DIR / row["audio"])
            wav = apply(aug, wav, seed=config.SEED + n)   # per-clip deterministic
            fn = "aug_" + row["audio"]
            save_wav_16k(config.VAL_AUG_DIR / fn, wav)
            out_mf.write(json.dumps(dict(row, audio=fn)) + "\n")
            n += 1
    out_mf.close()
    print(f"baked {n} augmented val clips -> {config.VAL_AUG_DIR}")


if __name__ == "__main__":
    main()
