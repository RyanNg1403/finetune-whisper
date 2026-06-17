#!/usr/bin/env bash
# Fetch + extract the dataset (data/audio/) and trained checkpoints/ from the project's
# Google Drive folder. The folder holds two archives (audio + checkpoints).
# Usage:  bash scripts/download_audio.sh ["<drive-folder-link>"]
#
# Needs the Drive folder to be public ("anyone with the link"). If gdown fails, just
# download the two archives manually from the folder and extract (see bottom of this file).
set -euo pipefail

LINK="${1:-https://drive.google.com/drive/folders/1SjRzRAguKup-FInrpSE5XVxAp_kt8ScS}"
DL="_drive_dl"

echo "Installing gdown…"
python -m pip install -q gdown

rm -rf "$DL"; mkdir -p "$DL"
echo "Downloading from Drive folder…"
python -m gdown --folder "$LINK" -O "$DL"

echo "Extracting archives…"
shopt -s globstar nullglob
extracted=0
for f in "$DL"/**/*.tar.gz "$DL"/**/*.tgz "$DL"/**/*.zip; do
  top=""
  case "$f" in
    *.zip) top=$(unzip -Z1 "$f" | head -1 | cut -d/ -f1) ;;
    *)     top=$(tar tzf "$f" | head -1 | cut -d/ -f1) ;;
  esac
  echo "  $(basename "$f")  (top-level: $top)"
  # route by what the archive contains
  if [ "$top" = "audio" ]; then          dest="data";        mkdir -p data
  elif [ "$top" = "data" ]; then         dest=".";
  elif [ "$top" = "checkpoints" ]; then  dest=".";
  else                                   dest="."; fi
  case "$f" in
    *.zip) unzip -oq "$f" -d "$dest" ;;
    *)     tar xzf "$f" -C "$dest" ;;
  esac
  extracted=$((extracted+1))
done
rm -rf "$DL"

if [ "$extracted" -eq 0 ]; then
  echo "No archives found in the download. Make sure the Drive folder is public and holds the two .tar.gz files."
  exit 1
fi
echo "Done ($extracted archive(s)). Expect data/audio/ and checkpoints/ in the repo root."

# --- Manual fallback (if gdown can't reach the folder) ---
# 1) Open the Drive folder, download the two archives.
# 2) tar xzf whisper-aidev-audio.tar.gz       -C data    # -> data/audio/
#    tar xzf whisper-aidev-checkpoints.tar.gz            # -> checkpoints/
