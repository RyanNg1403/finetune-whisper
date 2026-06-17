#!/usr/bin/env bash
# Download + extract the pre-synthesized audio dataset (~1.2 GB) from Google Drive.
# Usage:  bash scripts/download_audio.sh "<google-drive-share-link>"
# If no link is passed, it falls back to the one published in the README.
set -euo pipefail

LINK="${1:-PASTE_GOOGLE_DRIVE_SHARE_LINK_HERE}"
ARCHIVE="whisper-ai-terms-audio.tar.gz"

if [ "$LINK" = "PASTE_GOOGLE_DRIVE_SHARE_LINK_HERE" ]; then
  echo "No Drive link provided. Pass it as an argument or set it in this script:"
  echo "  bash scripts/download_audio.sh \"https://drive.google.com/file/d/<ID>/view\""
  exit 1
fi

echo "Installing gdown (Google Drive downloader)…"
python -m pip install -q gdown

echo "Downloading $ARCHIVE …"
python -m gdown --fuzzy "$LINK" -O "$ARCHIVE"

echo "Extracting into data/ …"
mkdir -p data
tar -xzf "$ARCHIVE" -C data

echo "Done. Audio is in data/audio/ (train / val_clean / val_aug)."
