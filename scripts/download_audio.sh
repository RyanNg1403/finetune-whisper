#!/usr/bin/env bash
# Fetch the dataset (data/audio/) + trained checkpoints/ from the project's Google Drive folder.
# Usage:  bash scripts/download_audio.sh ["<drive-folder-link>"]
#
# Note: gdown's folder mode has file-count limits, so for the large audio/ folder the most
# reliable route is the Drive web UI — open the folder, select all, click "Download" (Drive
# zips it), and extract into the repo root. This script is the best-effort CLI convenience.
set -euo pipefail

LINK="${1:-https://drive.google.com/drive/folders/1SjRzRAguKup-FInrpSE5XVxAp_kt8ScS}"

echo "Installing gdown…"
python -m pip install -q gdown

echo "Downloading data/ and checkpoints/ from Drive…"
python -m gdown --folder "$LINK" -O .

echo "Done. Expect data/audio/ and checkpoints/ in the repo root."
echo "If gdown truncated the download (folder file limits), use the Drive UI 'Download as zip' instead."
