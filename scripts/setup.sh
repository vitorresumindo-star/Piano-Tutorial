#!/usr/bin/env bash
# One-time setup: Python deps, three.js, Playwright Chromium and the CC-BY piano samples.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "==> Python packages"
python3 -m pip install -r requirements.txt
echo "==> Node packages (three.js, fonts, playwright)"
npm install
if ! node -e "require('playwright')" 2>/dev/null; then
  npx playwright install chromium
fi
SAMPLES=assets/downloads/SalamanderGrandPianoV3_44.1khz16bit
if [ ! -f "$SAMPLES/SalamanderGrandPianoV3.sfz" ]; then
  echo "==> Downloading Salamander Grand Piano V3 (CC BY 3.0, ~400 MB)"
  mkdir -p assets/downloads
  curl -L -o assets/downloads/salamander.tar.xz \
    "https://freepats.zenvoid.org/Piano/SalamanderGrandPiano/SalamanderGrandPianoV3+20161209_44khz16bit.tar.xz"
  tar -C assets/downloads -xf assets/downloads/salamander.tar.xz
  rm assets/downloads/salamander.tar.xz
fi
echo "==> Checking environment"
python3 -m pianostudio doctor
