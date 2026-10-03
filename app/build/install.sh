#!/bin/bash
# One-line installer for colleagues — no Gatekeeper prompt, no Developer ID needed.
#
#   curl -fsSL <link-to-this-script> | bash -s <link-to-the-zip>
#
# Files fetched with curl aren't marked "downloaded from the internet", so macOS
# doesn't block the (ad-hoc signed, un-notarized) app. The xattr line clears the
# mark anyway in case the zip arrived some other way.
# Dropbox links: use ?dl=1 (direct download), not ?dl=0.
set -euo pipefail

ZIP_URL="${1:?usage: install.sh <url-of-360-AutoCam-arm64-mac.zip>}"
DEST="${DEST:-/Applications}"
APP="$DEST/360 AutoCam.app"

if [ "$(uname -m)" != "arm64" ]; then
  echo "360 AutoCam needs an Apple Silicon Mac (M1 or newer)."; exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
echo "Downloading 360 AutoCam (~480 MB)…"
curl -fL --progress-bar "$ZIP_URL" -o "$TMP/app.zip"

echo "Installing to $DEST…"
ditto -x -k "$TMP/app.zip" "$TMP/unzipped"
rm -rf "$APP"
ditto "$TMP/unzipped/360 AutoCam.app" "$APP"
xattr -dr com.apple.quarantine "$APP" 2>/dev/null || true
codesign --verify --deep --strict "$APP" && echo "Installed and verified: $APP"

[ -z "${NO_OPEN:-}" ] && open "$APP" || true
