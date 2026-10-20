#!/usr/bin/env bash
# Release build of the fork for macOS arm64, signed with the local identity and
# installed to /Applications/Handy.app (replacing what is there). See FORK.md.
set -euo pipefail

IDENTITY="${HANDY_SIGN_IDENTITY:-Handy Fork Local}"
TARGET=aarch64-apple-darwin
DEST=/Applications/Handy.app

root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"

if ! security find-identity -p codesigning | grep -q "\"$IDENTITY\""; then
  echo "error: code-signing identity '$IDENTITY' not in the keychain (FORK.md, Signing)" >&2
  exit 1
fi

bun install --frozen-lockfile

# Updater artifacts need upstream's private key; the fork never ships updates.
CMAKE_POLICY_VERSION_MINIMUM=3.5 bun run tauri build \
  --target "$TARGET" --bundles app \
  --config '{"bundle":{"createUpdaterArtifacts":false}}'

app="src-tauri/target/$TARGET/release/bundle/macos/Handy.app"
[ -d "$app" ] || { echo "error: $app not found" >&2; exit 1; }

# Tauri signs ad-hoc; re-sign with the stable identity so macOS keeps the
# microphone and Accessibility grants across rebuilds.
codesign --force --deep --options runtime --timestamp=none \
  --entitlements src-tauri/Entitlements.plist \
  -s "$IDENTITY" "$app"
codesign --verify --deep --strict "$app"

if pgrep -x handy >/dev/null; then
  echo "Quitting the running Handy"
  osascript -e 'quit app "Handy"' || true
  sleep 2
fi
rm -rf "$DEST"
ditto "$app" "$DEST"

codesign -dv "$DEST" 2>&1 | grep -E '^(Identifier|Authority)='
echo "Installed $DEST $(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$DEST/Contents/Info.plist") ($(git rev-parse --short HEAD))"
