#!/bin/zsh
# Builds ../JARVIS.app from the files in this folder (macOS only; no downloads).
#   JARVIS.applescript  the stay-open applet (Dock icon, reopen, quit)
#   launcher.sh         starts/stops the server
#   JARVIS.icns         the icon (python make_icon.py icon.png, then sips + iconutil)
set -e
HERE="${0:A:h}"
APP="${HERE:h}/JARVIS.app"
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "${HERE:h}/backend/pyproject.toml")"

rm -rf "$APP"
osacompile -s -o "$APP" "$HERE/JARVIS.applescript"  # -s: stay open
install -m 755 "$HERE/launcher.sh" "$APP/Contents/Resources/launcher.sh"
cp "$HERE/JARVIS.icns" "$APP/Contents/Resources/applet.icns"
rm -f "$APP/Contents/Resources/Assets.car"  # holds the default script icon, which wins over .icns

PLIST="$APP/Contents/Info.plist"
plutil -replace CFBundleIdentifier -string "local.jarvis.launcher" "$PLIST"
plutil -replace CFBundleName -string "JARVIS" "$PLIST"
plutil -remove CFBundleIconName "$PLIST"
plutil -replace CFBundleShortVersionString -string "$VERSION" "$PLIST"
plutil -replace CFBundleVersion -string "$VERSION" "$PLIST"
plutil -replace NSAppSleepDisabled -bool YES "$PLIST"  # no App Nap: JARVIS must stay responsive
codesign --force --sign - "$APP"  # ad-hoc signature, valid again after the changes above
echo "Pronto: $APP ($VERSION)"
