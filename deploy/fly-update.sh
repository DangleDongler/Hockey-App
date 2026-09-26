#!/bin/bash
# Put a newer version of Shot Tracker on the server set up by fly-setup.sh.
# Run from the new version's folder:  bash deploy/fly-update.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SAVED="$HOME/.shottracker-fly"
[[ -f "$SAVED" ]] || { echo "Run bash deploy/fly-setup.sh first."; exit 1; }
# shellcheck disable=SC1090
source "$SAVED"
echo "Updating $NAME ($REGION)"
sed -i.bak -e "s/^app = .*/app = \"$NAME\"/" -e "s/^primary_region = .*/primary_region = \"$REGION\"/" fly.toml
rm -f fly.toml.bak
fly deploy --ha=false
echo "Done: https://$NAME.fly.dev"
