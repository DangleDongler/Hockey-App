#!/bin/bash
# Put Shot Tracker on the internet with Fly.io, the first time.
# Run from the app's folder:  bash deploy/fly-setup.sh
# Needs the fly command installed and logged in (see "PUT IT ON YOUR PHONE.txt").
set -euo pipefail
cd "$(dirname "$0")/.."
SAVED="$HOME/.shottracker-fly"

command -v fly >/dev/null || { echo "The fly command isn't installed yet: see step 1 of PUT IT ON YOUR PHONE.txt"; exit 1; }
fly auth whoami >/dev/null 2>&1 || { echo "Log in to Fly first:  fly auth login"; exit 1; }

echo
echo "A name for your app. It becomes the web address: NAME.fly.dev"
echo "Lowercase letters, numbers and dashes, e.g. smith-shot-tracker"
read -r -p "Name: " NAME
[[ "$NAME" =~ ^[a-z0-9][a-z0-9-]{1,40}[a-z0-9]$ ]] || { echo "Use lowercase letters, numbers and dashes only."; exit 1; }

echo
echo "Where the server should be: pick the code of the place nearest you."
fly platform regions || true
read -r -p "Region code (e.g. sea, ord, yyz, iad, lax): " REGION
[[ "$REGION" =~ ^[a-z]{3}$ ]] || { echo "A region code is three letters, like sea."; exit 1; }

echo
echo "The password you'll type on your phone (at least 8 characters)."
read -r -s -p "Password: " PW; echo
read -r -s -p "Same again: " PW2; echo
[[ "$PW" == "$PW2" ]] || { echo "Those didn't match."; exit 1; }
[[ ${#PW} -ge 8 ]] || { echo "At least 8 characters, please."; exit 1; }

printf 'NAME=%s\nREGION=%s\n' "$NAME" "$REGION" > "$SAVED"
sed -i.bak -e "s/^app = .*/app = \"$NAME\"/" -e "s/^primary_region = .*/primary_region = \"$REGION\"/" fly.toml
rm -f fly.toml.bak

echo; echo "== 1/4 Creating the app $NAME"
fly apps create "$NAME" --org personal
echo; echo "== 2/4 Making room for your sessions (5 GB)"
fly volumes create shot_data --size 5 --region "$REGION" --app "$NAME" --yes
echo; echo "== 3/4 Setting the password"
printf 'SHOTTRACKER_PASSWORD=%s\n' "$PW" | fly secrets import --app "$NAME" --stage
echo; echo "== 4/4 Building and starting it (a few minutes the first time)"
fly deploy --ha=false

echo
echo "Done. On your phone, open:  https://$NAME.fly.dev"
echo "Then Share > Add to Home Screen."
