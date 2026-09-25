#!/bin/bash
# Double-click to start the Hockey Shot Tracker.
# The first time, it sets itself up (a few minutes); after that it starts in seconds.

cd "$(dirname "$0")" || exit 1

pause_and_exit() {
    echo
    read -r -p "Press Return to close this window." _
    exit "${1:-1}"
}

good_python() {
    "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1
}

PY=""
for candidate in \
    /Library/Frameworks/Python.framework/Versions/Current/bin/python3 \
    /opt/homebrew/bin/python3 /usr/local/bin/python3 \
    python3.13 python3.12 python3.11 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1 && good_python "$candidate"; then
        PY="$candidate"
        break
    fi
done

if [ -z "$PY" ]; then
    echo "This needs Python 3.10 or newer, which isn't installed yet."
    echo
    echo "  1. On the page that opens, download the latest Python for macOS and install it."
    echo "  2. Then double-click this file again."
    open "https://www.python.org/downloads/macos/"
    pause_and_exit 1
fi

if [ ! -x .venv/bin/python ]; then
    echo "Setting up for the first time. This takes a few minutes..."
    "$PY" -m venv .venv || { echo "Could not set up Python here."; pause_and_exit 1; }
fi

# Install (or update) what the app needs whenever the list has changed.
if ! cmp -s requirements.txt .venv/installed-requirements.txt; then
    echo "Installing what the app needs..."
    .venv/bin/python -m pip install --disable-pip-version-check -q --upgrade pip &&
    .venv/bin/python -m pip install --disable-pip-version-check -q -r requirements.txt ||
        { echo; echo "The install failed. Check the internet connection and try again."; pause_and_exit 1; }
    cp requirements.txt .venv/installed-requirements.txt
fi

.venv/bin/python start.py
pause_and_exit $?
