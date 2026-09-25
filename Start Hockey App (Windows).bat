@echo off
rem Double-click to start the Hockey Shot Tracker.
rem The first time, it sets itself up (a few minutes); after that it starts in seconds.
setlocal
cd /d "%~dp0"

set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if not defined PY (
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
    if not errorlevel 1 set "PY=python"
)
if not defined PY (
    echo This needs Python 3.10 or newer, which isn't installed yet.
    echo.
    echo   1. On the page that opens, download the latest Python for Windows and install it.
    echo      On the first installer screen, tick "Add python.exe to PATH".
    echo   2. Then double-click this file again.
    start "" "https://www.python.org/downloads/windows/"
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Setting up for the first time. This takes a few minutes...
    %PY% -m venv .venv
    if errorlevel 1 goto failed
)

fc /b requirements.txt ".venv\installed-requirements.txt" >nul 2>nul
if errorlevel 1 (
    echo Installing what the app needs...
    ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q --upgrade pip
    ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt
    if errorlevel 1 goto failed
    copy /y requirements.txt ".venv\installed-requirements.txt" >nul
)

".venv\Scripts\python.exe" start.py
pause
exit /b 0

:failed
echo.
echo The setup failed. Check the internet connection and try again.
pause
exit /b 1
