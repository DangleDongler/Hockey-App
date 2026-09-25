"""Start the Shot Tracker web app for use at home.

Run by the "Start Hockey App" launchers (or directly: ``python start.py``).
It serves the app to this computer and to phones on the same Wi-Fi, keeps
sessions in the ``sessions`` folder next to this file so your history
survives restarts, and opens the page in your browser.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIRST_PORT = 8000


def lan_address() -> str | None:
    """This computer's address on the home network, for the phone to use."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # No packet is sent: connecting a UDP socket only picks the route.
        s.connect(("192.168.0.1", 9))
        ip = s.getsockname()[0]
        return None if ip.startswith("127.") else ip
    except OSError:
        return None
    finally:
        s.close()


def free_port(first: int = FIRST_PORT, tries: int = 20) -> int:
    for port in range(first, first + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise SystemExit(f"ports {first}-{first + tries - 1} are all in use; close something and try again")


def main() -> None:
    # Before the app is imported: it reads these once.
    os.environ.setdefault("SHOTTRACKER_DATA", str(ROOT / "sessions"))
    os.environ.setdefault("SHOTTRACKER_MAX_UPLOAD_MB", "4000")
    Path(os.environ["SHOTTRACKER_DATA"]).mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "server")]

    import uvicorn

    port = free_port()
    here = f"http://127.0.0.1:{port}"
    ip = lan_address()
    line = "=" * 64
    print(line)
    print("  Hockey Shot Tracker is running.")
    print()
    print(f"  On this computer:  {here}")
    if ip:
        print(f"  On your phone:     http://{ip}:{port}")
        print("                     (same Wi-Fi as this computer)")
    print()
    print(f"  Sessions are kept in: {os.environ['SHOTTRACKER_DATA']}")
    print("  Leave this window open while you use it. Close it to stop.")
    print(line, flush=True)

    if not os.environ.get("SHOTTRACKER_NO_BROWSER"):
        threading.Timer(1.5, lambda: webbrowser.open(here)).start()
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()
