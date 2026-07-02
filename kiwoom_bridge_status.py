from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the local Kiwoom bridge status.")
    parser.add_argument("--require-login", action="store_true", help="Return success only when /health reports login=true.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    base_url = (os.getenv("KIWOOM_BRIDGE_URL") or "http://127.0.0.1:8765").rstrip("/")
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    if not _socket_open(host, port):
        if not args.quiet:
            print("reachable=false login=false")
        return 1

    health = _health(base_url)
    if health is None:
        if not args.quiet:
            print("reachable=true login=unknown")
        return 2 if args.require_login else 0

    login = bool(health.get("login"))
    if not args.quiet:
        ok = bool(health.get("ok"))
        provider = health.get("provider") or "unknown"
        print(f"reachable=true ok={str(ok).lower()} login={str(login).lower()} provider={provider}")
    if args.require_login and not login:
        return 3
    return 0


def _socket_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.5):
            return True
    except OSError:
        return False


def _health(base_url: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=3) as response:
            payload = response.read().decode("utf-8", errors="replace")
    except (OSError, urllib.error.URLError):
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


if __name__ == "__main__":
    raise SystemExit(main())
