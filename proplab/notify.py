"""Discord alerts. Set DISCORD_WEBHOOK_URL in .env; without it, messages just print."""
from __future__ import annotations

import os

import requests


def send(msg: str) -> None:
    print(msg, flush=True)
    url = os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        return
    try:
        requests.post(url, json={"content": msg[:1900]}, timeout=10)
    except Exception as exc:  # an alert failing must never crash the bot
        print(f"[discord] failed to send: {exc}", flush=True)
