# /// script
# requires-python = ">=3.11"
# dependencies = ["requests"]
# ///
"""Pull open Kalshi crypto markets and save them as JSON.

    uv run ingest/fetch_kalshi.py

Writes ingest/data/kalshi_markets_<timestamp>.json (a snapshot) and
ingest/data/kalshi_markets_latest.json (always the most recent pull).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
DATA_DIR = Path(__file__).parent / "data"


def get_tickers():
    resp = requests.get(f"{BASE_URL}/series?category=Crypto&include_volume=true&tags=15%20min")
    resp.raise_for_status()
    data = resp.json()["series"]
    i = 0
    while i < len(data):
        if data[i]["volume_fp"] == "0.00":
            data.pop(i)
            continue
        i += 1

    array = []
    for item in data:
        resp = requests.get(f"{BASE_URL}/markets?limit=100&series_ticker={item['ticker']}&status=open")
        resp.raise_for_status()
        array.append(resp.json())
    return array


if __name__ == "__main__":
    array = get_tickers()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_file = DATA_DIR / f"kalshi_markets_{timestamp}.json"
    latest_file = DATA_DIR / "kalshi_markets_latest.json"

    snapshot_file.write_text(json.dumps(array, indent=2), encoding="utf-8")
    latest_file.write_text(json.dumps(array, indent=2), encoding="utf-8")

    print(f"saved {len(array)} series to {snapshot_file.name} and kalshi_markets_latest.json")
