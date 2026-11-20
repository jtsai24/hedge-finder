# /// script
# requires-python = ">=3.11"
# dependencies = ["fastapi>=0.162", "uvicorn>=0.54", "openai>=3.51,<4", "python-dotenv>=1.2.4"]
# ///
"""Hedge-bet finder prototype.

    uv run demo/hedge_app.py      then open http://localhost:8001

Pipeline for one input bet:
  1. Ask CHAT_MODEL to describe the real-world opposing event to bet on (a search query, not an essay).
  2. Embed that generated description and search it against a vector database of real Kalshi
     markets (01-inference/embeddings.py and 03-database/vectors.py show the pattern).
  3. Return the matching real markets as candidate hedges.

Step 1 is wired to the real inference endpoint. Step 2/3 (the vector search) is MOCKED for now:
it keyword-scores demo/mock_markets.json instead of querying pgvector, so the UI and the
generation step can be built and demoed before the real market-ingestion pipeline is ready.
Swap `mock_search` for a real pgvector query once that table exists, without touching the UI
or the generation step.
"""
import json
import os
import re
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(override=True)
for name in ("AKAMAI_BASE_URL", "AKAMAI_API_KEY", "CHAT_MODEL"):
    if not os.environ.get(name):
        sys.exit(f"{name} is not set. Add it to .env, then run: uv run check.py")

client = OpenAI(base_url=os.environ["AKAMAI_BASE_URL"], api_key=os.environ["AKAMAI_API_KEY"])
MODEL = os.environ["CHAT_MODEL"]
HERE = Path(__file__).parent
MARKETS_FILE = HERE / "mock_markets.json"
app = FastAPI()

SYSTEM_PROMPT = (
    "You help find hedge bets on prediction markets. Given a bet someone holds, describe, in one short "
    "sentence, the real-world event that would be the natural OPPOSING bet to hold against it — "
    "something that pays off if the original bet loses. Write it as a plain market question, like "
    "'Any team other than the Lions wins the Super Bowl'. No preamble, no quotes, one sentence only."
)


def generate_opposition(bet: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": bet}],
        max_tokens=80,
    )
    return (response.choices[0].message.content or "").strip()


STOPWORDS = {"the", "a", "an", "is", "in", "of", "to", "for", "on", "at", "wins", "than", "other"}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOPWORDS}


def _overlap_score(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / max(len(wa | wb), 1)


def mock_search(query: str, bet: str, top_k: int = 4, self_match_threshold: float = 0.85):
    """Stand-in for a real pgvector nearest-neighbor search. Scores by shared significant words,
    and drops any market too similar to the ORIGINAL bet (can't hedge a bet against itself).
    A real embedding search would separate these by meaning instead of this word-overlap heuristic
    (e.g. it would know "BTC" and "Bitcoin" are the same asset and still exclude the self-match
    correctly, without this explicit guard). Replace with: embed(query) via EMBEDDING_MODEL, then
    `order by embedding <=> %s limit %s` on a real markets table, excluding the bet's own ticker,
    as 03-database/vectors.py demonstrates for the embedding half."""
    markets = json.loads(MARKETS_FILE.read_text(encoding="utf-8"))
    scored = []
    for m in markets:
        if _overlap_score(m["title"], bet) >= self_match_threshold:
            continue  # too close to the bet itself to count as a hedge
        score = _overlap_score(query, m["title"])
        scored.append({**m, "similarity": round(score, 3)})
    scored.sort(key=lambda m: m["similarity"], reverse=True)
    return scored[:top_k]


class BetRequest(BaseModel):
    bet: str


@app.get("/", response_class=HTMLResponse)
def page():
    return (HERE / "opportunities.html").read_text(encoding="utf-8")


@app.post("/search")
def search(body: BetRequest):
    bet = body.bet.strip()
    if not bet:
        raise HTTPException(400, "Enter a bet first.")
    opposition = generate_opposition(bet)
    candidates = mock_search(opposition, bet)
    return {"bet": bet, "generated_opposition": opposition, "candidates": candidates, "retrieval": "mock"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0" if "PORT" in os.environ else "127.0.0.1", port=int(os.environ.get("PORT", 8001)))
