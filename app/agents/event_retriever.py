"""Agent 2 — Event Retriever.

STUB for this phase: ignores the incoming preference profile entirely and
returns the contents of mocks/mock_events.json, reshaped into a valid
RankedEvents response so the frontend has something real to render.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.schemas.models import Event, RankedEvents

MOCK_EVENTS_PATH = Path(__file__).resolve().parent.parent / "mocks" / "mock_events.json"


def get_stub_events(user_id: str) -> RankedEvents:
    # TODO: replace with real Agent 2 logic (web_search + embeddings + ChromaDB)
    with open(MOCK_EVENTS_PATH, "r", encoding="utf-8") as f:
        raw_events = json.load(f)

    events = [Event(**item) for item in raw_events]
    return RankedEvents(user_id=user_id, events=events)
