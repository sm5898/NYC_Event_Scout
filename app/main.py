"""FastAPI entrypoint: serves the API plus the static frontend."""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.agents.event_retriever import get_stub_events
from app.agents.preference_profiler import build_preference_profile
from app.schemas.models import (
    PreferenceProfile,
    PreferenceProfilerRequest,
    RankedEvents,
)

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="NYC Event Scout")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/agents/preference-profiler", response_model=PreferenceProfile)
def preference_profiler(request: PreferenceProfilerRequest) -> PreferenceProfile:
    return build_preference_profile(request.raw_text, request.selected_categories)


@app.post("/agents/event-retriever", response_model=RankedEvents)
def event_retriever(profile: PreferenceProfile) -> RankedEvents:
    return get_stub_events(profile.user_id)


# Mounted last so it never shadows the explicit API routes above.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
