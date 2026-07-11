"""Pydantic schemas shared across all NYC Event Scout agents.

Only two endpoints are live this phase (preference-profiler, event-retriever),
but all four schemas are defined here now so the contract is locked for the
whole team ahead of Agent 3 and the signals endpoint.
"""

from __future__ import annotations

from typing import List, Literal, Union

from pydantic import BaseModel, Field

Price = Union[str, float]


# --- Agent 1 output: preference profile ---------------------------------


class Category(BaseModel):
    name: str
    weight: float = Field(ge=0.0, le=1.0)


class Org(BaseModel):
    org_id: str
    name: str
    category: str
    source: Literal["seeded", "user_added"]


class PreferenceProfile(BaseModel):
    user_id: str
    categories: List[Category]
    orgs: List[Org]
    raw_text: str
    profile_embedding_seed: str


# --- Agent 2 output: ranked events (stubbed this phase) ------------------


class Event(BaseModel):
    event_id: str
    org_id: str
    title: str
    date: str  # ISO 8601
    location: str
    price: Price
    link: str
    similarity_score: float = Field(ge=0.0, le=1.0)


class RankedEvents(BaseModel):
    user_id: str
    events: List[Event]


# --- Agent 3 output: final feed (model only, no endpoint yet) ------------


class FeedItem(BaseModel):
    event_id: str
    title: str
    date: str  # ISO 8601
    location: str
    price: Price
    link: str
    final_score: float
    reason: str


class FinalFeed(BaseModel):
    user_id: str
    generated_at: str  # ISO 8601
    feed: List[FeedItem]
    best_bets_this_weekend: List[str]


# --- Accept/skip signals (model only, no endpoint yet) --------------------


class Signal(BaseModel):
    event_id: str
    action: Literal["accept", "skip"]
    timestamp: str  # ISO 8601


class SignalBatch(BaseModel):
    user_id: str
    signals: List[Signal]


# --- API request bodies ---------------------------------------------------


class PreferenceProfilerRequest(BaseModel):
    raw_text: str = ""
    selected_categories: List[str] = Field(default_factory=list)
