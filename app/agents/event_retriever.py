"""Agent 2 — Event Retriever (Hybrid RAG core + live retrieval).

Ranks events for a user by cosine similarity between the user's preference vector
and each event's embedding.

Candidate events come from ``_load_live_events`` (app/agents/live_events.py): real,
grounded events found via web search per org/category in the profile. Any failure
there (no token, nothing found, malformed event) raises and the whole request
falls back to ``get_stub_events`` (``app/mocks/mock_events.json``) — see
``_load_raw_events``, kept intact as that fallback's data source.

User-vector resolution is HYBRID:
  1. If ``profile.embedding_id`` is set AND found in the ChromaDB ``user_preferences``
     collection, use that precomputed 384-dim vector (e.g. "pref_test_user_001").
  2. Otherwise embed ``profile.profile_embedding_seed`` at query time with
     all-MiniLM-L6-v2.

Events are embedded on ``title + description + type + org``, stored in the
ChromaDB ``events`` collection (cosine space), then queried against the user
vector. ``similarity_score = clamp(1 - cosine_distance, 0, 1)`` is attached to
each Event, and events are returned sorted by that score (highest first) for
Agent 3.

Any failure in the RAG path — live retrieval, missing optional deps (chromadb /
sentence-transformers), model download failure, Chroma error — degrades
gracefully to ``get_stub_events`` so the endpoint never 500s. Heavy imports are
done lazily inside the loader functions so importing this module (and the stub
path) works even when those packages aren't installed.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

from app.agents.live_events import fetch_live_events
from app.schemas.models import Event, PreferenceProfile, RankedEvents

logger = logging.getLogger(__name__)

# Fields every live event dict must carry — no defaults; a missing one drops the
# whole request to the stub fallback (see _normalize).
_REQUIRED_LIVE_EVENT_FIELDS = ("event_id", "org_id", "title", "date", "location", "price", "link")

APP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = APP_DIR.parent
MOCK_EVENTS_PATH = APP_DIR / "mocks" / "mock_events.json"
# Committed store holding the seeded `user_preferences` vector. Agent 2 only
# reads from it; it never writes here.
CHROMA_PATH = REPO_ROOT / "chroma"
# Separate, gitignored store for the event vectors Agent 2 writes at query time,
# so the committed chroma/ above stays pristine.
DEFAULT_EVENTS_CHROMA_PATH = REPO_ROOT / "chroma_events"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
USER_PREF_COLLECTION = "user_preferences"
EVENTS_COLLECTION = "events"

# Cached embedder; loaded once on first real use.
_embedder = None


def _user_pref_chroma_path() -> str:
    """Path of the ChromaDB store to read the user-preference vector from.

    Defaults to the committed ``chroma/``. Override with USER_PREF_CHROMA_PATH.
    """
    return os.environ.get("USER_PREF_CHROMA_PATH", str(CHROMA_PATH))


def _events_chroma_path() -> str:
    """Path of the ChromaDB store to write/read the ``events`` collection.

    Defaults to a gitignored ``chroma_events/`` dir (separate from the committed
    ``chroma/``). Override with EVENTS_CHROMA_PATH (e.g. a temp dir in tests).
    """
    return os.environ.get("EVENTS_CHROMA_PATH", str(DEFAULT_EVENTS_CHROMA_PATH))


# --- lazy loaders (patched in tests; keep heavy imports out of module import) -


def _load_embedder():
    """Return a cached all-MiniLM-L6-v2 SentenceTransformer (imported lazily)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _load_chroma_client(path: str):
    """Return a persistent ChromaDB client rooted at ``path`` (imported lazily)."""
    import chromadb

    return chromadb.PersistentClient(path=str(path))


# --- helpers ------------------------------------------------------------------


def _load_raw_events() -> List[dict]:
    with open(MOCK_EVENTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize(item: dict) -> dict:
    """Validate + shape one live-event dict to the Event contract's required fields.

    Raises ValueError if a required field is missing/empty so a malformed live
    event fails fast here in ``_load_live_events`` rather than surfacing later as
    a confusing ``Event(**data)`` error — either way it propagates up and the
    whole request falls back to the stub, per the integration contract.
    """
    normalized = {}
    for key in _REQUIRED_LIVE_EVENT_FIELDS:
        value = item.get(key)
        if value in (None, ""):
            raise ValueError(f"live event missing required field {key!r}: {item!r}")
        normalized[key] = value
    for key in ("description", "type", "org"):
        normalized[key] = item.get(key) or ""
    return normalized


def _load_live_events(profile: PreferenceProfile) -> List[dict]:
    """Live retrieval + normalization -> list of raw event dicts.

    This is the integration seam: swap-in for ``_load_raw_events()`` in
    ``get_ranked_events``. ``_load_raw_events`` itself is untouched and remains
    the stub-fallback source (see ``get_stub_events``).
    """
    raw = fetch_live_events(profile)  # live retrieval — uses profile.orgs/categories/raw_text
    normalized = [_normalize(e) for e in raw]

    # Dedupe by event_id (keep first occurrence) — a repeated id would otherwise
    # appear twice in the ranked output even though Chroma upsert collapses it.
    seen = set()
    deduped = []
    for item in normalized:
        if item["event_id"] in seen:
            continue
        seen.add(item["event_id"])
        deduped.append(item)
    return deduped


def event_embedding_text(item: dict) -> str:
    """The text Agent 2 embeds for an event: title + description + type + org."""
    parts = [
        item.get("title", ""),
        item.get("description", ""),
        item.get("type", ""),
        item.get("org", ""),
    ]
    return " ".join(p for p in parts if p).strip()


def _embed(embedder, text: str) -> List[float]:
    return embedder.encode(text).tolist()


def _resolve_user_vector(profile: PreferenceProfile, embedder) -> List[float]:
    """Hybrid: stored ``user_preferences`` vector by embedding_id, else embed seed."""
    embedding_id = getattr(profile, "embedding_id", None)
    if embedding_id:
        try:
            client = _load_chroma_client(_user_pref_chroma_path())
            collection = client.get_collection(USER_PREF_COLLECTION)
            got = collection.get(ids=[embedding_id], include=["embeddings"])
            embeddings = got.get("embeddings")
            # ChromaDB returns embeddings as a numpy array — never use truthiness
            # (`or []`, `if embeddings`) on it; that raises "ambiguous truth value".
            if embeddings is not None and len(embeddings) > 0:
                vector = embeddings[0]
                if vector is not None and len(vector) > 0:
                    logger.info("Agent 2: using stored user vector %r", embedding_id)
                    return [float(x) for x in vector]
            logger.info(
                "Agent 2: embedding_id %r not in %s; embedding seed instead",
                embedding_id,
                USER_PREF_COLLECTION,
            )
        except Exception:
            logger.exception(
                "Agent 2: failed reading stored vector %r; embedding seed instead",
                embedding_id,
            )

    seed = profile.profile_embedding_seed or profile.raw_text or ""
    return _embed(embedder, seed)


def _build_events_collection(client, embedder, raw_events: List[dict]):
    """Create/refresh the ``events`` collection (cosine) and upsert event vectors."""
    collection = client.get_or_create_collection(
        name=EVENTS_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    ids = [str(e["event_id"]) for e in raw_events]
    documents = [event_embedding_text(e) for e in raw_events]
    embeddings = [_embed(embedder, doc) for doc in documents]
    collection.upsert(ids=ids, embeddings=embeddings, documents=documents)
    return collection


def _scores_by_id(collection, user_vector: List[float], n: int) -> dict:
    """Query the events collection and map event_id -> clamped cosine similarity."""
    result = collection.query(query_embeddings=[user_vector], n_results=n)
    ids = result["ids"][0]
    distances = result["distances"][0]
    scores = {}
    for event_id, distance in zip(ids, distances):
        similarity = 1.0 - float(distance)  # cosine distance -> similarity
        scores[str(event_id)] = max(0.0, min(1.0, similarity))
    return scores


# --- public API ---------------------------------------------------------------


def get_ranked_events(profile: PreferenceProfile) -> RankedEvents:
    """Agent 2 for real: embed events, resolve the user vector (hybrid), rank by
    cosine similarity, and return Events sorted by ``similarity_score`` desc.

    Falls back to :func:`get_stub_events` on any failure so the endpoint never 500s.
    """
    try:
        raw_events = _load_live_events(profile)
        if not raw_events:
            return RankedEvents(user_id=profile.user_id, events=[])

        embedder = _load_embedder()
        events_client = _load_chroma_client(_events_chroma_path())

        collection = _build_events_collection(events_client, embedder, raw_events)
        user_vector = _resolve_user_vector(profile, embedder)
        scores = _scores_by_id(collection, user_vector, n=len(raw_events))

        events: List[Event] = []
        for item in raw_events:
            data = dict(item)
            data["similarity_score"] = scores.get(str(item["event_id"]), 0.0)
            events.append(Event(**data))

        events.sort(key=lambda e: e.similarity_score, reverse=True)
        return RankedEvents(user_id=profile.user_id, events=events)

    except Exception:
        logger.exception("Agent 2 RAG path failed; falling back to stub events")
        return get_stub_events(profile.user_id)


def get_stub_events(user_id: str) -> RankedEvents:
    """Fallback: return mocks/mock_events.json reshaped into a valid RankedEvents.

    Kept as a safety net for the RAG path (missing deps, Chroma/model failure).
    """
    raw_events = _load_raw_events()
    events = [Event(**item) for item in raw_events]
    return RankedEvents(user_id=user_id, events=events)
