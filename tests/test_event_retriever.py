"""Offline unit tests for Agent 2 (event_retriever).

Both heavy dependencies — the SentenceTransformer embedder and the ChromaDB
client — are patched with fakes, so these tests never download a model, hit the
network, or require chromadb / sentence-transformers to be installed.
"""

from unittest.mock import patch

import app.agents.event_retriever as er
from app.schemas.models import Event, PreferenceProfile, RankedEvents


# --- fakes --------------------------------------------------------------------


class _FakeEncoding:
    def __init__(self, vector):
        self._vector = vector

    def tolist(self):
        return list(self._vector)


class _FakeEmbedder:
    """Deterministic stand-in for SentenceTransformer; records encoded texts."""

    def __init__(self):
        self.encoded = []

    def encode(self, text):
        self.encoded.append(text)
        # A fixed 3-dim vector is enough — the fake collection controls ranking.
        return _FakeEncoding([0.1, 0.2, 0.3])


class _FakeEventsCollection:
    def __init__(self, distances_by_id):
        self._distances_by_id = distances_by_id
        self.upserted_ids = None
        self.last_query_vector = None

    def upsert(self, ids, embeddings, documents):
        self.upserted_ids = list(ids)

    def query(self, query_embeddings, n_results):
        self.last_query_vector = query_embeddings[0]
        ids = list(self._distances_by_id.keys())[:n_results]
        distances = [self._distances_by_id[i] for i in ids]
        return {"ids": [ids], "distances": [distances]}


class _FakeUserPrefCollection:
    def __init__(self, stored):
        # stored: dict of embedding_id -> vector (or None to simulate "not found")
        self._stored = stored
        self.requested_ids = None

    def get(self, ids, include):
        self.requested_ids = list(ids)
        vectors = [self._stored.get(i) for i in ids]
        return {"embeddings": vectors}


class _FakeClient:
    def __init__(self, events_collection, user_pref_collection=None):
        self._events = events_collection
        self._user_pref = user_pref_collection

    def get_or_create_collection(self, name, metadata=None):
        assert name == er.EVENTS_COLLECTION
        return self._events

    def get_collection(self, name):
        assert name == er.USER_PREF_COLLECTION
        if self._user_pref is None:
            raise ValueError("collection does not exist")
        return self._user_pref


def _profile(**overrides):
    base = dict(
        user_id="user-xyz",
        categories=[],
        orgs=[],
        raw_text="I love quiet jazz and gardens",
        profile_embedding_seed="I love quiet jazz and gardens | arts_culture, parks_outdoors",
    )
    base.update(overrides)
    return PreferenceProfile(**base)


# --- tests --------------------------------------------------------------------


def test_ranks_and_sorts_by_similarity_and_embeds_seed():
    """Without embedding_id: embed the seed, rank all events, sort by score desc,
    and convert cosine distance -> clamped similarity."""
    embedder = _FakeEmbedder()
    # distances chosen so evt_004 (0.02) beats evt_002 (0.10) beats evt_001 (0.50)...
    distances = {
        "evt_001": 0.50,
        "evt_002": 0.10,
        "evt_003": 0.80,
        "evt_004": 0.02,
        "evt_005": 0.30,
    }
    events_coll = _FakeEventsCollection(distances)
    client = _FakeClient(events_coll)

    with patch.object(er, "_load_embedder", return_value=embedder), patch.object(
        er, "_load_chroma_client", return_value=client
    ):
        result = er.get_ranked_events(_profile())

    assert isinstance(result, RankedEvents)
    assert result.user_id == "user-xyz"
    assert len(result.events) == 5

    # Sorted highest-first, matching 1 - distance.
    order = [e.event_id for e in result.events]
    assert order == ["evt_004", "evt_002", "evt_005", "evt_001", "evt_003"]

    scores = {e.event_id: e.similarity_score for e in result.events}
    assert abs(scores["evt_004"] - 0.98) < 1e-6
    assert abs(scores["evt_003"] - 0.20) < 1e-6
    assert all(0.0 <= e.similarity_score <= 1.0 for e in result.events)

    # No embedding_id -> the seed was embedded for the query vector.
    assert _profile().profile_embedding_seed in embedder.encoded
    # All 5 events were upserted into the events collection.
    assert set(events_coll.upserted_ids) == set(distances.keys())


def test_uses_stored_vector_when_embedding_id_present():
    """With a resolvable embedding_id, the stored user_preferences vector is used
    as the query vector rather than an embedding of the seed."""
    embedder = _FakeEmbedder()
    events_coll = _FakeEventsCollection({"evt_001": 0.1, "evt_002": 0.2})
    stored_vector = [0.9, 0.8, 0.7, 0.6]
    user_pref = _FakeUserPrefCollection({"pref_test_user_001": stored_vector})
    client = _FakeClient(events_coll, user_pref)

    with patch.object(er, "_load_embedder", return_value=embedder), patch.object(
        er, "_load_chroma_client", return_value=client
    ):
        result = er.get_ranked_events(_profile(embedding_id="pref_test_user_001"))

    assert isinstance(result, RankedEvents)
    assert user_pref.requested_ids == ["pref_test_user_001"]
    # The stored vector — not a seed embedding — was passed to the events query.
    assert events_coll.last_query_vector == stored_vector


def test_stored_vector_handles_numpy_array_from_chroma():
    """Regression: real ChromaDB returns embeddings as a numpy array, not a list.
    The resolver must not use truthiness on it (which raises 'ambiguous truth
    value') and must still use the stored vector for the query."""
    np = __import__("pytest").importorskip("numpy")

    class _NumpyUserPref:
        # Mimics real chroma: .get returns embeddings as a 2-D numpy array.
        def get(self, ids, include):
            return {"embeddings": np.array([[0.5, 0.4, 0.3, 0.2]])}

    embedder = _FakeEmbedder()
    events_coll = _FakeEventsCollection({"evt_001": 0.1})
    client = _FakeClient(events_coll, _NumpyUserPref())

    with patch.object(er, "_load_embedder", return_value=embedder), patch.object(
        er, "_load_chroma_client", return_value=client
    ):
        result = er.get_ranked_events(_profile(embedding_id="pref_test_user_001"))

    assert isinstance(result, RankedEvents)
    # The numpy vector was converted to a plain list and used as the query vector.
    assert events_coll.last_query_vector == [0.5, 0.4, 0.3, 0.2]
    assert all(isinstance(x, float) for x in events_coll.last_query_vector)


def test_missing_embedding_id_falls_back_to_seed_embedding():
    """embedding_id present but absent from the collection -> embed the seed."""
    embedder = _FakeEmbedder()
    events_coll = _FakeEventsCollection({"evt_001": 0.1})
    user_pref = _FakeUserPrefCollection({"pref_test_user_001": None})  # not found
    client = _FakeClient(events_coll, user_pref)

    with patch.object(er, "_load_embedder", return_value=embedder), patch.object(
        er, "_load_chroma_client", return_value=client
    ):
        result = er.get_ranked_events(_profile(embedding_id="pref_missing"))

    assert isinstance(result, RankedEvents)
    # Query vector is the fake embedder output (seed was embedded).
    assert events_coll.last_query_vector == [0.1, 0.2, 0.3]


def test_falls_back_to_stub_on_error():
    """Any RAG failure (here: client construction raises) degrades to stub events
    with the original mock scores, never a crash."""
    with patch.object(er, "_load_embedder", return_value=_FakeEmbedder()), patch.object(
        er, "_load_chroma_client", side_effect=RuntimeError("chroma unavailable")
    ):
        result = er.get_ranked_events(_profile(user_id="fallback-user"))

    assert isinstance(result, RankedEvents)
    assert result.user_id == "fallback-user"
    assert len(result.events) == 5
    # Stub preserves the hardcoded mock scores rather than computing new ones.
    by_id = {e.event_id: e.similarity_score for e in result.events}
    assert by_id["evt_001"] == 0.91


def test_falls_back_when_deps_not_installed():
    """A missing optional dependency (ImportError from the loader) also degrades."""
    with patch.object(
        er, "_load_embedder", side_effect=ImportError("No module named 'sentence_transformers'")
    ):
        result = er.get_ranked_events(_profile())

    assert isinstance(result, RankedEvents)
    assert len(result.events) == 5


def test_event_embedding_text_uses_contract_fields():
    """The embedded text is title + description + type + org, in that order."""
    text = er.event_embedding_text(
        {
            "title": "Late Night Jazz",
            "description": "intimate set",
            "type": "concert",
            "org": "Village Vanguard",
        }
    )
    assert text == "Late Night Jazz intimate set concert Village Vanguard"


def test_stub_events_still_valid():
    """get_stub_events remains intact and returns valid Events (backward-compat)."""
    result = er.get_stub_events("someone")
    assert isinstance(result, RankedEvents)
    assert len(result.events) == 5
    assert all(isinstance(e, Event) for e in result.events)
