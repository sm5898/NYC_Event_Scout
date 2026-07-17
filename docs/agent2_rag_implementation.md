# Agent 2 — RAG Implementation Log

Status: implemented and real-model verified on branch `feature/agent2-rag`.
Owner: Agent 2 (RAG layer). Not yet committed.

## Hybrid RAG architecture

Agent 2 ranks candidate events by cosine similarity between a **user preference
vector** and each **event embedding**, then returns `Event`s sorted by
`similarity_score` (highest first) for Agent 3.

Flow (`app/agents/event_retriever.py :: get_ranked_events`):

1. Load raw events (currently `app/mocks/mock_events.json`; see integration work below).
2. Load the embedder (`all-MiniLM-L6-v2`) and a ChromaDB client for the events store.
3. Embed each event on `title + description + type + org` and upsert into the
   `events` collection (cosine space).
4. **Resolve the user vector — hybrid:**
   - If `PreferenceProfile.embedding_id` is set **and** found in the
     `user_preferences` collection → use that stored 384-dim vector.
   - Otherwise → embed `profile_embedding_seed` at query time with the same model.
5. Query the `events` collection with the user vector; convert cosine distance to
   `similarity_score = clamp(1 - distance, 0, 1)`.
6. Attach scores, sort descending, return `RankedEvents`.

The endpoint contract is unchanged: `POST /agents/event-retriever` takes a
`PreferenceProfile` and returns `RankedEvents` (route signature untouched).

## Files changed

| File | Change |
| --- | --- |
| `app/schemas/models.py` | `PreferenceProfile.embedding_id: Optional[str] = None`; `Event` gains `description`/`type`/`org` (empty defaults) and a defaulted `similarity_score`. All additive/backward-compatible. |
| `app/agents/event_retriever.py` | Hybrid RAG core (`get_ranked_events`); `get_stub_events` retained as fallback; `USER_PREF_CHROMA_PATH` / `EVENTS_CHROMA_PATH` overrides; numpy-array handling for Chroma reads. |
| `app/main.py` | Endpoint body now calls `get_ranked_events` (decorator/signature/`response_model` unchanged). |
| `app/mocks/mock_events.json` | Added `description`/`type`/`org` per event; existing keys preserved. |
| `requirements.txt` | Added `chromadb==1.5.9`, `sentence-transformers==5.6.0`. |
| `.gitignore` | Added `chroma_events/`. |
| `README.md` | Agent 2 documented as RAG core; `chroma/` reclassified as active read-only store. |
| `tests/test_event_retriever.py` | New — 8 offline tests. |

## ChromaDB collections and path design

Two separate persistent stores keep committed data pristine:

| Collection | Store path | Access | Notes |
| --- | --- | --- | --- |
| `user_preferences` | `chroma/` (committed, tracked) | **read-only** | Holds the seeded 384-dim vector `pref_test_user_001`. Overridable via `USER_PREF_CHROMA_PATH`. |
| `events` | `chroma_events/` (**gitignored**) | read/write | Created/refreshed at query time; `hnsw:space=cosine`. Overridable via `EVENTS_CHROMA_PATH`. |

- Default event-store path: `DEFAULT_EVENTS_CHROMA_PATH = <repo>/chroma_events` (ignored),
  so normal runs never dirty the committed `chroma/`.
- `chromadb==1.5.9` is pinned to match the version that created the committed
  `chroma/` store (on-disk format compatibility).

## Embedding model and dimension

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: **384** (matches the committed `user_preferences` collection and the
  seeded vector).
- Same model used for both user-seed embeddings and event embeddings.

## Fallback behavior (never 500s)

- If the RAG path fails for any reason — `chromadb` / `sentence-transformers` not
  installed, model load failure, Chroma error — `get_ranked_events` logs and
  returns `get_stub_events(...)`, i.e. `mock_events.json` reshaped into valid
  `RankedEvents`. Mirrors Agent 1's graceful-degradation contract.
- Within the hybrid user-vector step: a missing/unreadable `embedding_id` falls
  back to embedding `profile_embedding_seed` (then `raw_text`, then empty string).

## Test coverage and results

`tests/test_event_retriever.py` — 8 offline tests; the embedder and Chroma client
are patched, so no model download / network / heavy deps are required:

- ranks + sorts by similarity and embeds the seed when no `embedding_id`
- uses the stored vector when `embedding_id` resolves
- handles Chroma returning embeddings as a **numpy array** (regression)
- falls back to seed embedding when `embedding_id` is absent from the collection
- falls back to stub on Chroma error
- falls back to stub when optional deps are missing (ImportError)
- `event_embedding_text` uses `title + description + type + org`
- `get_stub_events` still returns valid `Event`s (backward-compat)

Full suite result: **13 passed** (8 Agent 2 + 4 Agent 1 + 1 health). One unrelated
pre-existing Starlette/httpx deprecation warning.

## Real end-to-end verification

Ran once with the **real `all-MiniLM-L6-v2` model** against **temp Chroma paths
only** (committed `chroma/` copied to a temp dir before reading, never opened in
place). 13/13 checks passed.

- **Seed-embedding path:** real 384-dim event vectors created and persisted; docs
  = `title + description + type + org`; cosine ranking semantically sensible — for
  a "quiet jazz + botanical gardens" profile the order was Jazz-in-the-Park →
  Botanic Garden → Sunset Stroll → Late Night Jazz → Photography, scores in
  `[0,1]`, sorted descending.
- **Stored-vector path:** read the seeded `pref_test_user_001` (384-dim) and used
  it as the query vector (no fallback).
- **Default-path check:** with no overrides, Agent 2 wrote to the gitignored
  `chroma_events/`; committed `chroma/` and `storage/` verified unchanged.

Bug found and fixed during verification: real ChromaDB returns embeddings as a
numpy array; the original truthiness check (`... or []`) raised "ambiguous truth
value" and was silently swallowed by the fallback. Fixed with explicit
`is not None` / `len()` checks and a numpy regression test.

## Remaining integration work — live event retrieval (Spoorthy)

Agent 2's ranking core is complete; it currently embeds a static
`mock_events.json`. The remaining integration is to feed it **live events**:

- **Handoff point:** replace `_load_raw_events()` in `app/agents/event_retriever.py`
  (the single seam that returns the list of raw event dicts).
- **Expected event shape** (so embeddings + output validate against `Event`):
  `event_id`, `org_id`, `title`, `date` (ISO 8601), `location`, `price`
  (str or float), `link`, plus the embedding fields `description`, `type`, `org`.
  `similarity_score` is computed by Agent 2 — the producer should not set it.
- **Embedding text** is derived from `title + description + type + org`, so those
  four fields most affect ranking quality; richer `description`/`type` help.
- **Freshness:** the `events` collection is upserted per request keyed by
  `event_id`; live results can be passed straight through without manual cache
  clearing.
- **Related follow-up (Agent 1):** to exercise the stored-vector path for
  freshly-created users (not just the seeded `pref_test_user_001`), Agent 1 should
  persist each user's preference vector to `user_preferences` and set
  `PreferenceProfile.embedding_id`. Until then, live users take the seed-embedding
  path by design.

## Notes / non-secrets

- `HF_TOKEN` (Agent 1) is read from the environment / `.env`; no token values
  appear in code or this log. Agent 2's model is downloaded from the public HF Hub
  and needs no token.
