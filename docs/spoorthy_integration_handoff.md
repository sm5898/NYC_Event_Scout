# Handoff: Live event retrieval integration (@Spoorthy)

The Agent 2 RAG ranking core is on this branch and currently ranks a static `mock_events.json`. Your task is to feed it **live events**. You should not need to touch the embedding, ChromaDB, or ranking logic.

## 1. What the RAG core does now
Given a `PreferenceProfile`, Agent 2 embeds each candidate event on `title + description + type + org` with `all-MiniLM-L6-v2` (384-dim) into the `events` ChromaDB collection (cosine), resolves the user's preference vector (stored `user_preferences` vector when `embedding_id` resolves, else embeds `profile_embedding_seed`), ranks by cosine similarity, attaches `similarity_score = clamp(1 - distance, 0, 1)`, and returns `RankedEvents` sorted high→low for Agent 3. The `POST /agents/event-retriever` contract is unchanged.

## 2. Integration seam (the only wiring change)
In `app/agents/event_retriever.py`, `get_ranked_events()` starts with:
```python
raw_events = _load_raw_events()      # reads mock_events.json
```
**Do not repurpose `_load_raw_events()`** — it is the stub-fallback source. Add a new function and call it instead:
```python
def _load_live_events(profile: PreferenceProfile) -> list[dict]:
    """Live retrieval + normalization -> list of raw event dicts."""
    raw = fetch_live_events(profile)          # your retrieval (uses profile.categories/orgs/raw_text)
    return [_normalize(e) for e in raw]       # dedupe + shape to the fields below

# in get_ranked_events(), replace the first line with:
raw_events = _load_live_events(profile)
```
Everything downstream (embedding, upsert, query, sort) is unchanged.

## 3. Required input fields (per event dict — no defaults; must always be present)
`event_id` (str), `org_id` (str), `title` (str), `date` (str, ISO 8601), `location` (str), `price` (str **or** float), `link` (str).

## 4. Optional fields (safe defaults; include when available)
`description`, `type`, `org` (all str). These three plus `title` form the embedding text, so richer `description`/`type` directly improve ranking. Omitted → treated as `""`.

## 5. Do not set `similarity_score`
The RAG layer computes it; any value you pass is overwritten. Leave it out.

## 6. Expected output after normalization
A `list[dict]`, each dict constructing a valid `Event` (see `app/schemas/models.py`). If any required field is missing or mis-typed, `Event(**data)` raises and the **whole request falls back to the stub** (#9) — validate before returning.

## 7. Duplicate `event_id` handling
**Dedupe by `event_id` in normalization.** ChromaDB `upsert` collapses vectors by id, but the output loop iterates your list as-is, so a repeated `event_id` appears **twice** in the ranked output. Keep the best/first occurrence per id.

## 8. ChromaDB paths (you don't touch Chroma directly)
| Collection | Path | Access |
|---|---|---|
| `user_preferences` | `chroma/` (committed) | **read-only** |
| `events` | `chroma_events/` (gitignored) | writable, rebuilt per request |

Overridable via `USER_PREF_CHROMA_PATH` / `EVENTS_CHROMA_PATH`. Your retrieval returns plain dicts; Agent 2 does all embedding/storage.

## 9. Fallback behavior
- If `_load_live_events` **raises**, or the embedder/Chroma fails → `get_ranked_events` returns `get_stub_events` (`mock_events.json`). Never 500s. Wrap flaky I/O and raise on hard failure rather than returning half-valid dicts.
- If it returns an **empty list** → current behavior returns an **empty** `RankedEvents` (not the stub). If you want stub-on-empty, flag it — don't change it silently.
- Add your own timeout/error handling in retrieval; there is no network timeout in the RAG layer.

## 10. Tests to run after integration
```bash
pytest tests/test_event_retriever.py -q   # the 8 Agent-2 tests must still pass
pytest -q                                 # full suite → expect 13 passed
```
Add: (a) a test patching `_load_live_events` asserting live events flow through and `similarity_score` is **computed** (not your input); (b) a malformed-live-event test asserting stub fallback. Keep tests offline by patching `_load_live_events` / `_load_embedder` / `_load_chroma_client` (existing tests show the pattern).

## 11. Files you'll likely modify
- `app/agents/event_retriever.py` — add `_load_live_events` + swap the one line in `get_ranked_events`.
- New module, e.g. `app/agents/live_events.py` — retrieval + `_normalize`.
- `requirements.txt` — only if retrieval needs new deps (pin versions).
- `tests/test_event_retriever.py` — add the tests above.

## 12. Files to avoid modifying
- `app/schemas/models.py` (locked contract — additive-only, coordinate first).
- `app/main.py` route signatures.
- `_resolve_user_vector`, `_build_events_collection`, `_scores_by_id`, `get_stub_events`, `_load_raw_events` (ranking core + stub path).
- Committed `chroma/` and `storage/` stores.

## 13. Integration checklist
- [ ] Add `_load_live_events(profile)`; swap the one line in `get_ranked_events`.
- [ ] Keep `_load_raw_events()` / `mock_events.json` intact (stub fallback).
- [ ] Populate all required fields (#3); include `description`/`type`/`org` where possible.
- [ ] Do **not** set `similarity_score`.
- [ ] Dedupe by `event_id`.
- [ ] Raise on hard failure (→ stub); handle timeouts inside retrieval.
- [ ] `pytest -q` → 13 passing + new tests green.
- [ ] Confirm no writes to committed `chroma/`; event vectors land in `chroma_events/`.
