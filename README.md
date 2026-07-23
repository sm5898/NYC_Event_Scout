# NYC Event Scout

A multi-agent event discovery system for NYC. This phase implements:

- **Agent 1 — Preference Profiler** (`app/agents/preference_profiler.py`): real
  implementation, built on the approach already prototyped in
  `notebooks/agent1_preference_profiler.ipynb` (Hugging Face `InferenceClient`,
  `publicai` provider). It runs a live DuckDuckGo web search for NYC organizations,
  then calls that same Hugging Face-hosted LLM to turn a user's free text +
  selected interest categories into normalized category weights and a seeded list
  of real NYC organizations grounded in those search results, adapted to the
  locked `PreferenceProfile` schema below.
- **Agent 2 — Event Retriever** (`app/agents/event_retriever.py` +
  `app/agents/live_events.py`): **real, live**. For each org in the user's
  `PreferenceProfile` (or each selected category if Agent 1 fell back to no orgs),
  `live_events.py` runs a real DuckDuckGo search and asks the same Hugging Face
  model Agent 1 uses to extract grounded, real upcoming events (an event's `link`
  must be one of the URLs the search actually returned, or it's dropped).
  `event_retriever.py` then embeds each event on `title + description + type + org`
  with `sentence-transformers/all-MiniLM-L6-v2` (384-dim) into the ChromaDB
  `events` collection, resolves the user's preference vector **hybrid**-style —
  the stored `user_preferences` vector when `PreferenceProfile.embedding_id` is
  set and found, otherwise an on-the-fly embedding of `profile_embedding_seed` —
  and returns `Event`s sorted by cosine `similarity_score`. If live retrieval
  fails (no token, nothing found, malformed event), or `chromadb` /
  `sentence-transformers` aren't installed, or the vector path fails, it degrades
  gracefully to the `app/mocks/mock_events.json` stub (`get_stub_events`).
- A minimal vanilla HTML/CSS/JS frontend that drives both endpoints in sequence.
- Agent 3 (final feed) and the accept/skip signals endpoint are **not implemented
  yet** — only their pydantic schemas exist (`FinalFeed`, `SignalBatch` in
  `app/schemas/models.py`) so the contract is locked for later phases.

## Repo layout

```
app/
├── main.py                        # FastAPI app: API routes + static frontend
├── agents/
│   ├── preference_profiler.py     # Agent 1 — real
│   ├── event_retriever.py         # Agent 2 — RAG core (+ stub fallback)
│   └── live_events.py             # Agent 2 — live retrieval (search + HF extraction)
├── schemas/
│   └── models.py                  # all four shared pydantic schemas
└── mocks/
    └── mock_events.json           # stub-fallback events only (live path doesn't touch this)
frontend/
├── index.html
├── style.css
└── app.js
tests/
├── test_health.py
├── test_preference_profiler.py
├── test_event_retriever.py
└── test_live_events.py
```

`chroma/` at the repo root is a committed ChromaDB store: its `user_preferences`
collection holds the seeded 384-dim preference vector (`pref_test_user_001`) that
Agent 2 reads when a profile carries a matching `embedding_id` (read-only). Agent 2
writes its `events` collection to a separate, gitignored `chroma_events/` store at
query time, so the committed `chroma/` stays pristine. Both paths are overridable
via `USER_PREF_CHROMA_PATH` / `EVENTS_CHROMA_PATH`. `notebooks/`, `storage/`, and
`agents/prompts/` remain earlier prototype artifacts not wired into the `app/`
service.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# then edit .env and set:
#   HF_TOKEN=hf_...

uvicorn app.main:app --reload
```

Then open **http://localhost:8000** — type some interests, check a few categories,
and click "Find Events". You should see a real Agent-1-generated org list (drawn
from a live web search, not invented), followed by real, live events for those
orgs (also grounded in search results, not invented) ranked by similarity to your
profile. The live event pipeline runs one search + one HF call per org, so expect
this step to take significantly longer than Agent 1 — tens of seconds for a
profile with several orgs.

`GET http://localhost:8000/health` should return `{"status": "ok"}`.

## Running tests

```bash
pytest
```

All Hugging Face and search calls are mocked across the suite (`InferenceClient` /
`DDGS` in both `preference_profiler.py` and `live_events.py`) — nothing hits the
network or a real API, so `pytest` works without any token set. 23 tests total:

- `test_preference_profiler.py` (4): profile matches schema shape and merges the
  model's categories/orgs; graceful fallback on an LLM exception, a search
  failure, and a missing `HF_TOKEN`.
- `test_live_events.py` (6): events are grounded/shaped correctly; an event whose
  `link` wasn't actually in the search results is dropped; falls back to
  per-category search when a profile has no orgs; raises (→ stub fallback) with
  no token, nothing to search, or no events extracted.
- `test_event_retriever.py` (12): the RAG ranking core (embedding, hybrid vector
  resolution, cosine scoring, sorting) — unchanged by the live-events work, now
  fed via `_load_live_events`/`_load_raw_events()` depending on the test; plus
  live-events flow through with a **computed** `similarity_score`, malformed live
  events fall back to stub, `_normalize` field validation, and event_id dedupe.
- `test_health.py` (1).

## Where the API key goes

Put it in `.env` at the repo root (never commit this file — it's already in
`.gitignore`):

```
HF_TOKEN=hf_...
```

Get one at https://huggingface.co/settings/tokens. `app/main.py` calls
`load_dotenv()` before anything else reads the environment, so
`uvicorn app.main:app --reload` picks it up automatically.

Optional `HF_PROVIDER` / `HF_MODEL` env vars override the defaults (`publicai` /
`swiss-ai/Apertus-70B-Instruct-2509` — the same combo already validated in the
prototype notebook) if either isn't available under your HF plan.

## What's next (later phases)

- **Date validation for live events**: `live_events.py` asks the model for ISO
  8601 dates but doesn't validate the format — verified live, a non-ISO value
  (`"TBA"`) got through once. Worth adding a real parse-or-drop check.
- **Live event freshness**: since retrieval goes through generic web search
  rather than a live events API/calendar, some extracted events come from stale,
  cached search results (verified live: a couple of 2022/2023-dated events slipped
  through alongside genuinely current ones). A recency check on the date field
  would help.
- **Latency**: live retrieval is one search + one HF call per org (up to
  `MAX_TARGETS = 5`), sequential — a full request can take a minute or more.
  Worth parallelizing (e.g. `concurrent.futures`) if this needs to feel snappier.
- **Agent 1 JSON robustness**: found live — the model occasionally wraps its JSON
  in a ```` ```json ```` fence plus trailing prose commentary despite being told
  not to, which can break the regex-based parser. Same underlying pattern is used
  in `live_events.py`'s extraction, so it's worth hardening both at once (e.g.
  strip fences explicitly before parsing) rather than just retrying.
- **Persisting `embedding_id`**: Agent 1 still doesn't write to the
  `user_preferences` ChromaDB collection or set `PreferenceProfile.embedding_id`,
  so the hybrid "stored vector" path in Agent 2 is only exercised by the one
  seeded `pref_test_user_001` profile, never a real live user.
- **Agent 3**: consumes `RankedEvents` + user history, produces a `FinalFeed`
  (already modeled) with `final_score`/`reason` per event and a
  `best_bets_this_weekend` shortlist.
- **Signals endpoint**: accept `SignalBatch` (accept/skip actions) to feed back into
  Agent 3's ranking — schema is defined, no endpoint yet.

## Judgment calls made this phase

- **LLM provider**: Hugging Face's Inference API doesn't offer a built-in,
  autonomous "web_search" tool the model can call itself the way some other
  providers do. So the search step is done directly in Python — a real, keyless
  DuckDuckGo query via the `ddgs` package — and the results are handed to the LLM
  as grounding context, with an explicit instruction to only select organizations
  that actually appear in those results (not invent names). This is still a real
  search + a real LLM call, just orchestrated client-side rather than via
  agentic tool-calling.
- **Provider + model**: `provider="publicai"` with `swiss-ai/Apertus-70B-Instruct-2509`,
  matching what was already validated working in
  `notebooks/agent1_preference_profiler.ipynb` rather than picking a different
  default. Override with `HF_PROVIDER`/`HF_MODEL` in `.env` if either becomes
  unavailable for your account.
- **Model output parsing**: Agent 1 asks the model to return a single raw JSON
  object rather than relying on structured-output/JSON-schema enforcement, since
  that support varies across HF Inference Providers. The parser first tries a
  direct `json.loads`, then falls back to extracting the first `{...}` block via
  regex.
- **IDs**: `user_id` and each `org_id` are generated server-side with `uuid4` —
  never trusted from the model.
- **Grounding**: for Agent 1's orgs, the prompt is backed by a real keyword filter
  (`_looks_nyc_related`) that drops search results with no NYC signal before the
  model ever sees them. For Agent 2's live events, grounding is a hard,
  mechanical check: an event's `link` must exactly match one of the URLs the
  search actually returned, or the event is dropped — not just a prompt
  instruction.
- **Live events: orgs vs. categories as search targets**: `live_events.py`
  searches per org when `profile.orgs` is non-empty (the common case), and falls
  back to per-*category* search when it's empty (Agent 1's fallback-profile case,
  e.g. no token or a failed LLM call) — so a degraded Agent 1 doesn't also zero
  out Agent 2's ability to find anything.
- **Live events: soft vs. hard required fields**: `title`/`date`/`link` are
  hard-grounded — the whole event is dropped if the model can't point to a real
  one of each in the search results. `location`/`price` are softer: when the
  model doesn't have enough to say, `live_events.py` fills in honest placeholder
  text (`"{org}, New York, NY"` / `"See website"`) rather than dropping the event
  or inventing a specific value — judged a reasonable middle ground between the
  handoff's "no defaults" guidance and not discarding an otherwise-real event over
  a minor field.
- **Live events: deterministic IDs**: unlike Agent 1's `uuid4()` org/user ids,
  live event ids are a hash of `(org_id, title, date)` — so the same real event
  surfacing twice (e.g. from two different org searches) collides and gets
  deduped, rather than appearing twice with different random ids.
- **Graceful degradation surface**: any failure in Agent 1 (missing token, search
  failure, network error, malformed JSON from the model, empty search results)
  falls back to a profile with the user's *selected* categories at weight 1.0 and
  an empty `orgs` list. Any failure in Agent 2's live retrieval (missing token,
  nothing to search, no events extracted, a malformed event) falls back to
  `mock_events.json`. Neither ever 500s.
- **Static file mount order**: the frontend's `StaticFiles` mount is registered
  *after* the `/health` and `/agents/*` routes in `main.py` so it can't shadow them.
- **Mixed `price` types** in `mock_events.json` (string `"Free"`, `int 0`, `float`,
  and `"$18"`) intentionally exercise the `Union[str, float]` price field in the
  schema.
