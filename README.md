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
- **Agent 2 — Event Retriever** (`app/agents/event_retriever.py`): **stub only**.
  Ignores the profile it's given and returns the contents of
  `app/mocks/mock_events.json`, reshaped into a valid `RankedEvents` response.
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
│   └── event_retriever.py         # Agent 2 — stub
├── schemas/
│   └── models.py                  # all four shared pydantic schemas
└── mocks/
    └── mock_events.json           # sample data for the Agent 2 stub
frontend/
├── index.html
├── style.css
└── app.js
tests/
├── test_health.py
└── test_preference_profiler.py
```

`notebooks/`, `chroma/`, `storage/`, and `agents/prompts/` at the repo root are
leftover artifacts from an earlier exploratory prototype and are unrelated to the
`app/` service built this phase — left in place, not wired into anything here.

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
from a live web search, not invented), followed by the stubbed event list.

`GET http://localhost:8000/health` should return `{"status": "ok"}`.

## Running tests

```bash
pytest
```

Agent 1's tests **mock both the Hugging Face client and the search call**
(`app.agents.preference_profiler.InferenceClient` / `.DDGS`) — they never hit the
network or a real API, so `pytest` works without any token set. Tests cover:

1. The assembled profile matches the `PreferenceProfile` schema shape and correctly
   merges the model's categories/orgs.
2. A raised exception from the Hugging Face call is caught and a valid profile with
   an empty `orgs` list is returned instead of crashing.
3. Same graceful fallback when the web search call itself fails.
4. Same graceful fallback when no `HF_TOKEN` is configured at all.

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

- **Agent 2, for real**: replace `event_retriever.py`'s stub with actual search
  calls per organization, embeddings for the user's `profile_embedding_seed`, and a
  ChromaDB similarity search to produce `similarity_score`. The `# TODO` comment in
  that file marks the spot.
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
- **Grounding is a soft constraint**: the prompt tells the model to only use orgs
  present in the search results, but this isn't mechanically enforced against the
  search text afterward (no substring-match filter). For a class project this was
  judged to be a reasonable tradeoff — a strict filter would risk dropping real
  orgs the model referenced with slightly different phrasing than the search
  snippet.
- **Graceful degradation surface**: any failure in Agent 1 (missing token, search
  failure, network error, malformed JSON from the model, empty search results)
  falls back to a profile with the user's *selected* categories at weight 1.0 and
  an empty `orgs` list — never a 500.
- **Static file mount order**: the frontend's `StaticFiles` mount is registered
  *after* the `/health` and `/agents/*` routes in `main.py` so it can't shadow them.
- **Mixed `price` types** in `mock_events.json` (string `"Free"`, `int 0`, `float`,
  and `"$18"`) intentionally exercise the `Union[str, float]` price field in the
  schema.
