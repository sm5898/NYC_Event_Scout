"""Agent 2 live event retrieval — turns a PreferenceProfile into real NYC events.

For each organization Agent 1 found (``profile.orgs``), or each selected category
if a profile has no orgs (Agent 1's fallback path), runs a real DuckDuckGo search
for upcoming events and asks the same Hugging Face model/provider Agent 1 uses to
extract structured event fields from those search results. Grounded the same way
Agent 1's org search is: the model may only report events it can point to in the
given search results, and must leave out anything it can't confidently date,
locate, or link to.

This module only does retrieval + raw dict shaping — it returns plain dicts.
Agent 2's RAG core (app/agents/event_retriever.py) owns embedding, ChromaDB
storage, and ranking, and is untouched by this module. Per the integration
contract: this module raises on hard failure (no token, nothing to search, no
events could be extracted) rather than returning something half-valid, so the
caller can let that propagate into the stub fallback.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, List

from ddgs import DDGS
from huggingface_hub import InferenceClient

from app.schemas.models import PreferenceProfile

logger = logging.getLogger(__name__)

# Same provider + model as Agent 1 (notebooks/agent1_preference_profiler.ipynb).
DEFAULT_HF_PROVIDER = "publicai"
DEFAULT_HF_MODEL = "swiss-ai/Apertus-70B-Instruct-2509"
HF_TIMEOUT_SECONDS = 20.0

MAX_TARGETS = 5  # cap orgs/categories searched per request (latency + HF cost)
RESULTS_PER_TARGET = 6
MAX_EVENTS_PER_TARGET = 3

CATEGORY_LABELS = {
    "arts_culture": "arts and culture",
    "parks_outdoors": "parks and outdoors",
    "nightlife_bars": "nightlife and bars",
    "food_restaurants": "food and restaurants",
    "community_nonprofits": "community and nonprofit organizations",
}

EXTRACTION_SYSTEM_PROMPT = f"""You are the Live Event Extractor for NYC Event Scout. Given a \
NYC organization or interest category and a list of real web search results about it, extract \
up to {MAX_EVENTS_PER_TARGET} real, specific upcoming events.

STRICT rules — an event must satisfy ALL of these or be excluded:
- It must come from the search results below — never invent an event, date, price, or link.
- It must have a specific, real date (or clear date range) stated or clearly implied in the \
results. If you can't pin down a real date, do not include the event — do not guess one.
- "link" must be copied exactly from one of the URLs given to you below — never invent or \
modify a URL.
- "price" must be "Free" if a source says the event is free, a specific dollar string like \
"$25" if a source states one, or "See website" if no price is stated anywhere. Never invent a \
specific dollar amount that isn't in the results.
- "location" should be the real NYC address or neighborhood mentioned in the results if \
available, otherwise leave it as an empty string.
- "description" is a one-sentence, factual summary drawn only from the search results (empty \
string if you don't have enough to summarize).

If you cannot confidently extract ANY real event from the given results, return an empty JSON \
array: []

Respond with ONLY a JSON array and nothing else — no markdown fences, no commentary. Each item \
must have exactly this shape:
[{{"title": "...", "date": "...", "location": "...", "price": "...", "link": "...", "description": "..."}}]

"date" must be ISO 8601 (e.g. "2026-07-25T19:00:00-04:00", or "2026-07-25" if no time is given).
"""

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _extract_json_array(text: str) -> list:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_ARRAY_RE.search(text)
    if not match:
        raise ValueError("no JSON array found in model response")
    return json.loads(match.group(0))


def _search_target_events(query: str, max_results: int = RESULTS_PER_TARGET) -> List[Dict[str, str]]:
    """Real, keyless web search (DuckDuckGo) for upcoming events at one org/category."""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, region="us-en", max_results=max_results)
    except Exception:
        logger.exception("live event search failed for query=%r", query)
        return []

    return [
        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
        for r in (results or [])
        if r.get("title") and r.get("href")
    ]


def _format_search_results(results: List[Dict[str, str]]) -> str:
    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(f"{i}. {r['title']} — {r['snippet']} ({r['url']})")
    return "\n".join(lines)


def _collect_completion_text(completion: Any) -> str:
    choice = completion.choices[0]
    return (choice.message.content or "").strip()


def _stable_event_id(org_id: str, title: str, date: str) -> str:
    """Deterministic id from (org_id, title, date) so genuine duplicates collide
    (e.g. the same event surfacing from two different target searches) and get
    caught by event_retriever._load_live_events' dedupe step."""
    digest = hashlib.sha1(f"{org_id}|{title}|{date}".encode("utf-8")).hexdigest()[:16]
    return f"live_{digest}"


def _extract_events_for_target(
    client: InferenceClient,
    model: str,
    label: str,
    org_id: str,
    category: str,
    search_results: List[Dict[str, str]],
) -> List[dict]:
    """One HF call: ask the model to pull real, grounded events out of search_results."""
    if not search_results:
        return []

    user_message = (
        f"Organization / category: {label}\n\n"
        f"Web search results:\n{_format_search_results(search_results)}"
    )

    completion = client.chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=1024,
    )

    raw_output = _collect_completion_text(completion)
    if not raw_output:
        return []

    parsed = _extract_json_array(raw_output)
    if not isinstance(parsed, list):
        return []

    # Grounding check: only trust links the search actually returned.
    valid_links = {r["url"] for r in search_results if r.get("url")}

    events: List[dict] = []
    for item in parsed[:MAX_EVENTS_PER_TARGET]:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        date = item.get("date")
        link = item.get("link")
        # title/date/link are the hard-grounded fields: never defaulted, drop the
        # event entirely if the model didn't give us a real one of each.
        if not title or not date or not link or link not in valid_links:
            continue
        events.append(
            {
                "event_id": _stable_event_id(org_id, title, date),
                "org_id": org_id,
                "title": title,
                "date": date,
                # location/price are soft fields: honest placeholder text rather
                # than an invented specific value when the source didn't say.
                "location": item.get("location") or f"{label}, New York, NY",
                "price": item.get("price") or "See website",
                "link": link,
                "description": item.get("description") or "",
                "type": category,
                "org": label,
            }
        )
    return events


def _build_targets(profile: PreferenceProfile) -> List[Dict[str, str]]:
    """Search targets: one per org if the profile has any, else one per selected
    category (Agent 1's fallback-profile case has categories but no orgs)."""
    targets: List[Dict[str, str]] = []

    if profile.orgs:
        for org in profile.orgs[:MAX_TARGETS]:
            targets.append(
                {
                    "label": org.name,
                    "org_id": org.org_id,
                    "category": org.category,
                    "query": f"{org.name} NYC upcoming events schedule",
                }
            )
    else:
        for category in profile.categories[:MAX_TARGETS]:
            label = CATEGORY_LABELS.get(category.name, category.name)
            query = f"NYC {label} upcoming events {profile.raw_text or ''}".strip()
            targets.append(
                {
                    "label": label,
                    "org_id": f"category_{category.name}",
                    "category": category.name,
                    "query": query,
                }
            )

    return targets


def fetch_live_events(profile: PreferenceProfile) -> List[dict]:
    """Real retrieval: search + LLM extraction per org (or category as a fallback).

    Raises RuntimeError if there's nothing to search for, or if no real events
    could be extracted for any target, so event_retriever._load_live_events can
    let that propagate straight into the stub fallback rather than returning an
    empty-but-not-erroring result.
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not set; cannot fetch live events")

    targets = _build_targets(profile)
    if not targets:
        raise RuntimeError("profile has no orgs or categories; nothing to search for")

    provider = os.environ.get("HF_PROVIDER", DEFAULT_HF_PROVIDER)
    model = os.environ.get("HF_MODEL", DEFAULT_HF_MODEL)
    client = InferenceClient(token=hf_token, provider=provider, timeout=HF_TIMEOUT_SECONDS)

    events: List[dict] = []
    for target in targets:
        results = _search_target_events(target["query"])
        if not results:
            continue
        try:
            events.extend(
                _extract_events_for_target(
                    client, model, target["label"], target["org_id"], target["category"], results
                )
            )
        except Exception:
            logger.exception("event extraction failed for target=%r", target["label"])
            continue

    if not events:
        raise RuntimeError("no live events could be extracted for this profile")

    return events
