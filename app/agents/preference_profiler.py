"""Agent 1 — Preference Profiler.

Ports the org-generation approach prototyped in notebooks/agent1_preference_profiler.ipynb
(Hugging Face InferenceClient, publicai provider) into the FastAPI service, adapted to the
locked PreferenceProfile schema (category weights, org_id/source, etc.) and extended with a
live web search grounding step.

Turns a user's free-text description plus their selected interest categories
into a PreferenceProfile: normalized category weights, and a seeded list of
real NYC organizations found via a live DuckDuckGo web search.

LLM calls go through Hugging Face's Inference API (huggingface_hub.InferenceClient)
rather than a provider-hosted agentic tool. Hugging Face's serverless/router
inference doesn't offer a built-in "web_search" tool the way some providers do,
so the search step is done directly in this module (real network calls to
DuckDuckGo, no API key required) and the results are handed to the model as
grounding context, with an instruction to only use organizations that actually
appear in those results.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from typing import Any, Dict, Iterable, List, Optional

from ddgs import DDGS
from huggingface_hub import InferenceClient

from app.schemas.models import Category, Org, PreferenceProfile

logger = logging.getLogger(__name__)

# Same provider + model validated in the original prototype notebook
# (notebooks/agent1_preference_profiler.ipynb). Override via HF_MODEL / HF_PROVIDER
# in .env if either becomes unavailable on your plan.
DEFAULT_HF_PROVIDER = "publicai"
DEFAULT_HF_MODEL = "swiss-ai/Apertus-70B-Instruct-2509"

VALID_CATEGORIES = [
    "arts_culture",
    "parks_outdoors",
    "nightlife_bars",
    "food_restaurants",
    "community_nonprofits",
]

CATEGORY_LABELS = {
    "arts_culture": "arts and culture",
    "parks_outdoors": "parks and outdoors",
    "nightlife_bars": "nightlife and bars",
    "food_restaurants": "food and restaurants",
    "community_nonprofits": "community and nonprofit organizations",
}

SYSTEM_PROMPT = f"""You are the Preference Profiler for NYC Event Scout, a service that curates \
NYC event recommendations for users based on their interests.

You will be given: the user's selected interest categories (a subset of: \
{", ".join(VALID_CATEGORIES)}), free text describing their tastes in their own words, and a \
list of real web search results about NYC organizations/venues relevant to those interests.

Do two things:

1. Assign a weight from 0.0 to 1.0 to every category the user selected, plus any other \
category from that same list of five that the free text strongly implies (even if not \
selected). Higher weight means stronger interest. Weights do not need to sum to 1.

2. Pick 5 to 8 real NYC organizations, venues, or institutions from the search results \
provided to you that match these interests.

STRICT rules for step 2 — an organization must satisfy ALL of these or be excluded:
- It must actually appear in the search results below — never invent a name or use outside \
knowledge not present in the results.
- It must be clearly located in, or headquartered in, New York City — one of the five \
boroughs (Manhattan, Brooklyn, Queens, The Bronx, Staten Island). Read the snippet and URL \
for evidence of this, not just the title.
- If a search result is about a different city, state, or country (for example New Jersey, \
Long Island outside NYC, or anywhere outside the five boroughs), or if it's a generic \
national/international brand, website, or app with no specific NYC location mentioned, you \
MUST exclude it, even if it seems topically related to the user's interests.
- If you are not confident an organization is NYC-based from the evidence given, leave it out.

If the search results don't contain enough clearly NYC-based matches, return fewer organizations \
(even zero) rather than including anything uncertain or out-of-town.

Respond with ONLY a single JSON object and nothing else — no markdown fences, no commentary \
before or after it. It must have exactly this shape:

{{
  "categories": [{{"name": "<category slug>", "weight": <float 0-1>}}, ...],
  "orgs": [{{"name": "<org display name>", "category": "<category slug>"}}, ...]
}}
"""

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

_NYC_SIGNAL_KEYWORDS = (
    "new york city",
    "new york, ny",
    " nyc",
    "nyc.gov",
    "nyc.",
    "manhattan",
    "brooklyn",
    "queens",
    "the bronx",
    "bronx",
    "staten island",
    "harlem",
    "greenwich village",
    "central park",
    ", ny ",
    ", ny,",
)


def _fallback_profile(
    user_id: str, raw_text: str, selected_categories: Iterable[str]
) -> PreferenceProfile:
    """A valid, minimal profile used whenever the search/LLM path is unavailable or fails."""
    categories = [
        Category(name=c, weight=1.0) for c in selected_categories if c in VALID_CATEGORIES
    ]
    if not categories:
        categories = [Category(name="community_nonprofits", weight=0.5)]
    seed = f"{raw_text} | " + ", ".join(c.name for c in categories)
    return PreferenceProfile(
        user_id=user_id,
        categories=categories,
        orgs=[],
        raw_text=raw_text,
        profile_embedding_seed=seed,
    )


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        raise ValueError("no JSON object found in model response")
    return json.loads(match.group(0))


def _looks_nyc_related(result: Dict[str, str]) -> bool:
    haystack = f" {result.get('title', '')} {result.get('snippet', '')} {result.get('url', '')} ".lower()
    return any(keyword in haystack for keyword in _NYC_SIGNAL_KEYWORDS)


def _search_nyc_orgs(
    raw_text: str, selected_categories: Iterable[str], max_results: int = 8
) -> List[Dict[str, str]]:
    """Real, keyless web search (DuckDuckGo) for NYC orgs/venues matching the user's interests."""
    labels = [CATEGORY_LABELS.get(c, c) for c in selected_categories if c in VALID_CATEGORIES]
    query_parts = ["NYC New York City organizations venues for"]
    query_parts.append(" and ".join(labels) if labels else "local events")
    if raw_text:
        query_parts.append(raw_text)
    query = " ".join(query_parts).strip()

    try:
        with DDGS() as ddgs:
            # Over-fetch so filtering out non-NYC results still leaves enough to choose from.
            results = ddgs.text(query, region="us-en", max_results=max_results * 2)
    except Exception:
        logger.exception("web search failed for query=%r", query)
        return []

    all_results = [
        {
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", ""),
        }
        for r in (results or [])
        if r.get("title")
    ]

    # Drop results with no NYC signal in title/snippet/url before they ever reach the
    # model — cheaper and more reliable than relying on prompt instructions alone.
    filtered = [r for r in all_results if _looks_nyc_related(r)]
    if not filtered:
        logger.warning("NYC relevance filter removed all search results for query=%r", query)
        filtered = all_results

    return filtered[:max_results]


def _format_search_results(results: List[Dict[str, str]]) -> str:
    if not results:
        return "(no search results found)"
    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(f"{i}. {r['title']} — {r['snippet']} ({r['url']})")
    return "\n".join(lines)


def _collect_completion_text(completion: Any) -> str:
    choice = completion.choices[0]
    return (choice.message.content or "").strip()


def _parse_categories(parsed: dict, selected_categories: Iterable[str]) -> List[Category]:
    categories: List[Category] = []
    for raw in parsed.get("categories", []) or []:
        name = raw.get("name") if isinstance(raw, dict) else None
        if name not in VALID_CATEGORIES:
            continue
        try:
            weight = float(raw.get("weight", 0.5))
        except (TypeError, ValueError):
            weight = 0.5
        weight = max(0.0, min(1.0, weight))
        categories.append(Category(name=name, weight=weight))

    if not categories:
        categories = [
            Category(name=c, weight=1.0) for c in selected_categories if c in VALID_CATEGORIES
        ]
    if not categories:
        categories = [Category(name="community_nonprofits", weight=0.5)]
    return categories


def _parse_orgs(parsed: dict, fallback_category: str) -> List[Org]:
    orgs: List[Org] = []
    for raw in parsed.get("orgs", []) or []:
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        if not name:
            continue
        category = raw.get("category")
        if category not in VALID_CATEGORIES:
            category = fallback_category
        orgs.append(
            Org(org_id=str(uuid.uuid4()), name=name, category=category, source="seeded")
        )
    return orgs


def build_preference_profile(
    raw_text: str,
    selected_categories: List[str],
    hf_token: Optional[str] = None,
) -> PreferenceProfile:
    """Run Agent 1 for real: a live DuckDuckGo search, then one Hugging Face chat
    completion call grounded on those search results.

    Falls back to a minimal-but-valid profile (empty orgs list) on any failure —
    missing token, network error, malformed model output, etc. — rather than
    raising, so the endpoint never 500s just because search or the model call
    came back empty.
    """
    user_id = str(uuid.uuid4())
    hf_token = hf_token or os.environ.get("HF_TOKEN")

    if not hf_token:
        logger.warning("HF_TOKEN not set; returning fallback preference profile")
        return _fallback_profile(user_id, raw_text, selected_categories)

    try:
        search_results = _search_nyc_orgs(raw_text, selected_categories)

        provider = os.environ.get("HF_PROVIDER", DEFAULT_HF_PROVIDER)
        client = InferenceClient(token=hf_token, provider=provider)
        model = os.environ.get("HF_MODEL", DEFAULT_HF_MODEL)

        user_message = (
            f"Selected categories: {', '.join(selected_categories) or '(none selected)'}\n"
            f"User's own words: {raw_text or '(nothing provided)'}\n\n"
            f"Web search results:\n{_format_search_results(search_results)}"
        )

        completion = client.chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=1024,
        )

        raw_output = _collect_completion_text(completion)
        if not raw_output:
            raise ValueError("model response contained no text content")

        parsed = _extract_json_object(raw_output)
        categories = _parse_categories(parsed, selected_categories)
        orgs = _parse_orgs(parsed, fallback_category=categories[0].name)
        seed = f"{raw_text} | " + ", ".join(c.name for c in categories)

        return PreferenceProfile(
            user_id=user_id,
            categories=categories,
            orgs=orgs,
            raw_text=raw_text,
            profile_embedding_seed=seed,
        )

    except Exception:
        logger.exception("Agent 1 (preference profiler) failed; returning fallback profile")
        return _fallback_profile(user_id, raw_text, selected_categories)
