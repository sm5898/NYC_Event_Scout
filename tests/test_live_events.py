"""Offline unit tests for Agent 2's live event retrieval (app/agents/live_events.py).

Both the search call and the Hugging Face client are mocked — these tests never
hit the network or a real API.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.agents.live_events import fetch_live_events
from app.schemas.models import Category, Org, PreferenceProfile


def _fake_completion(text: str) -> SimpleNamespace:
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _search_hit(title: str, url: str, snippet: str) -> dict:
    return {"title": title, "href": url, "body": snippet}


def _profile(**overrides) -> PreferenceProfile:
    base = dict(
        user_id="user-1",
        categories=[Category(name="arts_culture", weight=0.8)],
        orgs=[
            Org(
                org_id="org_met",
                name="The Metropolitan Museum of Art",
                category="arts_culture",
                source="seeded",
            )
        ],
        raw_text="I love museums",
        profile_embedding_seed="I love museums | arts_culture",
    )
    base.update(overrides)
    return PreferenceProfile(**base)


@patch("app.agents.live_events.InferenceClient")
@patch("app.agents.live_events.DDGS")
def test_fetch_live_events_grounds_and_shapes_events(mock_ddgs_cls, mock_inference_cls):
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = [
        _search_hit(
            "Met Museum: Free Friday Nights",
            "https://www.metmuseum.org/events/free-friday",
            "Free evening access every Friday.",
        )
    ]
    mock_ddgs_cls.return_value.__enter__.return_value = mock_ddgs_instance

    mock_client = MagicMock()
    mock_inference_cls.return_value = mock_client
    mock_client.chat_completion.return_value = _fake_completion(
        json.dumps(
            [
                {
                    "title": "Free Friday Nights",
                    "date": "2026-07-25T18:00:00-04:00",
                    "location": "The Met, New York, NY",
                    "price": "Free",
                    "link": "https://www.metmuseum.org/events/free-friday",
                    "description": "Free evening access.",
                }
            ]
        )
    )

    events = fetch_live_events(_profile())

    assert len(events) == 1
    event = events[0]
    assert event["org_id"] == "org_met"
    assert event["title"] == "Free Friday Nights"
    assert event["link"] == "https://www.metmuseum.org/events/free-friday"
    assert event["type"] == "arts_culture"
    assert event["org"] == "The Metropolitan Museum of Art"
    assert event["event_id"].startswith("live_")
    mock_client.chat_completion.assert_called_once()


@patch("app.agents.live_events.InferenceClient")
@patch("app.agents.live_events.DDGS")
def test_fetch_live_events_drops_events_with_unlisted_links(mock_ddgs_cls, mock_inference_cls):
    """Grounding: an event whose link wasn't actually in the search results is
    dropped, even though its shape is otherwise valid."""
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = [
        _search_hit("Met Museum Events", "https://www.metmuseum.org/events", "Upcoming events.")
    ]
    mock_ddgs_cls.return_value.__enter__.return_value = mock_ddgs_instance

    mock_client = MagicMock()
    mock_inference_cls.return_value = mock_client
    mock_client.chat_completion.return_value = _fake_completion(
        json.dumps(
            [
                {
                    "title": "Made Up Gala",
                    "date": "2026-08-01",
                    "location": "",
                    "price": "",
                    "link": "https://not-a-real-search-result.example.com/gala",
                    "description": "",
                }
            ]
        )
    )

    # No event survives grounding -> no events anywhere -> raises (stub fallback).
    with pytest.raises(RuntimeError):
        fetch_live_events(_profile())


@patch("app.agents.live_events.InferenceClient")
@patch("app.agents.live_events.DDGS")
def test_fetch_live_events_falls_back_to_categories_when_no_orgs(mock_ddgs_cls, mock_inference_cls):
    """Agent 1's fallback profiles have categories but no orgs — live retrieval
    should still search by category rather than finding nothing to do."""
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = [
        _search_hit(
            "NYC Parks Events", "https://www.nycgovparks.org/events", "Upcoming park events."
        )
    ]
    mock_ddgs_cls.return_value.__enter__.return_value = mock_ddgs_instance

    mock_client = MagicMock()
    mock_inference_cls.return_value = mock_client
    mock_client.chat_completion.return_value = _fake_completion(
        json.dumps(
            [
                {
                    "title": "Central Park Cleanup",
                    "date": "2026-07-27",
                    "location": "Central Park, New York, NY",
                    "price": "Free",
                    "link": "https://www.nycgovparks.org/events",
                    "description": "Volunteer cleanup event.",
                }
            ]
        )
    )

    events = fetch_live_events(_profile(orgs=[]))

    assert len(events) == 1
    assert events[0]["org_id"] == "category_arts_culture"


@patch.dict("os.environ", {}, clear=True)
def test_fetch_live_events_raises_without_hf_token():
    with pytest.raises(RuntimeError):
        fetch_live_events(_profile())


@patch.dict("os.environ", {"HF_TOKEN": "test-token"})
def test_fetch_live_events_raises_when_nothing_to_search():
    profile = _profile(orgs=[], categories=[])
    with pytest.raises(RuntimeError):
        fetch_live_events(profile)


@patch("app.agents.live_events.InferenceClient")
@patch("app.agents.live_events.DDGS")
def test_fetch_live_events_raises_when_extraction_yields_nothing(mock_ddgs_cls, mock_inference_cls):
    """Search succeeds but the model can't confidently extract any event -> raise
    (never returns an empty list silently)."""
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = [
        _search_hit("Met Museum", "https://www.metmuseum.org/", "General museum info.")
    ]
    mock_ddgs_cls.return_value.__enter__.return_value = mock_ddgs_instance

    mock_client = MagicMock()
    mock_inference_cls.return_value = mock_client
    mock_client.chat_completion.return_value = _fake_completion("[]")

    with pytest.raises(RuntimeError):
        fetch_live_events(_profile())
