import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.agents.preference_profiler import build_preference_profile
from app.schemas.models import PreferenceProfile


def _fake_completion(text: str) -> SimpleNamespace:
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _fake_search_results():
    return [
        {
            "title": "The Metropolitan Museum of Art",
            "href": "https://www.metmuseum.org/",
            "body": "Encyclopedic art museum on Fifth Avenue.",
        },
        {
            "title": "The High Line",
            "href": "https://www.thehighline.org/",
            "body": "Elevated park built on a former rail line.",
        },
    ]


@patch("app.agents.preference_profiler.InferenceClient")
@patch("app.agents.preference_profiler.DDGS")
def test_profile_output_matches_schema_shape(mock_ddgs_cls, mock_inference_cls):
    """Agent 1's output should assemble into a valid PreferenceProfile with the
    model's categories/orgs merged in. Both the search engine and the LLM client
    are mocked — this never touches the network or a real API."""
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = _fake_search_results()
    mock_ddgs_cls.return_value.__enter__.return_value = mock_ddgs_instance

    mock_client = MagicMock()
    mock_inference_cls.return_value = mock_client
    mock_client.chat_completion.return_value = _fake_completion(
        json.dumps(
            {
                "categories": [
                    {"name": "arts_culture", "weight": 0.9},
                    {"name": "parks_outdoors", "weight": 0.4},
                ],
                "orgs": [
                    {
                        "name": "The Metropolitan Museum of Art",
                        "category": "arts_culture",
                    },
                    {"name": "The High Line", "category": "parks_outdoors"},
                ],
            }
        )
    )

    profile = build_preference_profile(
        "I love museums and quiet walks", ["arts_culture"], hf_token="test-token"
    )

    # Confirms output matches the schema shape (round-trips through pydantic).
    assert isinstance(profile, PreferenceProfile)
    validated = PreferenceProfile.model_validate(profile.model_dump())

    assert validated.user_id
    assert validated.raw_text == "I love museums and quiet walks"
    assert {c.name for c in validated.categories} == {"arts_culture", "parks_outdoors"}
    assert all(0.0 <= c.weight <= 1.0 for c in validated.categories)
    assert len(validated.orgs) == 2
    assert all(org.source == "seeded" for org in validated.orgs)
    assert all(org.org_id for org in validated.orgs)
    assert validated.profile_embedding_seed

    mock_ddgs_instance.text.assert_called_once()
    mock_client.chat_completion.assert_called_once()


@patch("app.agents.preference_profiler.InferenceClient")
@patch("app.agents.preference_profiler.DDGS")
def test_falls_back_gracefully_when_api_call_fails(mock_ddgs_cls, mock_inference_cls):
    """If the Hugging Face call raises (network error, bad response, etc.), Agent 1
    should still return a valid profile with an empty orgs list instead of crashing."""
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = _fake_search_results()
    mock_ddgs_cls.return_value.__enter__.return_value = mock_ddgs_instance

    mock_client = MagicMock()
    mock_inference_cls.return_value = mock_client
    mock_client.chat_completion.side_effect = RuntimeError("simulated API failure")

    profile = build_preference_profile(
        "I love parks", ["parks_outdoors"], hf_token="test-token"
    )

    assert isinstance(profile, PreferenceProfile)
    assert profile.user_id
    assert profile.raw_text == "I love parks"
    assert profile.orgs == []
    assert len(profile.categories) == 1
    assert profile.categories[0].name == "parks_outdoors"


@patch("app.agents.preference_profiler.InferenceClient")
@patch("app.agents.preference_profiler.DDGS")
def test_falls_back_gracefully_when_search_fails(mock_ddgs_cls, mock_inference_cls):
    """A broken/rate-limited search should also degrade gracefully rather than raise."""
    mock_ddgs_cls.return_value.__enter__.side_effect = RuntimeError("search unavailable")

    mock_client = MagicMock()
    mock_inference_cls.return_value = mock_client
    mock_client.chat_completion.return_value = _fake_completion(
        json.dumps({"categories": [{"name": "food_restaurants", "weight": 0.7}], "orgs": []})
    )

    profile = build_preference_profile(
        "tacos please", ["food_restaurants"], hf_token="test-token"
    )

    assert isinstance(profile, PreferenceProfile)
    assert profile.orgs == []
    assert profile.categories[0].name == "food_restaurants"


@patch.dict("os.environ", {}, clear=True)
def test_falls_back_when_no_hf_token_present():
    """No token configured should also degrade gracefully rather than raise."""
    profile = build_preference_profile("no token here", ["nightlife_bars"], hf_token=None)

    assert isinstance(profile, PreferenceProfile)
    assert profile.orgs == []
    assert profile.categories[0].name == "nightlife_bars"
