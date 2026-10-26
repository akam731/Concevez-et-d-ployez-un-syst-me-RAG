import json
from unittest.mock import MagicMock, Mock

import faiss
import numpy as np
import pytest
from httpx import HTTPStatusError, Request, Response

from utils import rag
from utils.database import create_index, save_database


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clear_resources_cache():
    """Empêche qu'un index chargé dans un test soit réutilisé par un autre."""
    rag.load_resources.cache_clear()
    yield
    rag.load_resources.cache_clear()


@pytest.fixture
def database_folder(tmp_path, monkeypatch):
    """Construit les trois fichiers réellement consommés par load_resources."""
    events = [
        {"position": 0, "metadata": {"title_fr": "Concert"}},
        {"position": 1, "metadata": {"title_fr": "Exposition"}},
    ]
    folder = tmp_path / "faiss_events"
    save_database(create_index([[1, 0], [0, 1]]), events, folder)
    monkeypatch.setattr(rag, "datas_folder", tmp_path)
    monkeypatch.setattr(rag, "load_dotenv", lambda *_args: None)
    monkeypatch.setenv("MISTRAL_API", "test-key")
    return folder


def test_embed_uses_mistral_client_without_network(monkeypatch):
    monkeypatch.setattr(rag, "load_dotenv", lambda *_args: None)
    monkeypatch.setenv("MISTRAL_API", "test-key")
    client = MagicMock()
    client.__enter__.return_value.embeddings.create.return_value.data = [
        Mock(embedding=[0.25, 0.75])
    ]
    mistral = Mock(return_value=client)
    monkeypatch.setattr(rag, "Mistral", mistral)

    np.testing.assert_array_equal(rag.embed("concert"), [0.25, 0.75])

    mistral.assert_called_once_with(api_key="test-key")
    client.__enter__.return_value.embeddings.create.assert_called_once_with(
        model="mistral-embed", inputs="concert"
    )


def test_embed_reports_missing_api_key(monkeypatch):
    monkeypatch.setattr(rag, "load_dotenv", lambda *_args: None)
    monkeypatch.delenv("MISTRAL_API", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    report_error = Mock()
    mistral = Mock()
    monkeypatch.setattr(rag.st, "error", report_error)
    monkeypatch.setattr(rag, "Mistral", mistral)

    assert rag.embed("concert") is None
    assert "MISTRAL_API" in report_error.call_args.args[0]
    mistral.assert_not_called()


def test_load_resources_reads_matching_index_and_metadata(database_folder):
    index, events = rag.load_resources()

    assert index.ntotal == 2
    assert index.d == 2
    assert [event["metadata"]["title_fr"] for event in events] == [
        "Concert", "Exposition"
    ]
    assert rag.load_resources()[0] is index


@pytest.mark.parametrize(
    ("filename", "field", "value", "message"),
    [
        ("events.json", None, [], "L'index et les métadonnées"),
        ("manifest.json", "dimension", 3, "L'index et les métadonnées"),
        ("events.json", "position", 1, "pas dans l'ordre"),
        ("manifest.json", "model", "autre-modèle", "mistral-embed"),
    ],
)
def test_load_resources_rejects_inconsistent_database(
    database_folder, filename, field, value, message
):
    path = database_folder / filename
    content = json.loads(path.read_text(encoding="utf-8"))
    if field == "position":
        content[0]["position"] = value
    elif field is None:
        content = value
    else:
        content[field] = value
    path.write_text(json.dumps(content), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        rag.load_resources()


def test_load_resources_requires_api_key(database_folder, monkeypatch):
    monkeypatch.delenv("MISTRAL_API", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    with pytest.raises(ValueError, match="MISTRAL_API"):
        rag.load_resources()


@pytest.mark.parametrize(
    ("question", "k"),
    [(None, 5), ("", 5), ("   ", 5), ("concert", 0), ("concert", True), ("concert", 1.5)],
)
def test_search_events_validates_inputs_before_loading(question, k, monkeypatch):
    load_resources = Mock()
    monkeypatch.setattr(rag, "load_resources", load_resources)

    with pytest.raises(ValueError):
        rag.search_events(question, k)

    load_resources.assert_not_called()


def test_search_events_ranks_matches_and_limits_results(database_folder, monkeypatch):
    embed = Mock(return_value=np.array([2.0, 0.0]))
    monkeypatch.setattr(rag, "embed", embed)

    results = rag.search_events("  concert  ", k=1)

    embed.assert_called_once_with("concert")
    assert len(results) == 1
    assert results[0]["position"] == 0
    assert results[0]["metadata"] == {"title_fr": "Concert"}
    assert results[0]["score"] == pytest.approx(1.0)


@pytest.mark.parametrize("embedding", [None, [1.0], [np.nan, 0.0], [0.0, 0.0]])
def test_search_events_rejects_unusable_embedding(database_folder, monkeypatch, embedding):
    monkeypatch.setattr(rag, "embed", Mock(return_value=embedding))

    with pytest.raises(ValueError, match="embedding"):
        rag.search_events("concert")


def test_search_events_skips_embedding_for_empty_index(monkeypatch):
    monkeypatch.setattr(rag, "load_resources", Mock(return_value=(faiss.IndexFlatIP(2), [])))
    embed = Mock()
    monkeypatch.setattr(rag, "embed", embed)

    assert rag.search_events("concert") == []
    embed.assert_not_called()


def test_built_prompt_keeps_recent_conversation_and_avoids_duplicate_question(
    tmp_path, monkeypatch
):
    prompt_file = tmp_path / "system.md"
    prompt_file.write_text("Réponds en français.\n", encoding="utf-8")
    monkeypatch.setattr(rag, "system_prompt_path", prompt_file)
    search = Mock()
    monkeypatch.setattr(rag, "search_events", search)
    context = [{"position": 0, "metadata": {"title_fr": "Concert"}}]
    messages = [
        {"role": "user", "content": "ancien"},
        {"role": "system", "content": "à exclure"},
        {"role": "assistant", "content": "réponse"},
        {"role": "user", "content": "Que faire ?"},
    ]

    result = rag.built_prompt(
        messages, question="Que faire ?", max_messages=3, context_segments=context
    )

    assert [message["role"] for message in result] == ["system", "assistant", "user"]
    assert result[1:] == messages[-2:]
    assert "Réponds en français." in result[0]["content"]
    assert "Concert" in result[0]["content"]
    assert "à exclure" not in result[0]["content"]
    search.assert_not_called()


def test_built_prompt_searches_when_context_is_unspecified(tmp_path, monkeypatch):
    prompt_file = tmp_path / "system.md"
    prompt_file.write_text("Consigne", encoding="utf-8")
    monkeypatch.setattr(rag, "system_prompt_path", prompt_file)
    search = Mock(return_value=[{"position": 0}])
    monkeypatch.setattr(rag, "search_events", search)

    result = rag.built_prompt([], question="Concert ?")

    search.assert_called_once_with("Concert ?")
    assert result[-1] == {"role": "user", "content": "Concert ?"}
    assert '"position": 0' in result[0]["content"]
    with pytest.raises(ValueError, match="max_messages"):
        rag.built_prompt([], max_messages=0)


def test_ask_invokes_model_with_retrieved_events(monkeypatch):
    events = [{"position": 0, "score": 1.0, "metadata": {"title_fr": "Concert"}}]
    search = Mock(return_value=events)
    prompt = [{"role": "system", "content": "Contexte"}]
    build_prompt = Mock(return_value=prompt)
    model = Mock()
    model.invoke.return_value.content = "Un concert ce soir."
    chat_model = Mock(return_value=model)
    monkeypatch.setattr(rag, "search_events", search)
    monkeypatch.setattr(rag, "built_prompt", build_prompt)
    monkeypatch.setattr(rag, "ChatMistralAI", chat_model)
    monkeypatch.setenv("MISTRAL_API", "test-key")

    result = rag.ask("  Où sortir ?  ", k=1, messages=[{"role": "user", "content": "Salut"}])

    assert result == {"answer": "Un concert ce soir.", "events": events}
    search.assert_called_once_with("  Où sortir ?  ", 1)
    build_prompt.assert_called_once_with(
        [{"role": "user", "content": "Salut"}],
        question="Où sortir ?",
        context_segments=events,
    )
    chat_model.assert_called_once_with(
        model=rag.llm_model,
        api_key="test-key",
        temperature=rag.llm_temperature,
        top_p=rag.llm_top_p,
        max_tokens=rag.llm_max_tokens,
    )
    model.invoke.assert_called_once_with(prompt)


def test_ask_returns_empty_database_message_without_model(monkeypatch):
    monkeypatch.setattr(rag, "search_events", Mock(return_value=[]))
    chat_model = Mock()
    monkeypatch.setattr(rag, "ChatMistralAI", chat_model)

    assert rag.ask("Question") == {
        "answer": "La base ne contient aucun événement.", "events": []
    }
    chat_model.assert_not_called()


@pytest.mark.parametrize(
    ("status", "headers", "expected"),
    [
        (429, {}, "Limite d'utilisation API atteinte."),
        (429, {"x-ratelimit-limit-req-minute": "0"}, "Mistral n'autorise aucune requête"),
        (503, {}, "HTTP 503"),
    ],
)
def test_ask_returns_useful_api_error(monkeypatch, status, headers, expected):
    monkeypatch.setattr(rag, "search_events", Mock(return_value=[{"position": 0}]))
    monkeypatch.setattr(rag, "built_prompt", Mock(return_value=[]))
    request = Request("POST", "https://example.invalid/chat")
    response = Response(status, headers=headers, request=request)
    model = Mock()
    model.invoke.side_effect = HTTPStatusError("API error", request=request, response=response)
    monkeypatch.setattr(rag, "ChatMistralAI", Mock(return_value=model))

    result = rag.ask("Question")

    assert expected in result["answer"]
    assert result["events"] == []


def test_ask_returns_local_error_message(monkeypatch):
    monkeypatch.setattr(rag, "search_events", Mock(side_effect=ValueError("base absente")))

    assert rag.ask("Question") == {
        "answer": "Impossible de répondre : vérifie la base FAISS, la question et l'accès à Mistral.",
        "events": [],
    }
