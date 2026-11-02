# Tests des endpoints de l'API

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import main


pytestmark = pytest.mark.functional


def test_rebuild_post(monkeypatch):
    """Test que la route /rebuild fonctionne bien en POST"""
    rebuild = Mock(return_value=SimpleNamespace(ntotal=3))
    monkeypatch.setattr(main.PreProcessing, "pre_processing", rebuild)
    monkeypatch.setattr(main, "load_resources", Mock())

    response = TestClient(main.app).post("/rebuild")

    assert response.status_code == 200
    rebuild.assert_called_once_with()


def test_rebuild_get():
    """Test que la route /rebuild ne fonctionne pas en GET"""
    response = TestClient(main.app).get("/rebuild")

    assert response.status_code == 405


def test_ask_post(monkeypatch):
    """Test que la route /ask fonctionne bien en POST avec une question posée"""
    result = {"answer": "Un concert ce soir.", "events": []}
    monkeypatch.setattr(main, "rag_ask", Mock(return_value=result))

    response = TestClient(main.app).post("/ask", json={"question": "Quels concerts ce soir ?"})

    assert response.status_code == 200
    assert set(response.json()) == {"answer", "events"}
    assert isinstance(response.json()["answer"], str)
    assert isinstance(response.json()["events"], list)


def test_ask_get():
    """Test que la route /ask ne fonctionne pas en GET"""
    response = TestClient(main.app).get("/ask")

    assert response.status_code == 405


def test_ask_post_without_question(monkeypatch):
    """Test que la route /ask ne fonctionne pas même en POST si aucune question posée"""

    monkeypatch.setattr(main, "rag_ask", Mock())

    response = TestClient(main.app).post("/ask", json={})

    assert response.status_code == 422


def test_health_get():
    """Vérifie que GET /health répond avec le code 200."""
    response = TestClient(main.app).get("/health")

    assert response.status_code == 200


def test_metadata_get():
    """Vérifie que GET /metadata répond avec le code 200."""
    response = TestClient(main.app).get("/metadata")

    assert response.status_code == 200
