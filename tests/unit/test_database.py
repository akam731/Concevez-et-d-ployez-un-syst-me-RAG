import json

import numpy as np
import faiss
import pandas as pd
import pytest

from utils.database import build_events, create_index, load_data, save_database

pytestmark = pytest.mark.unit


def test_load_data(tmp_path):
    """Vérifie le chargement des fichiers et la conservation de l'ordre et des codes postaux."""
    (tmp_path / "events_clean.csv").write_text(
        "title_fr;location_postalcode\nConcert;01000\nExposition;75001\n",
        encoding="utf-8",
    )
    np.save(tmp_path / "events_embeddings.npy", np.array([[1, 0], [0, 1]]))

    df, embeddings = load_data(tmp_path)

    assert df.title_fr.tolist() == ["Concert", "Exposition"]
    assert df.location_postalcode.tolist() == ["01000", "75001"]
    np.testing.assert_array_equal(embeddings, [[1, 0], [0, 1]])


def test_load_data_wrong_count(tmp_path):
    """Vérifie qu'un nombre de vecteurs différent du nombre d'événements est refusé."""
    (tmp_path / "events_clean.csv").write_text("title_fr\nConcert\n", encoding="utf-8")
    np.save(tmp_path / "events_embeddings.npy", np.array([[1, 0], [0, 1]]))

    with pytest.raises(ValueError, match="un vecteur par événement"):
        load_data(tmp_path)


def test_build_events():
    """Vérifie les positions, les valeurs manquantes et la sélection des métadonnées."""
    df = pd.DataFrame({
        "title_fr": ["Concert", None], "age_min": [7, 12],
        "colonne_ignoree": ["a", "b"],
    }, index=[10, 20])

    assert build_events(df) == [
        {"position": 0, "metadata": {"title_fr": "Concert", "age_min": "7"}},
        {"position": 1, "metadata": {"title_fr": "", "age_min": "12"}},
    ]


def test_create_index():
    """Vérifie la recherche cosinus sans modifier les embeddings d'origine."""
    embeddings = np.array([[3.0, 4.0], [0.0, 2.0]])
    original = embeddings.copy()

    index = create_index(embeddings)
    scores, positions = index.search(np.array([[0.6, 0.8]], dtype=np.float32), 2)

    assert index.ntotal == 2
    assert index.d == 2
    assert positions.tolist() == [[0, 1]]
    np.testing.assert_allclose(scores, [[1.0, 0.8]], atol=1e-6)
    np.testing.assert_array_equal(embeddings, original)


@pytest.mark.parametrize("embeddings", [[], [1, 2], [[0, 0]], [[np.nan, 1]], [[np.inf, 1]]])
def test_create_index_invalid_vectors(embeddings):
    """Vérifie que les vecteurs vides, mal formés, nuls ou non finis sont refusés."""
    with pytest.raises(ValueError):
        create_index(embeddings)


def test_save_database(tmp_path):
    """Vérifie que les trois fichiers sauvegardés peuvent être relus correctement."""
    index = create_index([[1, 0]])
    events = [{"position": 0, "metadata": {"title_fr": "Musée"}}]
    output_dir = tmp_path / "database"

    save_database(index, events, output_dir)

    loaded_index = faiss.read_index(str(output_dir / "events.faiss"))
    assert loaded_index.ntotal == 1
    np.testing.assert_array_equal(loaded_index.reconstruct(0), [1, 0])
    assert json.loads((output_dir / "events.json").read_text(encoding="utf-8")) == events
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["events"] == 1
    assert manifest["dimension"] == 2
    assert manifest["model"] == "mistral-embed"
    assert manifest["metric"] == "cosine"


def test_save_database_wrong_count(tmp_path):
    """Vérifie qu'on ne sauvegarde pas un index avec des métadonnées manquantes."""
    with pytest.raises(ValueError, match="nombre de vecteurs"):
        save_database(create_index([[1, 0]]), [], tmp_path / "database")
    assert not (tmp_path / "database").exists()
