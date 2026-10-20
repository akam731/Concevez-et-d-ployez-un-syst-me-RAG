from pathlib import Path
from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest

from utils.preprocessing import (
    EVENTS_URL, TEXT_COLUMNS, build_event_texts, download_events,
    filter_events, save_embeddings,
)
from utils import settings

pytestmark = pytest.mark.unit


def test_settings():
    """Vérifie l'accessibilité des paramètres du programme"""
    assert settings.datas_folder == Path(settings.__file__).resolve().parents[1] / "datas"
    assert settings.city == "Paris"
    assert settings.min_date == pd.Timestamp("2025-10-10", tz="UTC")


def test_compares_timezones():
    """Vérifie que les dates sont comparées en tenant compte des fuseaux horaires."""
    df = pd.DataFrame({"firstdate_begin": [
        "2025-10-10T01:00:00+02:00", "2025-10-10T02:00:00+02:00",
    ]})
    assert filter_events(df, settings.min_date).index.tolist() == [1]


def test_remove_html():
    """Vérifie le retrait des balises HTML et le traitement des valeurs absentes ou numériques."""
    df = pd.DataFrame({
        "title_fr": ["<p>Musée &amp; concert</p>", None],
        "age_min": [7, None],
        "description_fr": ["   ", "<b>Bonjour</b><p>Paris</p>"],
    })
    assert build_event_texts(df, list(df.columns)) == [
        "title_fr: Musée & concert\nage_min: 7.0",
        "description_fr: Bonjour Paris",
    ]


def test_columns():
    """Vérifie les colonnes par défaut et les données vides lors de la création des textes."""
    df = pd.DataFrame([{col: None for col in TEXT_COLUMNS}])
    assert build_event_texts(df) == [""]
    assert build_event_texts(df.iloc[:0]) == []


def test_missing_column():
    """Vérifie qu'une colonne absente déclenche une erreur."""
    with pytest.raises(KeyError):
        build_event_texts(pd.DataFrame({"title_fr": ["Concert"]}))


def test_download(tmp_path, monkeypatch):
    """Vérifie qu'un téléchargement simulé crée le CSV avec le bon contenu."""
    response = MagicMock()
    response.__enter__.return_value = response
    response.iter_content.return_value = [b"uid;title_fr\n", b"", b"1;Concert\n"]
    get = Mock(return_value=response)
    monkeypatch.setattr("utils.preprocessing.requests.get", get)

    destination = tmp_path / "events.csv"
    assert download_events("Paris", destination) == destination
    assert destination.read_bytes() == b"uid;title_fr\n1;Concert\n"
    assert get.call_args.kwargs["params"]["where"] == 'location_city = "Paris"'
    response.raise_for_status.assert_called_once_with()


def test_save_embeddings(tmp_path):
    """Vérifie que les vecteurs générés par lots sont sauvegardés en float32."""
    model = Mock()
    model.embed_documents.side_effect = [[[1, 2]], [[3, 4]]]
    destination = tmp_path / "embeddings.npy"

    save_embeddings(["Concert", "Musée"], model, destination, batch_size=1)

    vectors = np.load(destination)
    np.testing.assert_array_equal(vectors, [[1, 2], [3, 4]])
    assert vectors.dtype == np.float32
    assert [call.args[0] for call in model.embed_documents.call_args_list] == [
        ["Concert"], ["Musée"],
    ]


def test_invalid_batch_size(tmp_path):
    """Vérifie qu'une taille de lot nulle est refusée."""
    with pytest.raises(ValueError):
        save_embeddings(["Concert"], Mock(), tmp_path / "embeddings.npy", batch_size=0)
