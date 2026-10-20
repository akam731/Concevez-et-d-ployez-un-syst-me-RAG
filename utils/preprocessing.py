"""Étapes de préparation partagées par les notebooks et les tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


EVENTS_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    "evenements-publics-openagenda/exports/csv"
)
TEXT_COLUMNS = [
    "title_fr", "description_fr", "longdescription_fr", "keywords_fr",
    "daterange_fr", "location_name", "location_address", "location_district",
    "conditions_fr", "accessibility_label_fr", "age_min", "age_max",
    "location_access_fr",
]


def download_events(city, destination):
    """Télécharge l'export CSV d'une ville et renvoie son chemin."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    params = {"where": f'location_city = "{city}"', "limit": -1}
    with requests.get(EVENTS_URL, params=params, stream=True, timeout=180) as response:
        response.raise_for_status()
        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
    return destination


def filter_events(df, min_date):
    """Garde les événements à partir de min_date, borne comprise, en UTC."""
    result = df.copy()
    result["firstdate_begin"] = pd.to_datetime(
        result["firstdate_begin"], utc=True, errors="coerce"
    )
    return result.loc[result["firstdate_begin"] >= min_date].copy()


def build_event_texts(df, columns=None):
    """Construit les textes d'embedding sans HTML ni valeurs manquantes."""
    columns = TEXT_COLUMNS if columns is None else columns
    rows = df[columns].fillna("").astype(str)
    return [
        "\n".join(
            f"{col}: {BeautifulSoup(value, 'html.parser').get_text(' ', strip=True)}"
            for col, value in row.items()
            if value.strip()
        )
        for _, row in rows.iterrows()
    ]


def save_embeddings(texts, embedding_model, destination, batch_size=50):
    """Génère les vecteurs et sauvegarde après chaque lot réussi."""
    if batch_size <= 0:
        raise ValueError("batch_size doit être positif")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        vectors = embedding_model.embed_documents(texts[i:i + batch_size])
        all_embeddings.extend(vectors)
        np.save(destination, np.asarray(all_embeddings, dtype=np.float32))
    return np.asarray(all_embeddings, dtype=np.float32)