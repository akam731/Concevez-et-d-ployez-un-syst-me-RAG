"""Création de la base FAISS à partir des embeddings déjà calculés."""

import json
from pathlib import Path

import numpy as np
import faiss
import pandas as pd

from utils.preprocessing import TEXT_COLUMNS


def load_data(data_folder):
    """Charge les événements et leurs embeddings sans changer leur ordre."""
    data_folder = Path(data_folder)
    df = pd.read_csv(
        data_folder / "events_clean.csv", sep=";",
        dtype={"location_postalcode": "string"},
    )
    embeddings = np.load(data_folder / "events_embeddings.npy", allow_pickle=False)
    if embeddings.ndim != 2 or len(df) != embeddings.shape[0]:
        raise ValueError("Il faut un vecteur par événement du CSV.")
    return df, embeddings


def build_events(df):
    """Associe chaque position aux métadonnées définies dans TEXT_COLUMNS."""
    return [
        {
            "position": position,
            "metadata": {
                col: "" if pd.isna(row[col]) else str(row[col])
                for col in TEXT_COLUMNS if col in df.columns
            },
        }
        for position, (_, row) in enumerate(df.iterrows())
    ]


def create_index(embeddings):
    """Normalise une copie des vecteurs et construit un index de similarité cosinus."""
    vectors = np.array(embeddings, dtype=np.float32, order="C", copy=True)
    if vectors.ndim != 2 or 0 in vectors.shape:
        raise ValueError("Les embeddings doivent former un tableau 2D non vide.")
    if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("Les vecteurs doivent être finis et non nuls.")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_database(index, events, output_dir):
    """Sauvegarde l'index, les métadonnées et la configuration dans trois fichiers."""
    if index.ntotal != len(events):
        raise ValueError("Le nombre de vecteurs doit correspondre au nombre d'événements.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "events.faiss"))
    (output_dir / "events.json").write_text(
        json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    manifest = {
        "model": "mistral-embed", "events": len(events), "dimension": index.d,
        "index_type": "IndexFlatIP", "metric": "cosine", "unit": "event",
        "source_csv": "events_clean.csv", "source_embeddings": "events_embeddings.npy",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
