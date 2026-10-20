"""Vectorisation des événements nettoyés, sans les affichages du notebook."""

import os

import pandas as pd
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings

from utils.preprocessing import build_event_texts, save_embeddings
from utils.settings import datas_folder


def main():
    """Génère et sauvegarde les embeddings avec les paramètres du notebook 2."""
    # Charger les événements dans l'ordre qui sera utilisé pour FAISS.
    df = pd.read_csv(
        datas_folder / "events_clean.csv", sep=";",
        dtype={"location_postalcode": "string"},
    )
    if df.empty:
        raise ValueError("Aucun événement à vectoriser dans events_clean.csv.")
    texts = build_event_texts(df)

    # Configurer le même modèle que dans le notebook d'embedding.
    load_dotenv(datas_folder.parent / ".env")
    api_key = os.getenv("MISTRAL_API") or os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("Ajouter MISTRAL_API dans le fichier .env du projet.")
    embedding_model = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)

    # Sauvegarder les vecteurs après chaque lot de 50 événements.
    embeddings = save_embeddings(
        texts, embedding_model, datas_folder / "events_embeddings.npy", batch_size=50,
    )
    print("Nombre d'événements :", len(df))
    print("Nombre d'embeddings :", len(embeddings))
    print("Dimensions :", embeddings.shape)
    return embeddings


if __name__ == "__main__":
    main()
