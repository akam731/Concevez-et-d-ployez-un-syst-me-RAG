"""Création de la base FAISS avec les embeddings existants."""

from utils.database import load_data, build_events, create_index, save_database
from utils.settings import datas_folder


def main():
    """Charge les vecteurs existants et sauvegarde l'index avec les métadonnées."""
    # Charger les fichiers produits par l'extraction et l'embedding.
    df, embeddings = load_data(datas_folder)
    events = build_events(df)

    # Créer l'index sans recalculer les embeddings.
    index = create_index(embeddings)
    output_dir = datas_folder / "faiss_events"
    save_database(index, events, output_dir)

    print(f"Vecteurs indexés : {index.ntotal:,}")
    print(f"Dimension : {index.d}")
    print("Base sauvegardée dans :", output_dir)
    return index


if __name__ == "__main__":
    main()
