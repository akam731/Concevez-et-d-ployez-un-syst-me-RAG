"""Téléchargement et filtrage des événements OpenAgenda, sans graphiques."""

import pandas as pd

from utils.preprocessing import download_events, filter_events
from utils.settings import city, datas_folder, min_date


def main():
    """Exécute l'extraction et le nettoyage avec les paramètres du projet."""
    # Télécharger les événements de la ville choisie.
    download_events(city, datas_folder / "events.csv")

    # Charger le CSV en conservant les zéros initiaux des codes postaux.
    df = pd.read_csv(
        datas_folder / "events.csv", sep=";",
        dtype={"location_postalcode": "string"},
    )

    # Conserver les événements à partir de la date minimale.
    events = filter_events(df, min_date)
    print("Avant :", df.shape)
    print("Après :", events.shape)

    # Enregistrer les événements filtrés pour le notebook d'embedding.
    events.to_csv(
        datas_folder / "events_clean.csv",
        sep=";", index=False, encoding="utf-8-sig",
    )

    print("Tous les UUID sont uniques :", df["uid"].is_unique)
    print(df.columns.tolist())
    return events


if __name__ == "__main__":
    main()
