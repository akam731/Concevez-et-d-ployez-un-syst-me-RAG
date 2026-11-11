# Paramètres du projet
from pathlib import Path
import pandas as pd

# Dossier de datas
datas_folder = Path(__file__).resolve().parents[1] / "datas"

# Ville choisie
city = "Paris"

# Date minimale
min_date = pd.Timestamp("2025-10-10", tz="UTC")

# Modèle Mistral utilisé pour générer les réponses
llm_model = "mistral-large-4"

# Interval entre chaque requetes
mistral_min_interval_seconds = 1

# Température du modèle
llm_temperature = 0.2

# Top-p du modèle
llm_top_p = 0.9

# Nombre maximum de tokens générés dans la réponse
llm_max_tokens = 800

# Fichier contenant les consignes système du chatbot
system_prompt_path = Path(__file__).resolve().parent / "system.md"
