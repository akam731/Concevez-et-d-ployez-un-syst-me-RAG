# Paramètres du projet
from pathlib import Path
import pandas as pd

# Dossier de datas
datas_folder =  Path("../datas").resolve()

# Ville choisie
city = "Paris"

# Date minimale
min_date = pd.Timestamp("2025-10-10", tz="UTC")