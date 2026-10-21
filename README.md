#  Développez un assistant pour la recommandation d'évènements culturels


## Objectif

Créer un assistant capable de répondre aux questions sur des événements culturels récents à partir des données d'Open Agenda. Le système utilisera une recherche vectorielle avec Faiss, un RAG avec LangChain et un modèle Mistral. Une API REST permettra de poser des questions à l'assistant.

Source des données : [événements publics OpenAgenda](https://public.opendatasoft.com/explore/assets/evenements-publics-openagenda/view/).

## État du projet

Le projet est en cours de démarrage. Le système RAG et l'API ne sont pas encore disponibles.

## Étapes du projet

- [X] Étape 1 - Configurez l’environnement
- [X] Étape 2 - Effectuez le pré-processing des données Open Agenda
- [X] Étape 3 - Implémentez la base de données vectorielle avec Faiss
- [X] Étape 4 - Intégrez LanChain pour le système RAG 
- [ ] Étape 5 - Créez une API pour exposer le système RAG
- [ ] Étape 6 - Conteneurisez, déployez localement et préparez votre démonstration

## Arborescence

```text
.
├── .env                           # Clé API et variables locales
├── .env.demo                      # Exemple de variables d'environnement
├── .gitignore                     # Fichiers exclus de Git
├── .venv/                         # Environnement virtuel UV
├── 1-PreProcessing/               
│   ├── 1-extract_events.ipynb     # Extraction et nettoyage des événements
│   └── 2-embedding.ipynb          # Génération des embeddings
├── 2-DataBase/                    # Notebook de création de la base FAISS
│   └── 3-create_database.ipynb
├── app.py                         # Interface Streamlit du RAG
├── datas/                         # Données générées localement
│   ├── events.csv                 # Événements bruts OpenAgenda
│   ├── events_clean.csv           # Événements filtrés et nettoyés
│   └── events_embeddings.npy      # Vecteurs des événements
├── docs/                          # Documentations du projet
│   └── Fiche auto évaluation.pdf  
├── main.py                        # Point d'entrée de l'API FastAPI
├── processing/                    
│   ├── create_database.py         # Création de la base de données FAISS
│   ├── embedding.py               # Génération des embeddings
│   ├── extract_events.py          # Récupération des événements
│   └── preprocessing.py           # Class regroupant tous les processus
├── tests/
│   └── unit/                      # Tests unitaires
│       ├── test_database.py
│       └── test_preprocessing.py
├── utils/                         # Fonctions partagés
│   ├── database.py
│   ├── preprocessing.py
│   ├── rag.py                    
│   ├── settings.py                # Paramètres du projet
│   └── system.md                  # Prompt System du chat
├── pyproject.toml                 # Configuration et dépendances Python
├── README.md                      # Documentation principale du projet
└── uv.lock                        # Versions verrouillées des dépendances
```

## Lancer le projet

Le fichier `pyproject.toml` définit actuellement un environnement Python 3.14 ou plus récent. Les dépendances et les commandes de lancement seront ajoutées au fur et à mesure du développement.

Avec `uv` installé, exécutez à la racine du projet :

```bash
uv sync
```
```bash
uv run streamlit run app.py
```

La première commande crée l'environnement virtuel et synchronise les dépendances du projet.

## Tests et couverture

Depuis la racine du projet, lancer les tests et générer le rapport de couverture HTML :

```powershell
uv run pytest
```

Ouvrir le rapport de couverture sous Windows :

```powershell
Start-Process .\htmlcov\index.html
```

Les tests sont dans `tests/unit/`. Les appels à OpenAgenda et Mistral sont simulés.
Le rapport `htmlcov/index.html` mesure la couverture du code Python dans `utils/`.
