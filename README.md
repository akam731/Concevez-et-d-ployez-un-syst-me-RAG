#  Développez un assistant pour la recommandation d'évènements culturels


## Objectif

Créer un assistant capable de répondre aux questions sur des événements culturels récents à partir des données d'Open Agenda. Le système utilisera une recherche vectorielle avec Faiss, un RAG avec LangChain et un modèle Mistral. Une API REST permettra de poser des questions à l'assistant.

Source des données : [événements publics OpenAgenda](https://public.opendatasoft.com/explore/assets/evenements-publics-openagenda/view/).

## État du projet

Le système RAG, le chat Streamlit et l'API REST sont disponibles localement. La conteneurisation et le déploiement restent à réaliser.

## Étapes du projet

- [X] Étape 1 - Configurez l’environnement
- [X] Étape 2 - Effectuez le pré-processing des données Open Agenda
- [X] Étape 3 - Implémentez la base de données vectorielle avec Faiss
- [X] Étape 4 - Intégrez LangChain pour le système RAG
- [X] Étape 5 - Créez une API pour exposer le système RAG
- [ ] Étape 6 - Conteneurisez, déployez localement et préparez votre démonstration

## Arborescence

```text
.
├── .github/workflows/
│       └── tests.yml              # Tests automatiques GitHub Actions
├── 1-PreProcessing/
│   ├── 1-extract_events.ipynb     # Extraction et nettoyage des événements
│   └── 2-embedding.ipynb          # Génération des embeddings
├── 2-DataBase/
│   └── 3-create_database.ipynb    # Création de l'index FAISS
├── datas/
│   └── faiss_events/
│       ├── events.faiss            # Index vectoriel
│       ├── events.json             # Métadonnées des événements
│       └── manifest.json           # Configuration de l'index
├── docs/
│   ├── Fiche auto évaluation.pdf
│   ├── Rapport technique.docx      # Rapport technique
│   └── Shema UML.png               # Schema UML du projet
├── processing/                     # Scripts réutilisables de préparation des données
│   ├── create_database.py
│   ├── embedding.py
│   ├── extract_events.py
│   └── preprocessing.py            # Enchaînement des étapes pour /rebuild
├── tests/
│   ├── api.http                    # Exemples de requêtes à l'API
│   ├── functional/test_api.py      # Tests fonctionels des routes FastAPI
│   ├── rag/test_rag.py             # Évaluation Ragas des réponses
│   └── unit/
│       ├── test_database.py        # Tests unitaires sur la base de données
│       ├── test_preprocessing.py   # Tests unitaires sur le preprocessing
│       └── test_rag.py             # Tests unitaires sur le rag
├── utils/                          # Fonctions et paramètres partagés
│   ├── database.py
│   ├── preprocessing.py
│   ├── rag.py
│   ├── settings.py
│   └── system.md                  # Prompt system
├── .env.demo                      # Exemple de variables d'environnement
├── .gitignore                     # Fichiers exclus de Git
├── README.md                      # Documentation principale
├── app.py                         # Interface de chat Streamlit
├── main.py                        # API REST FastAPI
├── pyproject.toml                 # Configuration et dépendances Python
└── uv.lock                        # Versions verrouillées des dépendances
```

Les fichiers intermédiaires `events.csv`, `events_clean.csv` et `events_embeddings.npy` sont générés localement dans `datas/` et ne sont pas versionnés.

## Lancer le projet

### Installer les dépendances
Le projet nécessite Python 3.12 ou plus récent.

Avec `uv` installé, exécutez à la racine du projet :

```bash
uv sync
```


### Discuter avec le RAG

L'interface Streamlit permet de poser des questions au RAG et de consulter les événements utilisés pour répondre.

```bash
uv run streamlit run app.py
```

### Lancer l'API REST

Renseignez `MISTRAL_API` et `INTERNAL_API_KEY` dans le fichier `.env`, puis lancez l'API avec Uvicorn :

```bash
uv run uvicorn main:app --reload
```

L'API est disponible sur `http://127.0.0.1:8000` et sa documentation interactive sur `http://127.0.0.1:8000/docs`.

| Méthode | Route | Rôle | Clé interne requise |
| --- | --- | --- | --- |
| POST | `/ask` | Pose une question au RAG et retourne la réponse et les événements sources. | Oui |
| POST | `/rebuild` | Reconstruit la base vectorielle FAISS. | Oui |
| GET, POST | `/health` | Vérifie l'état de l'api. | Non |
| GET, POST | `/metadata` | Retourne les informations et les routes de l'API. | Non |

Pour `/ask`, envoyez un JSON contenant `question`. Pour les deux routes protégées, transmettez la clé interne dans l'en-tête `X-API-Key`.


## Tests et couverture

Lancer les tests et générer le rapport de couverture HTML :

```bash
uv run pytest -v
```

> Des tests unitaires et fonctionnels sont lancés pour tester le code. Et des tests RAGAS sont lancés pour tester le RAG

Ouvrir le rapport de couverture :

```bash
Start-Process .\tests\htmlcov\index.html
```

Les tests sont répartis dans `tests/unit/`, `tests/functional/` et `tests/rag/`.
Le rapport `tests/htmlcov/index.html` mesure la couverture du code Python dans `utils/`. Les tests échouent si elle est inférieure à 90 %.

## Documentation

| Document | Contenu |
| --- | --- |
| [Rapport technique](docs/Rapport%20technique.docx) | Présentation de la conception, de l'implémentation et de l'évaluation du projet. |
| [Schéma d'architecture](docs/Shema%20UML.png) | Vue d'ensemble des composants et des échanges du système RAG. |
| [Fiche d'autoévaluation](docs/Fiche%20auto%20%C3%A9valuation.pdf) | Critères d'évaluation du projet. |
