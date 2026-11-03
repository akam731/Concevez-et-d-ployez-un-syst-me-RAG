#  Développez un assistant pour la recommandation d'évènements culturels


## Objectif

Créer un assistant capable de répondre aux questions sur des événements culturels récents à partir des données d'Open Agenda. Le système utilise une recherche vectorielle avec Faiss, un RAG avec LangChain et un modèle Mistral. Une API REST permet de poser des questions à l'assistant.

Source des données : [événements publics OpenAgenda](https://public.opendatasoft.com/explore/assets/evenements-publics-openagenda/view/).

## État du projet

Le système RAG, la base FAISS et le chat Streamlit sont disponibles. L'application FastAPI se trouve dans `main.py`.

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
├── .venv/                         # Environnement virtuel Python
├── .env                           # Variables d'environnement locales
├── .env.demo                      # Exemple de variables d'environnement
├── .gitignore                     # Fichiers exclus de Git
│
├── app.py                         # Interface de discussion Streamlit
├── main.py                        # Application API FastAPI
│
├── 1-PreProcessing/               # Extraction, nettoyage et vectorisation
│   ├── 1-extract_events.ipynb     # Récupération et nettoyage des événements
│   └── 2-embedding.ipynb          # Génération des embeddings
│
├── 2-DataBase/                    # Création de la base vectorielle
│   └── 3-create_database.ipynb    # Création et sauvegarde de l'index FAISS
│
├── datas/                         # Données du projet
│   ├── events.csv                 # Événements bruts récupérés depuis OpenAgenda
│   ├── events_clean.csv           # Événements filtrés et nettoyés
│   └── events_embeddings.npy      # Vecteurs générés
│
├── docs/                          # Documentation du projet
├── pyproject.toml                 # Configuration et dépendances Python
├── README.md                      # Documentation principale du projet
└── uv.lock                        # Versions verrouillées des dépendances
```

## Lancer le projet

### Installer les dépendances

Le projet requiert Python 3.14 ou plus récent et `uv`. Depuis la racine du projet :

```bash
uv sync
cp .env.demo .env
```

Remplacez `[API_KEY]` dans `MISTRAL_API` par votre clé API Mistral et `[INTERNAL_API_KEY]` par une clé interne secrète. Cette dernière protège les routes `POST /ask` et `POST /rebuild`. L'index FAISS utilisé par le chat se trouve dans `datas/faiss_events/`.

### Lancer le chat

```bash
uv run streamlit run app.py
```

Ouvrez ensuite `http://localhost:8501` dans votre navigateur.

### Lancer l'API REST

Dans un autre terminal, depuis la racine du projet :

```bash
uv run uvicorn main:app --reload
```

L'API est accessible sur `http://127.0.0.1:8000`. Ses routes sont :

| Méthode | Route | Description |
| --- | --- | --- |
| `POST` | `/ask` | Reçoit `{"question": "..."}` et renvoie la réponse du RAG et les événements retrouvés. Requiert l'en-tête `X-API-Key`. |
| `POST` | `/rebuild` | Actualise les événements, régénère les embeddings et reconstruit l'index FAISS. Requiert l'en-tête `X-API-Key`. |
| `GET` | `/docs` | Documentation interactive Swagger. |
| `GET` | `/redoc` | Documentation interactive alternative. |
| `GET, POST` | `/health` | État de santé de l'API. |
| `GET, POST` | `/metadata` | Metadatas de l'API. |

La documentation interactive de l'API est disponible sur `http://127.0.0.1:8000/docs`.
Pour appeler `/ask` ou `/rebuild`, envoyez la valeur de `INTERNAL_API_KEY` dans l'en-tête `X-API-Key`. Dans Swagger, cliquez sur **Authorize** et renseignez cette clé.

## Tests et couverture

Depuis la racine du projet, lancer les tests et générer le rapport de couverture HTML :

```bash
uv run pytest -v
```

Ouvrir le rapport de couverture sous Windows :

```bash
Start-Process .\tests\htmlcov\index.html
```

Les tests unitaires sont dans `tests/unit/` et les tests HTTP de l'API dans `tests/functional/`. Les appels à OpenAgenda et Mistral sont simulés.
Le rapport `tests/htmlcov/index.html` mesure la couverture du code Python dans `utils/`.
