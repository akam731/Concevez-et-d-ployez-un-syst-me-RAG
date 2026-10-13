#  Développez un assistant pour la recommandation d'évènements culturels


## Objectif

Créer un assistant capable de répondre aux questions sur des événements culturels récents à partir des données d'Open Agenda. Le système utilisera une recherche vectorielle avec Faiss, un RAG avec LangChain et un modèle Mistral. Une API REST permettra de poser des questions à l'assistant.

Source des données : [événements publics OpenAgenda](https://public.opendatasoft.com/explore/assets/evenements-publics-openagenda/view/).

## État du projet

Le projet est en cours de démarrage. Le système RAG et l'API ne sont pas encore disponibles.

## Étapes du projet

- [X] Étape 1 - Configurez l’environnement
- [X] Étape 2 - Effectuez le pré-processing des données Open Agenda
- [ ] Étape 3 - Implémentez la base de données vectorielle avec Faiss
- [ ] Étape 4 - Intégrez LanChain pour le système RAG 
- [ ] Étape 5 - Créez une API pour exposer le système RAG
- [ ] Étape 6 - Conteneurisez, déployez localement et préparez votre démonstration

## Arborescence

```text
.
├── .venv/                         # Environnement virtuel Python
├── .env                           # Variables d'environnement locales
├── .env.demo                      # Exemple de variables d'environnement
├── .gitignore                     # Fichiers exclus de Git
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

Le fichier `pyproject.toml` définit actuellement un environnement Python 3.14 ou plus récent. Les dépendances et les commandes de lancement seront ajoutées au fur et à mesure du développement.

Avec `uv` installé, exécutez à la racine du projet :

```bash
uv sync
```

Cette commande crée l'environnement virtuel et synchronise les dépendances du projet.
