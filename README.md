#  Développez un assistant pour la recommandation d'évènements culturels


## Objectif

Créer un assistant capable de répondre aux questions sur des événements culturels récents à partir des données d'Open Agenda. Le système utilisera une recherche vectorielle avec Faiss, un RAG avec LangChain et un modèle Mistral. Une API REST permettra de poser des questions à l'assistant.

Source des données : [événements publics OpenAgenda](https://public.opendatasoft.com/explore/assets/evenements-publics-openagenda/view/).

## État du projet

Le projet est en cours de démarrage. Le système RAG et l'API ne sont pas encore disponibles.

## Étapes du projet

- [X] Étape 1 - Configurez l’environnement
- [ ] Étape 2 - Effectuez le pré-processing des données Open Agenda
- [ ] Étape 3 - Implémentez la base de données vectorielle avec Faiss
- [ ] Étape 4 - Intégrez LanChain pour le système RAG 
- [ ] Étape 5 - Créez une API pour exposer le système RAG
- [ ] Étape 6 - Conteneurisez, déployez localement et préparez votre démonstration

## Arborescence

```text
.
├── .venv/            # environnement virtuel
├── .env              # variables d'environnement locales
├── .env.demo         # exemple de variables d'environnement
├── .gitignore        # fichiers exclus de Git
├── docs              # dossier contenant les documentations du projet
├── pyproject.toml    # configuration et dépendances Python
├── README.md         # documentation du projet
└── uv.lock           # versions verrouillées des dépendances
```

## Lancer le projet

Le fichier `pyproject.toml` définit actuellement un environnement Python 3.14 ou plus récent. Les dépendances et les commandes de lancement seront ajoutées au fur et à mesure du développement.

Avec `uv` installé, exécutez à la racine du projet :

```bash
uv sync
```

Cette commande crée l'environnement virtuel et synchronise les dépendances du projet.
