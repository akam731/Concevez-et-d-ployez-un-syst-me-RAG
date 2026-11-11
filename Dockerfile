# Part d'une image avec python 3.12
FROM python:3.12-slim-trixie

# Ajoute uv
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

# FAISS utilise OpenMP pour les calculs vectoriels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Définit /app comme dossier de travail dans le conteneur.
WORKDIR /app

# Configure uv pour installer les dépendances dans cet environnement.
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Les dépendances sont installées avant de copier le code pour conserver le cache Docker.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copie le code nécessaire à l’API et au RAG.
COPY main.py ./
COPY processing ./processing
COPY utils ./utils
COPY datas/faiss_events ./datas/faiss_events

# Indique que l’API écoute sur le port 8000.
EXPOSE 8000

# Éxécute la commande qui démarre FastAPI avec Uvicorn
CMD ["/app/.venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
