# Routes de l'API
import hmac
import logging
import os
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from processing.preprocessing import PreProcessing
from utils.rag import ask as rag_ask, load_resources

logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_internal_api_key(api_key: str | None = Security(api_key_header)) -> None:
    """Vérifie la clé interne avant d'autoriser les routes protégées."""
    expected_key = os.getenv("INTERNAL_API_KEY")
    if not expected_key:
        logger.error("La variable INTERNAL_API_KEY n'est pas configurée.")
        raise HTTPException(
            status_code=500,
            detail="La clé interne de l'API n'est pas configurée.",
        )

    if api_key is None or not hmac.compare_digest(api_key, expected_key):
        raise HTTPException(
            status_code=401,
            detail="Clé API manquante ou invalide.",
        )


class AskRequest(BaseModel):
    """Class contenant l'attribut attendu pour pouvoir poser une question au modèle"""
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    """Class contenant les attributs attendus pour afficher la réponse du modèle"""
    answer: str
    events: list[dict[str, Any]]


# POST : Pose la question au système RAG
@app.post("/ask", response_model=AskResponse,
          name="Pose une question au modèle",
          description="Récupère la question, et demande une réponse au modèle. Fournis la réponses et les sources.",
          dependencies=[Depends(require_internal_api_key)])
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="La question ne peut pas être vide.")
    try:
        return rag_ask(request.question)
    except Exception as error:
        logger.exception("Échec de communication avec le modèle.")
        raise HTTPException(status_code=500, detail="Échec de communication avec le modèle.") from error


# POST : Régénère la base vectorielle
@app.post("/rebuild",
          name="Reconstruit la base de données vectorielle",
          description="Reconstruit la base de données vectorielle et vide le cache du rag. Retourne le nombre d'événements que contient la nouvelle base de données.",
          dependencies=[Depends(require_internal_api_key)])
def rebuild():
    try:
        events = PreProcessing().pre_processing()  # Construction de la base
        load_resources.cache_clear()  # Nettoyage du cache du rag
    except Exception as error:
        logger.exception("Échec de la reconstruction de la base vectorielle")
        raise HTTPException(status_code=500, detail="La reconstruction de la base a échoué.") from error
    return {
        "status_code": 200,
        "message": "Reconstruction de la base vectorielle avec succès",
        "n_events": events.ntotal}


# GET / POST : Permet d'avoir l'état de santé de l'api
@app.get("/health",
          name="État de santé de l'api",
          description="Retourne le statut et l'état de santé de l'api")
@app.post("/health",
          name="État de santé de l'api",
          description="Retourne le statut et l'état de santé de l'api")
def health():
    return {"status_code": 200, "message": "OK"}


# GET / POST : Permet d'avoir les metadatas de l'api
@app.get("/metadata",
         name="Metadatas de l'api",
         description="Détail de l'api, nom, version, environement, endpoints...")
@app.post("/metadata",
         name="Metadatas de l'api",
         description="Détail de l'api, nom, version, environement, endpoints...")
def metadata():
    return {
        "name": "Développez un assistant pour la recommandation d'évènements culturels",
        "version": "1.0.0",
        "environment": "development",
        "supportedLanguages": ["fr"],
        "endpoints": [
            {
                "path": "/ask",
                "method": "POST",
                "description": "Pose une question au système RAG."
            },
            {
                "path": "/rebuild",
                "method": "POST",
                "description": "Reconstruit la base vectorielle."
            },
            {
                "path": "/health",
                "method": ["POST","GET"],
                "description": "Retourne les métadonnées de l'API."
            },
            {
                "path": "/metadata",
                "method": ["POST","GET"],
                "description": "Retourne les métadonnées de l'API."
            }
        ]
    }
