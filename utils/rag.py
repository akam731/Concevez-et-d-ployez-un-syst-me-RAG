import json
import os
from functools import lru_cache
import numpy as np
import faiss
from dotenv import load_dotenv
from httpx import HTTPStatusError
import streamlit as st
from mistralai.client import Mistral
from langchain_mistralai import ChatMistralAI
from utils.settings import (
    datas_folder, llm_model, llm_temperature, llm_top_p, llm_max_tokens,
    system_prompt_path,
)


def embed(text: str):
    """ Génère l'embedding d'un text
    :param text : texte à envoyer à Mistral pour obtenir son vecteur.
    """
    try:
        # Chargement des variables d'environment
        load_dotenv(datas_folder.parent / ".env")
        api_key = os.getenv("MISTRAL_API") or os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Ajouter MISTRAL_API dans le fichier .env du projet.")
        # Création de l'embedding grâce à l'api Mistral
        with Mistral(api_key=api_key) as client:
            response = client.embeddings.create(model="mistral-embed", inputs=text)
        # Retourne l'embedding généré
        return np.array(response.data[0].embedding)
    except Exception as e:
        st.error(f"Erreur lors de la génération de l'embedding: {e}")


# Conserve les premiers résultats en cache pour ne pas recharger la base de données à chaque appel
@lru_cache(maxsize=1)
def load_resources():
    """Charge et vérifie la base de données FAISS """

    # Récupération des variables d'environnement
    load_dotenv(datas_folder.parent / ".env")
    api_key = os.getenv("MISTRAL_API") or os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("Ajouter MISTRAL_API dans le fichier .env du projet.")

    # Récupération des fichiers de la bdd FAISS
    folder = datas_folder / "faiss_events"
    index = faiss.read_index(str(folder / "events.faiss"))
    events = json.loads((folder / "events.json").read_text(encoding="utf-8"))
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if index.ntotal != len(events) or index.d != manifest["dimension"]:
        raise ValueError("L'index et les métadonnées de la base ne correspondent pas.")
    if any(event["position"] != position for position, event in enumerate(events)):
        raise ValueError("Les métadonnées ne sont pas dans l'ordre des vecteurs.")

    # Vérification du model d'embedding pour garantir la compatibilité
    if manifest["model"] != "mistral-embed":
        raise ValueError("La base doit utiliser le modèle mistral-embed.")
    return index, events


def search_events(question, k=5):
    """ Retrouve les événements les plus proches de la question.

    :param question: question dont on recherche les événements les plus proches.
    :param k: nombre maximum d'événements à récupérer, 5 par défaut.
    :return: tableau contenant les events
    """

    # Vérifier que la question contient du texte et que k est un nombre d'événements positif.
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Veuillez poser une question.")
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k doit être un entier strictement positif.")

    # Charger l'index FAISS et les métadonnées associées à chaque vecteur.
    index, events = load_resources()
    # Ne rien rechercher si l'index ne contient aucun vecteur.
    if index.ntotal == 0:
        return []
    # Générer l'embedding de la question
    question_embedding = embed(question.strip())
    if question_embedding is None:
        raise ValueError("Impossible de calculer l'embedding de la question.")

    # Création du vecteur
    vector = np.asarray([question_embedding], dtype=np.float32)
    # Vérifier la dimension du vecteur et refuser les valeurs invalides ou un vecteur nul.
    if vector.shape != (1, index.d) or not np.isfinite(vector).all():
        raise ValueError("L'embedding de la question est incompatible avec l'index.")
    if np.linalg.norm(vector) == 0:
        raise ValueError("L'embedding de la question est nul.")

    # Normalisation du vecteur de la question pour le comparer par la suite
    faiss.normalize_L2(vector)
    # Récupération des événements concernés par la question
    scores, positions = index.search(vector, min(k, index.ntotal))
    # Retourne la liste des événements
    return [
        {"position": int(position), "score": float(score),
         "metadata": events[int(position)]["metadata"]}
        for position, score in zip(positions[0], scores[0])
    ]


def built_prompt(messages, question=None, max_messages=5, context_segments=None):
    """Prépare les instructions, les événements retrouvés et les messages récents.

    :param messages : historique de discussion
    :param question : question à ajouter
    :param max_messages : nombre maximum de messages récents à conserver
    :param context_segments : événements déjà retrouvés
    """
    # Contrôles
    if max_messages <= 0:
        raise ValueError("max_messages doit être strictement positif.")
    if context_segments is None:
        context_segments = search_events(question) if question else []
        
    # Récupérer le prompt system
    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
    # Ajout des événements retrouvés
    system_prompt += "# Événements retrouvés\n\n" + json.dumps(context_segments, ensure_ascii=False)
    formatted_messages = [{"role": "system", "content": system_prompt}]
    formatted_messages.extend(
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages[-max_messages:]
        if msg["role"] in ("user", "assistant")
    )
    if question and (not formatted_messages or formatted_messages[-1] != {"role": "user", "content": question}):
        formatted_messages.append({"role": "user", "content": question})
    # Retour du prompt final
    return formatted_messages


def ask(question, k=5, messages=None):
    """Pose une question au LLM avec recherche des événements concernés.

    :param question : question à laquelle le modèle doit répondre.
    :param k : nombre maximum d'événements utilisés comme contexte, 5 par défaut.
    :param messages : historique des messages
    """
    try:
        # Récupération des événements pertinents
        results = search_events(question, k)
        if not results:
            return {"answer": "La base ne contient aucun événement.", "events": []}

        # Génère le prompt
        prompt = built_prompt(
            messages or [], question=question.strip(), context_segments=results,
        )
        # Récupère le modèle
        api_key = os.getenv("MISTRAL_API") or os.getenv("MISTRAL_API_KEY")
        model = ChatMistralAI(
            model=llm_model,
            api_key=api_key,
            temperature=llm_temperature,
            top_p=llm_top_p,
            max_tokens=llm_max_tokens,
        )
        # Pose la question au model
        response = model.invoke(prompt)
        # Retourne la réponse
        return {"answer": response.content, "events": results}

    # Erreur API
    except HTTPStatusError as error:
        if error.response.status_code == 429:
            if error.response.headers.get("x-ratelimit-limit-req-minute") == "0":
                return {
                    "answer": "Mistral n'autorise aucune requête pour ce modèle. Il faut modifier le llm_model dans utils/settings.py.",
                    "events": [],
                }
            return {"answer": "Limite d'utilisation API atteinte.", "events": []}
        return {"answer": f"L'API Mistral a refusé la demande (HTTP {error.response.status_code}).", "events": []}

    # Erreur system
    except (ValueError, FileNotFoundError, RuntimeError):
        return {"answer": "Impossible de répondre : vérifie la base FAISS, la question et l'accès à Mistral.", "events": []}
