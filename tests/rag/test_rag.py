# Évaluation du RAG avec Ragas sur une liste de questions.

import asyncio
import math
import os
import sys
from pathlib import Path
from statistics import mean
from time import sleep
from types import ModuleType

import pytest
from dotenv import load_dotenv
from utils import settings
from utils.rag import ask as rag_ask

# Liste des questions
CASES = [
    {
        "question": "Combien coûte l'entrée au ciné-concert  Vive Kaprálová ?",
        "reference": "L'entrée coûte 10 € et se réserve sur Eventbrite.",
    },
    {
        "question": "Où se tient l'exposition Amazônia. Créations et futurs autochtones ?",
        "reference": "L'exposition se tient au Musée du Quai Branly - Jacques Chirac, 37 quai Jacques Chirac à Paris.",
    },
    {
        "question": "Quand a lieu le concert Coeur Canari de Jann Halexander ?",
        "reference": "Le concert a lieu le vendredi 7 août à 19h au Théâtre de l'Île Saint-Louis.",
    },
    {
        "question": "Quels sont les tarifs du récital Hommage à Guru M. K. Saroja ?",
        "reference": "Le plein tarif est de 20 €, le tarif réduit de 15 € et le tarif enfant de 10 €.",
    },
    {
        "question": "L'atelier scolaire La fortification bastionnée est-il gratuit et faut-il s'inscrire ?",
        "reference": "L'atelier est gratuit et l'inscription est obligatoire.",
    },
    {
        "question": "À quelle heure commence la conférence Bien démarrer un projet d'écriture avec Antoine Cupial ?",
        "reference": "La conférence commence à 13h le mercredi 26 novembre.",
    },
    {
        "question": "La visite guidée architecturale du Musée de La Poste est-elle incluse dans le billet ?",
        "reference": "Oui, la visite architecturale est incluse dans le billet d'entrée au musée.",
    },
    {
        "question": "Où a lieu L'Italie à travers son cinéma ?",
        "reference": "L'événement a lieu au Cinéma du Panthéon, 13 rue Victor-Cousin à Paris.",
    },
    {
        "question": "Où se déroule le Festival Journée européenne des langues ?",
        "reference": "Le festival se déroule au Goethe-Institut, 17 avenue d'Iéna à Paris.",
    },
    {
        "question": "À qui s'adresse l'événement Intermittents du spectacle : 20 minutes chrono pour préparer votre fin de droit ?",
        "reference": "Il s'adresse aux intermittents du spectacle dont les droits arrivent à terme et vont être réexaminés.",
    },
]

# Seuils moyens pour la validation des métrique
# Fidélité : la réponse est-elle soutenue par les événements ?
MIN_MEAN_FAITHFULNESS = 0.70
# Précision : les événements utiles sont-ils en tête ?
MIN_MEAN_CONTEXT_PRECISION = 0.60


def _context_text(event):
    """Transforme les métadonnées d'un événement en texte lisible par Ragas."""
    return "\n".join(
        f"{field}: {value}"
        for field, value in event["metadata"].items()
        if value
    )


def test_rag_answers_with_ragas(monkeypatch):
    """Évalue les dix réponses et la pertinence des événements retrouvés."""

    # Charger la même clé que le RAG, sans l'inscrire dans ce fichier.
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    api_key = os.getenv("MISTRAL_API") or os.getenv("MISTRAL_API_KEY")
    if not api_key:
        pytest.fail("Clé API Mistral absente.")

    # Ragas 0.4.3 cherche ChatVertexAI à son ancienne adresse.
    # On lui fourni donc manuelement
    from langchain_google_vertexai import ChatVertexAI
    old_vertexai_path = "langchain_community.chat_models.vertexai"
    vertexai_module = ModuleType(old_vertexai_path)
    vertexai_module.ChatVertexAI = ChatVertexAI
    monkeypatch.setitem(sys.modules, old_vertexai_path, vertexai_module)
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import ContextPrecision, Faithfulness

    samples = []

    # Parcours de la liste des questions
    for number, case in enumerate(CASES, start=1):
        # Récupération de la question et de la réponse de référence
        question = case["question"]
        reference = case["reference"]
        assert question.strip() and reference.strip(), f"Cas {number} incomplet."

        # Un appel à Mistral peut échouer momentanément pendant cette série de
        # questions. Réessayer seulement les erreurs temporaires ; si elles
        # persistent, le test échoue et affiche la réponse d'erreur.
        for attempt, delay in enumerate((0, 15, 45), start=1):
            if delay:
                sleep(delay)
            result = rag_ask(question)
            if result["events"] or not result["answer"].startswith(
                ("Impossible de répondre", "Limite d'utilisation API atteinte")
            ):
                break
        answer = result["answer"] # Réponse du rag
        contexts = [_context_text(event) for event in result["events"]]

        # Échoue si aucun événement retrouvé
        assert contexts, f"Aucun événement retrouvé pour le cas {number} après {attempt} tentative(s) : {answer}"
        # Vérifie que la réponse est valide
        assert isinstance(answer, str) and answer.strip()
        # Met de côté les résultats pour la vérification à suivre
        samples.append((number, question, reference, answer, contexts))

    # Fonction d'évaluation des samples en asynchrone
    async def score_samples():
        # Tableau des scores
        faithfulness_scores = []
        precision_scores = []

        # Envoi les requêtes d'évaluation à mistral
        async with AsyncOpenAI(api_key=api_key, base_url="https://api.mistral.ai/v1") as client:
            # Paramétrage du modèle évaluateur
            evaluator = llm_factory(settings.llm_model, provider="openai", client=client)
            faithfulness = Faithfulness(llm=evaluator)
            context_precision = ContextPrecision(llm=evaluator)

            # Parcours des diférents tests
            for number, question, reference, answer, contexts in samples:
                # Calcul des scores
                faithfulness_score = float((await faithfulness.ascore(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=contexts,
                )).value)
                precision_score = float((await context_precision.ascore(
                    user_input=question,
                    reference=reference,
                    retrieved_contexts=contexts,
                )).value)

                # Vérification que les scores sont bien compris entre 0 et 1
                assert math.isfinite(faithfulness_score) and 0 <= faithfulness_score <= 1
                assert math.isfinite(precision_score) and 0 <= precision_score <= 1

                # Sauvegarde des scores
                faithfulness_scores.append(faithfulness_score)
                precision_scores.append(precision_score)
        return faithfulness_scores, precision_scores

    # Éxécution de la fonction définie juste avant
    faithfulness_scores, precision_scores = asyncio.run(score_samples())

    # Le test échoue si la qualité moyenne passe sous les seuils choisis.
    assert mean(faithfulness_scores) >= MIN_MEAN_FAITHFULNESS
    assert mean(precision_scores) >= MIN_MEAN_CONTEXT_PRECISION
