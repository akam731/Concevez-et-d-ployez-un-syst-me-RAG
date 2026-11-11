# Évaluation du RAG avec Ragas sur une liste de questions.

import asyncio
import math
import os
import sys
from pathlib import Path
from statistics import mean
from time import monotonic, sleep
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
    from openai import APIConnectionError, AsyncOpenAI, InternalServerError, RateLimitError
    from ragas.llms import llm_factory
    from ragas.metrics.collections import ContextPrecision, Faithfulness

    interval = settings.mistral_min_interval_seconds
    assert interval > 0
    rag_interval = interval
    # Trois événements par réponse réduisent le nombre d'appels de précision.
    retrieval_k = int(os.getenv("RAGAS_TOP_K", "3"))
    assert retrieval_k > 0
    # Récupères toutes les questions par défaut, utile pour le débug
    cases = CASES[:int(os.getenv("RAGAS_CASE_LIMIT", str(len(CASES))))]
    assert cases, "RAGAS_CASE_LIMIT doit laisser au moins une question."
    samples = []
    last_answer_at = None

    # Parcours de la liste des questions
    for number, case in enumerate(cases, start=1):
        # Récupération de la question et de la réponse de référence
        question = case["question"]
        reference = case["reference"]
        assert question.strip() and reference.strip(), f"Cas {number} incomplet."

        # Test fois au cas ou le rag échouerai
        for attempt in range(1, 4):
            if last_answer_at is not None:
                sleep(max(0, rag_interval - (monotonic() - last_answer_at)))
            print(f"Question {number}/{len(cases)}, tentative {attempt}", flush=True)
            result = rag_ask(question, k=retrieval_k)
            last_answer_at = monotonic()
            if result["events"] or not result["answer"].startswith(
                ("Impossible de répondre", "Limite d'utilisation API atteinte")
            ):
                break
            rag_interval = min(rag_interval * 2, 40)

        answer = result["answer"] # Réponse du rag
        contexts = [_context_text(event) for event in result["events"]]

        # Échoue si aucun événement retrouvé
        assert contexts, f"Aucun événement retrouvé pour le cas {number} après {attempt} tentative(s) : {answer}"
        # Vérifie que la réponse est valide
        assert isinstance(answer, str) and answer.strip()
        # Met de côté les résultats pour la vérification à suivre
        samples.append((number, question, reference, answer, contexts))

    # Chaque métrique Ragas fait plusieurs appels au modèle. On espace donc chaque requête HTTP.
    class PacedAsyncOpenAI(AsyncOpenAI):
        def __init__(self):
            super().__init__(
                api_key=api_key,
                base_url="https://api.mistral.ai/v1",
                max_retries=0,
                timeout=120,
            )
            self._interval = rag_interval
            self._next_request_at = last_answer_at + self._interval
            self._pace_lock = asyncio.Lock()

        async def request(self, *args, **kwargs):
            transient_attempts = 0
            while True:
                async with self._pace_lock:
                    await asyncio.sleep(max(0, self._next_request_at - monotonic()))
                    self._next_request_at = monotonic() + self._interval
                try:
                    completion = await super().request(*args, **kwargs)
                    # Si le modèle a épuisé max_tokens avant de finir
                    if any(
                        choice.finish_reason == "length"
                        for choice in getattr(completion, "choices", [])
                    ):
                        options = args[1] if len(args) > 1 else kwargs.get("options")
                        body = getattr(options, "json_data", None)
                        token_field = (
                            "max_tokens" if isinstance(body, dict) and "max_tokens" in body
                            else "max_completion_tokens"
                        )
                        current_limit = body.get(token_field) if isinstance(body, dict) else None
                        if isinstance(current_limit, int) and current_limit < 12288:
                            new_limit = min(current_limit * 2, 12288)
                            body[token_field] = new_limit
                            print(
                                f"Réponse Ragas tronquée : nouvel essai avec {new_limit} tokens.",
                                flush=True,
                            )
                            continue
                        raise AssertionError(
                            "Réponse Ragas encore tronquée à 12288 tokens ; "
                            "réduire la taille des réponses ou des contextes."
                        )
                    # On récupère uniquement le bloc text
                    for choice in getattr(completion, "choices", []):
                        content = choice.message.content
                        if isinstance(content, list):
                            choice.message.content = "".join(
                                block.get("text", "") for block in content
                                if isinstance(block, dict) and block.get("type") == "text"
                            )
                    return completion
                except (APIConnectionError, InternalServerError, RateLimitError) as error:
                    transient_attempts += 1
                    if transient_attempts == 5:
                        raise
                    if isinstance(error, RateLimitError):
                        self._interval = max(self._interval, 1.25)
                    # Respecter Retry-After si Mistral le fournit.
                    response = getattr(error, "response", None)
                    retry_after = response.headers.get("retry-after") if response else None
                    try:
                        retry_after_seconds = float(retry_after) if retry_after else 0
                    except ValueError:
                        retry_after_seconds = 0
                    async with self._pace_lock:
                        self._next_request_at = max(
                            self._next_request_at,
                            monotonic() + self._interval,
                            monotonic() + retry_after_seconds,
                            monotonic() + (30 if isinstance(error, RateLimitError) else 0),
                        )
                    print(
                        f"Appel Mistral interrompu ({type(error).__name__}) ; "
                        f"nouvelle tentative {transient_attempts + 1}/5.",
                        flush=True,
                    )

    # Fonction d'évaluation en asynchrone
    async def score_samples():
        # Envoi les requêtes d'évaluation à mistral
        async with PacedAsyncOpenAI() as client:
            # Paramétrage du modèle évaluateur
            faithfulness_evaluator = llm_factory(
                settings.llm_model,
                provider="openai",
                client=client,
                max_tokens=4096,
                reasoning_effort="low",
            )
            precision_evaluator = llm_factory(
                settings.llm_model,
                provider="openai",
                client=client,
                max_tokens=1024,
                reasoning_effort="low",
            )
            slots = asyncio.Semaphore(2)

            async def score_one(sample):
                async with slots:
                    return await score_limited(sample)

            async def score_limited(sample):
                number, question, reference, answer, contexts = sample
                print(f"Évaluation Ragas {number}/{len(samples)}", flush=True)
                faithfulness = Faithfulness(llm=faithfulness_evaluator)
                context_precision = ContextPrecision(llm=precision_evaluator)

                faithfulness.statement_generator_prompt.examples = []
                faithfulness.nli_statement_prompt.examples = []
                context_precision.prompt.examples = []
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

                print(
                    f"Scores {number}/{len(samples)} : fidélité {faithfulness_score:.2f}, "
                    f"précision {precision_score:.2f}",
                    flush=True,
                )
                return faithfulness_score, precision_score

            # Deux évaluations maximum en même temps pour économiser le quota
            scores = await asyncio.gather(*(score_one(sample) for sample in samples))
        return [score[0] for score in scores], [score[1] for score in scores]

    # Éxécution de la fonction définie juste avant
    faithfulness_scores, precision_scores = asyncio.run(score_samples())

    # Le test échoue si la qualité moyenne passe sous les seuils choisis.
    print(
        f"Moyennes : fidélité {mean(faithfulness_scores):.2f}, "
        f"précision {mean(precision_scores):.2f}",
        flush=True,
    )
    assert mean(faithfulness_scores) >= MIN_MEAN_FAITHFULNESS
    assert mean(precision_scores) >= MIN_MEAN_CONTEXT_PRECISION