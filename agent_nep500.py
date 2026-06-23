# -*- coding: utf-8 -*-
"""
Code Agent Dataiku — Analyse du caractère probant des éléments collectés (NEP-500)
==================================================================================

Cet agent :
  1. récupère des extraits d'un document stocké dans un Knowledge Bank Dataiku
     (recherche vectorielle) ;
  2. les analyse comme des « éléments collectés » au regard de la NEP-500
     à l'aide de Mistral Small 3.2 servi par le LLM Mesh ;
  3. renvoie la réponse en STREAMING (token par token).

Il s'appuie sur :
  - dataiku.llm.python.BaseLLM     -> la classe de base d'un *Code Agent* Dataiku
  - LangGraph (StateGraph)         -> orchestration retrieve -> analyze
  - dataiku.KnowledgeBank          -> accès au Knowledge Bank en vector store LangChain
  - LangchainToDKUTracer           -> traces remontées dans Dataiku

Pré-requis
----------
  - Créer un Code Agent : Flow > Add item > Generative AI > Code Agent
    (ou GenAI menu > Agents & GenAI Models > New Agent > Code Agent),
    puis coller ce code.
  - Un code env (Python >= 3.10) avec : langgraph, langchain, langchain-core
    (le code env interne « Retrieval augmented generation » convient aussi).
  - Une connexion LLM Mesh Mistral activée donnant accès à Mistral Small 3.2.
"""

import dataiku
from dataiku.llm.python import BaseLLM
from dataiku.langchain import LangchainToDKUTracer

from typing import List
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END


# =============================================================================
# Configuration
# =============================================================================

# ID du LLM dans le LLM Mesh. Format : "<type>:<nom_connexion>:<modele>".
# Pour Mistral Small 3.2, le modèle est typiquement "mistral-small-2506".
# Exécutez le bloc list_llms() en bas de fichier pour récupérer l'ID exact.
LLM_ID = "mistral:VOTRE_CONNEXION_MISTRAL:mistral-small-2506"  # <-- À ADAPTER

# Knowledge Bank contenant le document à analyser.
KB_ID = "NCVSUByT"

# Nombre d'extraits récupérés. Pour un KB ne contenant qu'un document, une
# valeur élevée revient à charger l'essentiel du document (Mistral Small 3.2
# accepte un large contexte). À ajuster selon la taille du document.
RETRIEVAL_K = 20

# Tag posé sur le LLM d'analyse : permet d'isoler SES tokens lors du streaming.
ANALYSIS_TAG = "nep500_analysis"


# =============================================================================
# Prompt NEP-500 (cadre d'analyse)
# =============================================================================

NEP500_SYSTEM_PROMPT = """Tu es un commissaire aux comptes expérimenté. Ta mission est d'apprécier le \
CARACTÈRE PROBANT d'éléments collectés au cours d'un audit des comptes, au regard de la norme \
d'exercice professionnel NEP-500 (« Caractère probant des éléments collectés »).

Applique systématiquement le cadre suivant :

1. CARACTÈRE APPROPRIÉ (qualité) = fiabilité + pertinence.
   Principes de fiabilité :
   - les éléments d'origine externe sont plus fiables que ceux d'origine interne ;
   - les éléments d'origine interne sont d'autant plus fiables que le contrôle interne est efficace ;
   - les éléments obtenus directement par le commissaire aux comptes (ex. observation physique) sont
     plus fiables que ceux obtenus par demande d'information ;
   - les éléments étayés par des documents sont plus fiables ;
   - les documents originaux sont plus fiables que les copies.
   Pertinence : l'élément se rapporte-t-il réellement à l'assertion testée ?

2. CARACTÈRE SUFFISANT (quantité) : la quantité d'éléments nécessaire dépend du risque d'anomalies
   significatives ET de la qualité des éléments collectés. Apprécie si la quantité paraît suffisante.

3. ASSERTIONS couvertes — précise lesquelles parmi :
   - Flux d'opérations / événements : réalité, exhaustivité, mesure, séparation des exercices, classification.
   - Soldes des comptes : existence, droits et obligations, exhaustivité, évaluation et imputation.
   - Présentation et annexe : réalité et droits et obligations, exhaustivité, présentation et
     intelligibilité, mesure et évaluation.

4. TECHNIQUES DE CONTRÔLE identifiables ou à mettre en œuvre :
   inspection des enregistrements/documents, inspection des actifs corporels, observation physique,
   demande d'information, demande de confirmation des tiers, vérification d'un calcul,
   ré-exécution de contrôles, procédures analytiques.

5. ESPRIT CRITIQUE : relève les indices pouvant remettre en cause la validité des éléments ;
   en cas de doute ou d'incohérence entre éléments, indique les procédures d'audit COMPLÉMENTAIRES
   à mettre en œuvre pour élucider l'incohérence.

Règles de rédaction :
- Réponds en français, de façon structurée et argumentée.
- Justifie chaque appréciation par les critères ci-dessus (origine, nature, circonstances, original/copie…).
- Ne te fonde QUE sur les éléments collectés fournis. Si une information manque, signale-le ;
  n'invente jamais un élément.
- Conclus sur la mesure dans laquelle les éléments sont suffisants et appropriés pour fonder l'opinion."""


HUMAN_TEMPLATE = """Mission : analyser les éléments collectés ci-dessous selon la NEP-500.

=== ÉLÉMENTS COLLECTÉS (extraits du document) ===
{context}
=== FIN DES ÉLÉMENTS ===

Demande de l'utilisateur :
{question}

Structure ta réponse ainsi :
1. Synthèse des éléments collectés identifiés
2. Caractère approprié (fiabilité & pertinence) — élément par élément
3. Caractère suffisant
4. Assertions couvertes
5. Techniques de contrôle (identifiées / recommandées)
6. Incohérences relevées et investigations complémentaires
7. Conclusion sur le caractère probant"""


# =============================================================================
# État du graphe LangGraph
# =============================================================================

class AnalysisState(TypedDict):
    question: str       # consigne / question de l'utilisateur
    context: str        # extraits récupérés du Knowledge Bank
    sources: List[str]  # métadonnées des extraits
    analysis: str       # analyse NEP-500 produite


# =============================================================================
# Helpers
# =============================================================================

def _get_project():
    return dataiku.api_client().get_default_project()


def _get_vectorstore():
    """Renvoie le Knowledge Bank sous forme de vector store LangChain."""
    project = _get_project()
    kb = dataiku.KnowledgeBank(id=KB_ID, project_key=project.project_key)
    return kb.as_langchain_vectorstore()


def _build_messages(question: str, context: str):
    if not context.strip():
        context = "(Aucun extrait pertinent n'a été récupéré dans le Knowledge Bank.)"
    return [
        SystemMessage(content=NEP500_SYSTEM_PROMPT),
        HumanMessage(content=HUMAN_TEMPLATE.format(context=context, question=question)),
    ]


# =============================================================================
# Construction du graphe : retrieve -> analyze
# =============================================================================

def build_graph(settings):
    # LLM Mesh -> chat model LangChain, en honorant les settings de la requête.
    # (Pour forcer une température basse : DKUChatModel(llm_id=LLM_ID, temperature=0).)
    llm = (
        _get_project()
        .get_llm(LLM_ID)
        .as_langchain_chat_model(completion_settings=settings)
        .with_config({"tags": [ANALYSIS_TAG]})
    )

    def retrieve_node(state: AnalysisState):
        vs = _get_vectorstore()
        # Pour cibler un document précis dans un KB multi-documents, ajoutez un filtre :
        # vs.similarity_search(state["question"], k=RETRIEVAL_K, filter={"source": "..."})
        docs = vs.similarity_search(state["question"], k=RETRIEVAL_K)
        context = "\n\n".join(
            f"[Extrait {i + 1}] {d.page_content}" for i, d in enumerate(docs)
        )
        sources = [str(d.metadata) for d in docs]
        return {"context": context, "sources": sources}

    # Noeud SYNCHRONE volontairement : il fonctionne aussi bien avec app.invoke()
    # (process) qu'avec app.astream_events() (aprocess_stream), et astream_events
    # émet quand même les tokens (on_chat_model_stream) d'un .invoke() interne.
    def analyze_node(state: AnalysisState):
        messages = _build_messages(state["question"], state["context"])
        response = llm.invoke(messages)
        return {"analysis": response.content}

    graph = StateGraph(AnalysisState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("analyze", analyze_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "analyze")
    graph.add_edge("analyze", END)
    return graph.compile()


# =============================================================================
# Le Code Agent Dataiku
# =============================================================================

class MyLLM(BaseLLM):
    def __init__(self):
        pass

    @staticmethod
    def _extract_question(query) -> str:
        """Récupère la dernière consigne utilisateur de la requête LLM Mesh."""
        msgs = [m for m in query["messages"] if m.get("content")]
        if msgs:
            return msgs[-1]["content"]
        return "Analyser les éléments collectés selon la NEP-500."

    # ---- Réponse synchrone (repli non-streaming) ----
    def process(self, query, settings, trace):
        question = self._extract_question(query)
        tracer = LangchainToDKUTracer(dku_trace=trace)
        app = build_graph(settings)
        result = app.invoke({"question": question}, config={"callbacks": [tracer]})
        return {"text": result["analysis"]}

    # ---- Réponse en streaming (cas du chat UI) ----
    async def aprocess_stream(self, query, settings, trace):
        question = self._extract_question(query)
        tracer = LangchainToDKUTracer(dku_trace=trace)
        app = build_graph(settings)

        async for event in app.astream_events(
            {"question": question},
            version="v2",
            config={"callbacks": [tracer]},
        ):
            # On ne streame que les tokens du LLM d'analyse (filtré par tag).
            if event["event"] == "on_chat_model_stream" and ANALYSIS_TAG in event.get("tags", []):
                content = event["data"]["chunk"].content
                if content:
                    yield {"chunk": {"text": content}}


# =============================================================================
# Utilitaire : lister les LLM disponibles pour trouver l'ID de Mistral Small 3.2
# (à exécuter une fois dans un notebook, puis reporter la valeur dans LLM_ID)
# =============================================================================
#
# import dataiku
# project = dataiku.api_client().get_default_project()
# for llm in project.list_llms():
#     print(f"- {llm.description} (id: {llm.id})")
