# -*- coding: utf-8 -*-
"""
CODE AGENT DATAIKU — Gestion de projet multi-agents (LangGraph + BaseLLM)
Version STREAMING : la sortie est diffusée en temps réel, avec un titre
affiché à chaque fois qu'un nouvel agent prend la main.
==========================================================================
À coller dans : votre projet > Agents > + New agent > Code agent

Pipeline : Besoin -> Architecture -> Revue sécurité -> Cahier des charges
Boucle : si risque critique, retour chez l'architecte (max MAX_ITERATIONS).

PRÉREQUIS
---------
1. Code env Python (>= 3.9) avec : langgraph (version récente >= 0.2),
   langchain, langchain-core. À sélectionner dans les Settings de l'agent.
2. Remplacer LLM_ID par un LLM de votre Mesh (idéalement un modèle qui
   supporte le streaming). Pour lister les IDs :
       import dataiku
       p = dataiku.api_client().get_default_project()
       for l in p.list_llms(): print(l.id, "-", l.description)
"""

from typing import TypedDict

import dataiku
from dataiku.llm.python import BaseLLM
from dataiku.langchain.dku_llm import DKUChatLLM
from dataiku.langchain import LangchainToDKUTracer
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
LLM_ID = "openai:ma-connexion:gpt-4o"   # <-- REMPLACER par un LLM de votre Mesh
MAX_ITERATIONS = 2                       # allers-retours max archi <-> sécurité
STREAM_INTERMEDIAIRES = True             # False = ne streamer que le CDC final

# Noms affichés dans la sortie quand un agent prend la main
NOMS_AGENTS = {
    "besoin": "🔍 Agent 1 — Analyse du besoin",
    "architecture": "🏗️ Agent 2 — Architecture",
    "securite": "🔐 Agent 3 — Revue sécurité",
    "cahier_des_charges": "📝 Agent 4 — Cahier des charges",
}

# ---------------------------------------------------------------
# Prompts système des 4 sous-agents
# ---------------------------------------------------------------
PROMPT_BESOIN = (
    "Tu es un business analyst senior. À partir d'une demande brute, tu produis "
    "une expression de besoin structurée : contexte, objectifs, périmètre "
    "(inclus/exclus), utilisateurs cibles, exigences fonctionnelles, exigences "
    "non fonctionnelles, contraintes, critères de succès. Si des informations "
    "manquent, formule des hypothèses explicites dans une section dédiée."
)

PROMPT_ARCHITECTE = (
    "Tu es un architecte solution senior. Tu conçois une architecture technique : "
    "composants, flux de données, choix technologiques justifiés, intégrations, "
    "schéma logique (diagramme mermaid en texte), hébergement, scalabilité, "
    "et principes de sécurité by design."
)

PROMPT_SECURITE = (
    "Tu es un expert cybersécurité (RSSI). Tu audites une architecture technique. "
    "Produis un rapport de risques par catégorie : protection des données, IAM, "
    "réseau, conformité (RGPD...), supply chain, résilience. Pour chaque risque : "
    "criticité (faible / moyen / élevé / critique) et recommandation. "
    "Termine ta réponse par exactement une ligne : "
    "'VERDICT: VALIDE' (si aucun risque critique) ou 'VERDICT: REVISION' (sinon)."
)

PROMPT_CDC = (
    "Tu es un chef de projet / PMO senior. Rédige un cahier des charges complet "
    "et professionnel en Markdown avec les sections : 1. Contexte et objectifs, "
    "2. Périmètre, 3. Exigences fonctionnelles, 4. Exigences non fonctionnelles, "
    "5. Architecture technique, 6. Sécurité et conformité, 7. Planning macro et "
    "jalons, 8. Estimation des charges, 9. Risques et plan de mitigation, "
    "10. Critères d'acceptation."
)


# ---------------------------------------------------------------
# État partagé entre les sous-agents
# ---------------------------------------------------------------
class ProjectState(TypedDict):
    demande_initiale: str
    besoins: str
    architecture: str
    revue_securite: str
    securite_validee: bool
    cahier_des_charges: str
    iterations: int


# ---------------------------------------------------------------
# Le Code Agent Dataiku
# ---------------------------------------------------------------
class MyLLM(BaseLLM):
    """Agent de gestion de projet : orchestre 4 sous-agents via LangGraph,
    avec sortie streamée agent par agent, token par token."""

    def __init__(self):
        self.llm = DKUChatLLM(llm_id=LLM_ID, temperature=0.2)
        self.graph = self._build_graph()

    # ------------------------------------------------------------
    # Helper : appel du LLM en mode stream (les tokens émis ici sont
    # captés par graph.stream(stream_mode="messages") et re-diffusés)
    # ------------------------------------------------------------
    def _call_llm(self, system_prompt: str, user_prompt: str,
                  config: RunnableConfig = None) -> str:
        morceaux = []
        for chunk in self.llm.stream(
            [SystemMessage(content=system_prompt),
             HumanMessage(content=user_prompt)],
            config=config,
        ):
            morceaux.append(self._texte(chunk))
        return "".join(morceaux)

    @staticmethod
    def _texte(message) -> str:
        """Extrait le texte d'un message/chunk LangChain (str ou liste de blocs)."""
        content = getattr(message, "content", "") or ""
        if isinstance(content, list):
            content = "".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        return content

    # ------------------------------------------------------------
    # Les 4 sous-agents (nœuds du graphe)
    # ------------------------------------------------------------
    def _agent_besoin(self, state: ProjectState, config: RunnableConfig) -> dict:
        besoins = self._call_llm(
            PROMPT_BESOIN,
            f"Demande initiale du client :\n{state['demande_initiale']}",
            config,
        )
        return {"besoins": besoins}

    def _agent_architecture(self, state: ProjectState, config: RunnableConfig) -> dict:
        user = f"Besoins structurés :\n{state['besoins']}"
        if state.get("revue_securite") and not state.get("securite_validee", True):
            user += (
                f"\n\nVersion précédente de l'architecture :\n{state['architecture']}"
                f"\n\nRemarques de l'équipe sécurité à corriger IMPÉRATIVEMENT :\n"
                f"{state['revue_securite']}"
            )
        architecture = self._call_llm(PROMPT_ARCHITECTE, user, config)
        return {"architecture": architecture,
                "iterations": state.get("iterations", 0) + 1}

    def _agent_securite(self, state: ProjectState, config: RunnableConfig) -> dict:
        user = (
            f"Besoins :\n{state['besoins']}\n\n"
            f"Architecture à auditer :\n{state['architecture']}"
        )
        revue = self._call_llm(PROMPT_SECURITE, user, config)
        return {"revue_securite": revue,
                "securite_validee": "VERDICT: VALIDE" in revue.upper()}

    def _agent_cahier_des_charges(self, state: ProjectState,
                                  config: RunnableConfig) -> dict:
        user = (
            f"Besoins :\n{state['besoins']}\n\n"
            f"Architecture retenue :\n{state['architecture']}\n\n"
            f"Revue sécurité :\n{state['revue_securite']}"
        )
        cdc = self._call_llm(PROMPT_CDC, user, config)
        return {"cahier_des_charges": cdc}

    # ------------------------------------------------------------
    # Routage conditionnel après l'audit sécurité
    # ------------------------------------------------------------
    def _route_apres_securite(self, state: ProjectState) -> str:
        if state["securite_validee"] or state["iterations"] >= MAX_ITERATIONS:
            return "cahier_des_charges"
        return "architecture"

    # ------------------------------------------------------------
    # Construction du graphe LangGraph
    # ------------------------------------------------------------
    def _build_graph(self):
        wf = StateGraph(ProjectState)
        wf.add_node("besoin", self._agent_besoin)
        wf.add_node("architecture", self._agent_architecture)
        wf.add_node("securite", self._agent_securite)
        wf.add_node("cahier_des_charges", self._agent_cahier_des_charges)

        wf.set_entry_point("besoin")
        wf.add_edge("besoin", "architecture")
        wf.add_edge("architecture", "securite")
        wf.add_conditional_edges(
            "securite",
            self._route_apres_securite,
            {"architecture": "architecture",
             "cahier_des_charges": "cahier_des_charges"},
        )
        wf.add_edge("cahier_des_charges", END)
        return wf.compile()

    # ------------------------------------------------------------
    # Utilitaire : dernier message utilisateur
    # ------------------------------------------------------------
    @staticmethod
    def _extraire_demande(query) -> str:
        content = query["messages"][-1]["content"]
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        return content

    # ==============================================================
    # POINT D'ENTRÉE STREAMING (contrat BaseLLM de Dataiku)
    # Chaque chunk émis est un dict {"chunk": {"text": "..."}}
    # ==============================================================
    def process_stream(self, query, settings, trace):
        demande = self._extraire_demande(query)
        tracer = LangchainToDKUTracer(dku_trace=trace)

        etat_initial: ProjectState = {
            "demande_initiale": demande,
            "besoins": "",
            "architecture": "",
            "revue_securite": "",
            "securite_validee": False,
            "cahier_des_charges": "",
            "iterations": 0,
        }

        noeud_courant = None
        passages = {}  # compte les passages par agent (pour marquer les révisions)

        # stream_mode="messages" : LangGraph re-émet chaque token produit
        # par les LLM des nœuds, avec metadata["langgraph_node"] = nom du nœud
        for token, metadata in self.graph.stream(
            etat_initial,
            config={"callbacks": [tracer]},
            stream_mode="messages",
        ):
            noeud = metadata.get("langgraph_node")

            # Filtrage éventuel : ne streamer que le livrable final
            if not STREAM_INTERMEDIAIRES and noeud != "cahier_des_charges":
                continue

            # Nouvel agent qui prend la main -> on affiche son nom
            if noeud and noeud != noeud_courant:
                noeud_courant = noeud
                passages[noeud] = passages.get(noeud, 0) + 1
                titre = NOMS_AGENTS.get(noeud, noeud)
                if passages[noeud] > 1:
                    titre += f" (révision n°{passages[noeud] - 1})"
                yield {"chunk": {"text": f"\n\n---\n\n## {titre}\n\n"}}

            # Puis le contenu, token par token
            texte = self._texte(token)
            if texte:
                yield {"chunk": {"text": texte}}

    # ==============================================================
    # POINT D'ENTRÉE NON-STREAMÉ : agrège simplement le stream,
    # pour les contextes qui n'utilisent pas le streaming
    # ==============================================================
    def process(self, query, settings, trace):
        texte = "".join(
            r["chunk"]["text"]
            for r in self.process_stream(query, settings, trace)
            if "chunk" in r and "text" in r["chunk"]
        )
        return {"text": texte}
