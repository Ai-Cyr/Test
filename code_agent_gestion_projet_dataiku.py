# -*- coding: utf-8 -*-
"""
CODE AGENT DATAIKU — Gestion de projet multi-agents (LangGraph + BaseLLM)
==========================================================================
À coller dans : votre projet > Agents > + New agent > Code agent

Pipeline interne : Besoin -> Architecture -> Revue sécurité -> Cahier des charges
Boucle : si l'agent sécurité détecte un risque critique, l'architecture
repart en révision chez l'architecte (max MAX_ITERATIONS allers-retours).

Une fois créé, l'agent est exposé dans le LLM Mesh comme un LLM classique
(id de la forme "agent:xxxxx") : utilisable dans Prompt Studios, Dataiku
Answers, recettes LLM, webapps, API...

PRÉREQUIS
---------
1. Un code env Python (>= 3.9) contenant : langgraph, langchain, langchain-core
   (à sélectionner dans les Settings de l'agent).
2. Remplacer LLM_ID par un LLM de votre Mesh. Pour lister les IDs :
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
MAX_ITERATIONS = 2                       # allers-retours max architecture <-> sécurité

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
    demande_initiale: str      # brief brut de l'utilisateur
    besoins: str               # sortie agent 1
    architecture: str          # sortie agent 2
    revue_securite: str        # sortie agent 3
    securite_validee: bool     # verdict de l'agent sécurité
    cahier_des_charges: str    # sortie agent 4 (livrable final)
    iterations: int            # compteur de boucles archi <-> sécu


# ---------------------------------------------------------------
# Le Code Agent Dataiku
# ---------------------------------------------------------------
class MyLLM(BaseLLM):
    """Agent de gestion de projet : orchestre 4 sous-agents via LangGraph."""

    def __init__(self):
        self.llm = DKUChatLLM(llm_id=LLM_ID, temperature=0.2)
        self.graph = self._build_graph()

    # ------------------------------------------------------------
    # Helper : appel du LLM du Mesh (config propagée pour le tracing)
    # ------------------------------------------------------------
    def _call_llm(self, system_prompt: str, user_prompt: str,
                  config: RunnableConfig = None) -> str:
        response = self.llm.invoke(
            [SystemMessage(content=system_prompt),
             HumanMessage(content=user_prompt)],
            config=config,
        )
        return response.content

    # ------------------------------------------------------------
    # Les 4 sous-agents (nœuds du graphe)
    # ------------------------------------------------------------
    def _agent_besoin(self, state: ProjectState, config: RunnableConfig) -> dict:
        """Agent 1 — Business Analyst : structure le besoin."""
        besoins = self._call_llm(
            PROMPT_BESOIN,
            f"Demande initiale du client :\n{state['demande_initiale']}",
            config,
        )
        return {"besoins": besoins}

    def _agent_architecture(self, state: ProjectState, config: RunnableConfig) -> dict:
        """Agent 2 — Architecte : conçoit (ou révise) l'architecture."""
        user = f"Besoins structurés :\n{state['besoins']}"

        # Retour d'une revue sécurité négative -> on intègre les remarques
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
        """Agent 3 — RSSI : audite les risques de sécurité."""
        user = (
            f"Besoins :\n{state['besoins']}\n\n"
            f"Architecture à auditer :\n{state['architecture']}"
        )
        revue = self._call_llm(PROMPT_SECURITE, user, config)
        validee = "VERDICT: VALIDE" in revue.upper()
        return {"revue_securite": revue, "securite_validee": validee}

    def _agent_cahier_des_charges(self, state: ProjectState,
                                  config: RunnableConfig) -> dict:
        """Agent 4 — PMO : rédige le cahier des charges final."""
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
            {
                "architecture": "architecture",              # risque critique
                "cahier_des_charges": "cahier_des_charges",  # validé
            },
        )
        wf.add_edge("cahier_des_charges", END)
        return wf.compile()

    # ------------------------------------------------------------
    # Utilitaire : extraire le dernier message utilisateur
    # ------------------------------------------------------------
    @staticmethod
    def _extraire_demande(query) -> str:
        content = query["messages"][-1]["content"]
        if isinstance(content, list):  # contenu multimodal éventuel
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        return content

    # ==============================================================
    # POINT D'ENTRÉE DU CODE AGENT (contrat BaseLLM de Dataiku)
    # ==============================================================
    def process(self, query, settings, trace):
        # 1. Récupérer la demande de l'utilisateur
        demande = self._extraire_demande(query)

        # 2. Brancher le tracing Dataiku sur LangGraph/LangChain :
        #    chaque nœud et chaque appel LLM apparaîtra dans la trace
        #    de l'agent (visible dans le playground / Trace Explorer)
        tracer = LangchainToDKUTracer(dku_trace=trace)

        # 3. Exécuter le pipeline multi-agents
        etat_initial: ProjectState = {
            "demande_initiale": demande,
            "besoins": "",
            "architecture": "",
            "revue_securite": "",
            "securite_validee": False,
            "cahier_des_charges": "",
            "iterations": 0,
        }
        resultat = self.graph.invoke(
            etat_initial,
            config={"callbacks": [tracer]},
        )

        # 4. Construire la réponse finale
        verdict = (
            "validée par la revue sécurité"
            if resultat["securite_validee"]
            else "publiée SOUS RÉSERVE : des risques critiques restent à lever"
        )
        entete = (
            f"_Architecture {verdict} — "
            f"{resultat['iterations']} itération(s) architecture/sécurité._\n\n---\n\n"
        )

        return {"text": entete + resultat["cahier_des_charges"]}
