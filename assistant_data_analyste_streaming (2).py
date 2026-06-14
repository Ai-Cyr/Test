# -*- coding: utf-8 -*-
"""
Assistant Data Analyste — Code Agent Dataiku (BaseLLM) + LangGraph
==================================================================
Deux nœuds de raisonnement DISTINCTS, chacun avec son propre prompt :

    START -> [think_initial] -> [act] -+-> [tools] -> [think_after_sql] -> [act]
                                       +-> END

- [think_initial]   : raisonnement AVANT toute requête (planification).
- [think_after_sql] : raisonnement APRES exécution SQL (interprétation /
                      diagnostic / décision de continuer ou conclure).
- [act]             : décide d'exécuter du SQL ou de donner la réponse finale.
- [tools]           : exécution SQL (SELECT uniquement) via SQLExecutor2.

Les deux raisonnements ET la réponse finale sont streamés à l'utilisateur.

À adapter : LLM_ID, SQL_CONNECTION, SCHEMA_DESCRIPTION.
"""

import dataiku
from dataiku import SQLExecutor2
from dataiku.llm.python import BaseLLM

from typing import Annotated, TypedDict
from langchain_core.messages import (
    AnyMessage,
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ==============================================================
# CONFIGURATION
# ==============================================================

LLM_ID = "mistral:votre-connexion:mistral-small-latest"
SQL_CONNECTION = "votre_connexion_sql"
MAX_ROWS = 50
MAX_ITERATIONS = 6  # nombre max d'appels au noeud "act"

SCHEMA_DESCRIPTION = """
Tables disponibles :
- customers(customer_id, name, country, created_at)
- orders(order_id, customer_id, amount, order_date, status)
"""

# ---- Prompt du PREMIER raisonnement (planification) ----------
THINK_INITIAL_PROMPT = """Tu es le module de raisonnement d'un assistant data analyste.
Tu interviens AVANT toute requête, pour planifier.

{schema}

Analyse la demande de l'utilisateur, puis :
1. Reformule ce qu'il cherche réellement à savoir.
2. Identifie les tables et colonnes pertinentes.
3. Décris la (les) requête(s) SQL à écrire (logique, jointures, agrégats, filtres).

Sois concis (5-8 lignes), structuré, en français.
Ne réponds PAS à la question et n'écris PAS encore le SQL final : tu planifies."""

# ---- Prompt du SECOND raisonnement (post-SQL) ---------------
THINK_AFTER_SQL_PROMPT = """Tu es le module de raisonnement d'un assistant data analyste.
Tu interviens APRES l'exécution d'une ou plusieurs requêtes SQL, dont les
résultats figurent dans l'historique de conversation ci-dessus.

{schema}

À partir des résultats obtenus :
1. Vérifie que les données répondent bien à la question initiale.
2. Repère toute anomalie (valeurs nulles, résultat vide, chiffres incohérents).
3. Si une requête a échoué, diagnostique précisément l'erreur et indique
   la correction à apporter.
4. Conclus : soit les données suffisent pour répondre (dis-le), soit une
   requête complémentaire est nécessaire (précise laquelle et pourquoi).

Sois concis (4-7 lignes), en français. Tu ne réponds PAS encore à l'utilisateur."""

# ---- Prompt du noeud d'action -------------------------------
ACT_PROMPT = """Tu es un assistant data analyste connecté à une base SQL.

{schema}

Raisonnement le plus récent à suivre :
---
{reasoning}
---

- S'il faut interroger la base, appelle l'outil `execute_sql` (SELECT uniquement).
- Si tu as déjà tous les résultats nécessaires, donne ta réponse finale :
  claire, en français, avec les chiffres clés, un tableau markdown si pertinent,
  et une courte interprétation analytique.
- Ne fabrique jamais de données."""

# ==============================================================
# TOOL SQL
# ==============================================================

@tool
def execute_sql(query: str) -> str:
    """Exécute une requête SQL en lecture (SELECT) et retourne les résultats.

    Args:
        query: La requête SQL à exécuter (SELECT uniquement).
    """
    forbidden = ["insert", "update", "delete", "drop", "alter", "truncate", "create", "grant"]
    q_lower = query.strip().lower()
    if not (q_lower.startswith("select") or q_lower.startswith("with")):
        return "Erreur : seules les requêtes SELECT (ou CTE WITH) sont autorisées."
    if any(kw in q_lower.split() for kw in forbidden):
        return "Erreur : requête refusée (mot-clé d'écriture détecté)."
    try:
        executor = SQLExecutor2(connection=SQL_CONNECTION)
        df = executor.query_to_df(query)
        if df.empty:
            return "La requête n'a retourné aucun résultat."
        truncated = ""
        if len(df) > MAX_ROWS:
            df = df.head(MAX_ROWS)
            truncated = f"\n(Résultats tronqués aux {MAX_ROWS} premières lignes)"
        return df.to_markdown(index=False) + truncated
    except Exception as e:
        return f"Erreur SQL : {str(e)}"


TOOLS = [execute_sql]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

# ==============================================================
# ETAT
# ==============================================================

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    reasoning: str        # dernier raisonnement (initial ou post-SQL)
    iterations: int

# ==============================================================
# GRAPHE
# ==============================================================

def build_graph(llm, llm_with_tools):

    # ---------- Noeud THINK INITIAL ----------
    def think_initial_node(state: AgentState):
        prompt = [
            SystemMessage(content=THINK_INITIAL_PROMPT.format(schema=SCHEMA_DESCRIPTION))
        ] + state["messages"]
        response = llm.invoke(prompt)
        return {"reasoning": response.content}

    # ---------- Noeud THINK APRES SQL ----------
    def think_after_sql_node(state: AgentState):
        prompt = [
            SystemMessage(content=THINK_AFTER_SQL_PROMPT.format(schema=SCHEMA_DESCRIPTION))
        ] + state["messages"]
        response = llm.invoke(prompt)
        return {"reasoning": response.content}

    # ---------- Noeud ACT ----------
    def act_node(state: AgentState):
        prompt = [
            SystemMessage(
                content=ACT_PROMPT.format(
                    schema=SCHEMA_DESCRIPTION,
                    reasoning=state.get("reasoning", ""),
                )
            )
        ] + state["messages"]
        response = llm_with_tools.invoke(prompt)
        return {"messages": [response], "iterations": state.get("iterations", 0) + 1}

    # ---------- Noeud TOOLS ----------
    def tool_node(state: AgentState):
        last = state["messages"][-1]
        results = []
        for tc in last.tool_calls:
            output = TOOLS_BY_NAME[tc["name"]].invoke(tc["args"])
            results.append(
                ToolMessage(content=str(output), tool_call_id=tc["id"], name=tc["name"])
            )
        return {"messages": results}

    # ---------- Routage apres ACT ----------
    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if state.get("iterations", 0) >= MAX_ITERATIONS:
            return END
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    builder = StateGraph(AgentState)
    builder.add_node("think_initial", think_initial_node)
    builder.add_node("think_after_sql", think_after_sql_node)
    builder.add_node("act", act_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "think_initial")
    builder.add_edge("think_initial", "act")
    builder.add_conditional_edges("act", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "think_after_sql")   # apres le SQL -> 2e raisonnement
    builder.add_edge("think_after_sql", "act")

    return builder.compile()

# ==============================================================
# CODE AGENT DATAIKU
# ==============================================================

class MyLLM(BaseLLM):
    def __init__(self):
        client = dataiku.api_client()
        project = client.get_default_project()
        dku_llm = project.get_llm(LLM_ID)
        self.llm = dku_llm.as_langchain_chat_model()
        self.llm_with_tools = self.llm.bind_tools(TOOLS)
        self.graph = build_graph(self.llm, self.llm_with_tools)

    def _to_langchain_messages(self, query):
        lc_messages = []
        for msg in query.get("messages", []):
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
        return lc_messages

    # ---------- Mode non-streaming (obligatoire) ----------
    def process(self, query, settings, trace):
        state = {
            "messages": self._to_langchain_messages(query),
            "reasoning": "",
            "iterations": 0,
        }
        final_state = self.graph.invoke(state)
        return {"text": final_state["messages"][-1].content}

    # ---------- Mode STREAMING ----------
    def process_stream(self, query, settings, trace):
        state = {
            "messages": self._to_langchain_messages(query),
            "reasoning": "",
            "iterations": 0,
        }

        # En-tetes affiches a l'entree de chaque noeud "parlant"
        HEADERS = {
            "think_initial":   "\n\n**🧠 Réflexion — planification**\n\n*",
            "think_after_sql": "\n\n**🧠 Réflexion — analyse des résultats**\n\n*",
            "act":             "\n\n---\n\n**📊 Réponse**\n\n",
        }

        current = None       # noeud "parlant" en cours
        italic_open = False  # un bloc de raisonnement en italique est-il ouvert ?

        def close_italic():
            nonlocal italic_open
            if italic_open:
                italic_open = False
                return "*"
            return ""

        for msg_chunk, metadata in self.graph.stream(state, stream_mode="messages"):
            node = metadata.get("langgraph_node")

            # ----- Raisonnement (les 2 noeuds think) -----
            if node in ("think_initial", "think_after_sql") and isinstance(
                msg_chunk, (AIMessage, AIMessageChunk)
            ):
                if msg_chunk.content:
                    if current != node:
                        prefix = close_italic()
                        yield {"chunk": {"text": prefix + HEADERS[node]}}
                        italic_open = True
                        current = node
                    yield {"chunk": {"text": msg_chunk.content}}

            # ----- Reponse finale (noeud act) -----
            elif node == "act" and isinstance(msg_chunk, (AIMessage, AIMessageChunk)):
                if msg_chunk.content and not getattr(msg_chunk, "tool_calls", None):
                    if current != node:
                        prefix = close_italic()
                        yield {"chunk": {"text": prefix + HEADERS[node]}}
                        current = node
                    yield {"chunk": {"text": msg_chunk.content}}

            # ----- Notification d'execution SQL -----
            elif node == "tools" and isinstance(msg_chunk, ToolMessage):
                prefix = close_italic()
                yield {"chunk": {"text": prefix + "\n\n> 🔧 *Exécution de la requête SQL...*\n"}}
                current = "tools"
