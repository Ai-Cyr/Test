# -*- coding: utf-8 -*-
"""
Assistant Data Analyste — Code Agent Dataiku (BaseLLM) + LangGraph
==================================================================
Architecture du graphe :

    START ──> [think] ──> [act] ──┬──> [tools] ──> [think] (boucle)
                                  └──> END (réponse finale)

- [think] : nœud LLM dédié au raisonnement (PAS un tool).
            Sa sortie est streamée à l'utilisateur en temps réel.
- [act]   : nœud LLM avec tools, décide d'exécuter du SQL ou de répondre.
            La réponse finale est streamée aussi.
- [tools] : exécution SQL (SELECT uniquement) via SQLExecutor2.

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
MAX_ITERATIONS = 6  # nombre max de cycles think -> act -> tools

# Décrivez ici vos tables pour aider l'agent à écrire de bonnes requêtes
SCHEMA_DESCRIPTION = """
Tables disponibles :
- customers(customer_id, name, country, created_at)
- orders(order_id, customer_id, amount, order_date, status)
"""

THINK_PROMPT = """Tu es le module de raisonnement d'un assistant data analyste.
Ton rôle : réfléchir à voix haute, étape par étape, AVANT toute action.

{schema}

Analyse la demande de l'utilisateur et les éventuels résultats SQL déjà obtenus, puis :
1. Reformule ce que l'utilisateur cherche à savoir.
2. Identifie les tables/colonnes pertinentes.
3. Planifie la (les) requête(s) SQL nécessaire(s), ou indique si les données
   déjà obtenues suffisent pour répondre.
4. Si une requête a échoué, diagnostique l'erreur et propose une correction.

Sois concis (5-10 lignes max), structuré, en français.
Ne réponds PAS à la question : tu ne fais que raisonner."""

ACT_PROMPT = """Tu es un assistant data analyste connecté à une base SQL.

{schema}

Voici ton raisonnement préalable :
---
{reasoning}
---

Suis ce plan :
- S'il faut interroger la base, appelle l'outil `execute_sql` (SELECT uniquement).
- Si tu as déjà tous les résultats nécessaires, donne ta réponse finale à
  l'utilisateur : claire, en français, avec les chiffres clés et, si pertinent,
  un tableau markdown et une courte interprétation analytique.
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
# ÉTAT
# ==============================================================

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    reasoning: str        # dernier raisonnement produit par le nœud think
    iterations: int

# ==============================================================
# GRAPHE
# ==============================================================

def build_graph(llm, llm_with_tools):

    # ---------- Nœud THINK : raisonnement streamé ----------
    def think_node(state: AgentState):
        prompt = [
            SystemMessage(content=THINK_PROMPT.format(schema=SCHEMA_DESCRIPTION))
        ] + state["messages"]
        # .invoke() suffit : avec stream_mode="messages", LangGraph streame
        # quand même les tokens de cet appel LLM vers l'extérieur.
        response = llm.invoke(prompt)
        return {
            "reasoning": response.content,
            "iterations": state.get("iterations", 0) + 1,
        }

    # ---------- Nœud ACT : décision tool ou réponse finale ----------
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
        return {"messages": [response]}

    # ---------- Nœud TOOLS : exécution SQL ----------
    def tool_node(state: AgentState):
        last = state["messages"][-1]
        results = []
        for tc in last.tool_calls:
            output = TOOLS_BY_NAME[tc["name"]].invoke(tc["args"])
            results.append(
                ToolMessage(content=str(output), tool_call_id=tc["id"], name=tc["name"])
            )
        return {"messages": results}

    # ---------- Routage ----------
    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if state.get("iterations", 0) >= MAX_ITERATIONS:
            return END
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    builder = StateGraph(AgentState)
    builder.add_node("think", think_node)
    builder.add_node("act", act_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "think")
    builder.add_edge("think", "act")
    builder.add_conditional_edges("act", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "think")  # après le SQL, on re-raisonne

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

        current_node = None  # pour détecter les transitions entre nœuds

        for msg_chunk, metadata in self.graph.stream(state, stream_mode="messages"):
            node = metadata.get("langgraph_node")

            # ----- Tokens du nœud THINK : raisonnement streamé -----
            if node == "think" and isinstance(msg_chunk, (AIMessage, AIMessageChunk)):
                if current_node != "think":
                    yield {"chunk": {"text": "\n\n**🧠 Raisonnement**\n\n*"}}
                    current_node = "think"
                if msg_chunk.content:
                    yield {"chunk": {"text": msg_chunk.content}}

            # ----- Tokens du nœud ACT : réponse finale streamée -----
            elif node == "act" and isinstance(msg_chunk, (AIMessage, AIMessageChunk)):
                # On ignore les chunks de tool calls (pas de texte utile)
                if msg_chunk.content and not getattr(msg_chunk, "tool_calls", None):
                    if current_node != "act":
                        # ferme l'italique du raisonnement, ouvre la réponse
                        yield {"chunk": {"text": "*\n\n---\n\n**📊 Réponse**\n\n"}}
                        current_node = "act"
                    yield {"chunk": {"text": msg_chunk.content}}

            # ----- Notification d'exécution SQL -----
            elif node == "tools" and isinstance(msg_chunk, ToolMessage):
                if current_node == "think":
                    yield {"chunk": {"text": "*"}}  # ferme l'italique
                yield {"chunk": {"text": "\n\n> 🔧 *Exécution de la requête SQL...*\n"}}
                current_node = "tools"
