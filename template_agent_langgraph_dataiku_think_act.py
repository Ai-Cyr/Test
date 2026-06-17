# -*- coding: utf-8 -*-
"""
Template Agent Dataiku + LangGraph
==================================

Pattern conservé :

    START -> think -> act -> tools -> think -> act -> ... -> END

Objectif : un squelette réutilisable pour des agents Dataiku Code Agent
avec Mistral Small 3.2 ou tout autre modèle exposé via le LLM Mesh.

Points clés :
- `think` produit un plan/contrôle visible court, pas un raisonnement interne détaillé.
- `act` décide soit d'appeler un tool, soit de répondre à l'utilisateur.
- `tools` exécute les actions autorisées, puis on revient à `think` pour analyser le résultat.
- `process` et `process_stream` sont fournis pour Dataiku BaseLLM.

À adapter dans chaque projet :
- LLM_ID
- SQL_CONNECTION
- SCHEMA_DESCRIPTION
- TOOLS
- Prompts métier
"""

from __future__ import annotations

import re
from typing import Annotated, Any, TypedDict

import dataiku
from dataiku import SQLExecutor2
from dataiku.llm.python import BaseLLM

from langchain_core.messages import (
    AnyMessage,
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# =============================================================================
# CONFIGURATION PROJET
# =============================================================================

# Exemple Dataiku. Le nom exact dépend de la connexion LLM Mesh déclarée chez toi.
# Pour Mistral Small 3.2, l'ID provider officiel est souvent `mistral-small-2506`,
# mais Dataiku peut exposer un alias différent selon la configuration de l'admin.
LLM_ID = "mistral:votre-connexion:mistral-small-2506"

SQL_CONNECTION = "votre_connexion_sql"
MAX_ROWS = 50
MAX_TOOL_LOOPS = 4

# Afficher la note du noeud think dans le streaming.
# Conseil : garder True en dev / audit, éventuellement False en production.
STREAM_VISIBLE_THINK = True

# Afficher le résultat brut des tools dans le chat.
# Conseil : garder False si les résultats peuvent contenir des données sensibles.
STREAM_TOOL_OUTPUT = False

SCHEMA_DESCRIPTION = """
Tables disponibles :
- customers(customer_id, name, country, created_at)
- orders(order_id, customer_id, amount, order_date, status)

Règles métier utiles :
- `orders.amount` est exprimé en euros.
- `orders.status = 'paid'` signifie une commande payée.
"""

VISIBLE_THINK_PROMPT = """Tu es le module de cadrage et de contrôle d'un agent data analyste.

Important : produis une note visible et concise pour l'utilisateur, pas un raisonnement interne détaillé.

{schema}

À chaque passage dans ce noeud :
1. Résume l'objectif utilisateur en une phrase.
2. Indique les données ou résultats déjà disponibles.
3. Indique la prochaine action logique : requête SQL, correction SQL, ou réponse finale.
4. Après un résultat de tool, vérifie si le résultat suffit ou s'il faut une nouvelle action.

Format attendu :
- 3 à 6 lignes maximum.
- Français.
- Ne donne pas encore la réponse finale métier.
"""

ACT_PROMPT = """Tu es un assistant data analyste connecté à une base SQL.

{schema}

Note de cadrage/contrôle produite juste avant :
---
{visible_think}
---

Règles :
- Si une donnée manque, appelle l'outil `execute_sql`.
- Utilise uniquement des requêtes SELECT ou WITH.
- Si les résultats disponibles suffisent, réponds clairement en français.
- Ne fabrique jamais de données.
- Si un outil retourne une erreur, corrige la requête ou explique la limite.
- Pour une réponse finale, donne les chiffres clés, un tableau markdown si utile, puis une courte interprétation.
"""

FINALIZE_PROMPT = """Tu es un assistant data analyste.

La limite de boucles d'outils a été atteinte ou l'agent ne peut plus avancer de façon fiable.
Réponds à l'utilisateur avec ce qui est disponible, sans inventer de données.
Explique brièvement la limite rencontrée et propose une prochaine action concrète.
"""

# =============================================================================
# OUTILS
# =============================================================================


def _strip_sql_comments(query: str) -> str:
    """Supprime les commentaires SQL simples pour faciliter les contrôles de sécurité."""
    query = re.sub(r"--.*?$", "", query, flags=re.MULTILINE)
    query = re.sub(r"/\*.*?\*/", "", query, flags=re.DOTALL)
    return query.strip()


def _validate_read_only_sql(query: str) -> tuple[bool, str]:
    """Validation défensive basique : lecture seule, un seul statement."""
    cleaned = _strip_sql_comments(query)
    if not cleaned:
        return False, "requête vide."

    # Autorise un unique point-virgule final, refuse les multi-statements.
    without_final_semicolon = cleaned[:-1].strip() if cleaned.endswith(";") else cleaned
    if ";" in without_final_semicolon:
        return False, "multi-statements refusés."

    lowered = without_final_semicolon.lower()
    if not re.match(r"^(select|with)\b", lowered):
        return False, "seules les requêtes SELECT ou CTE WITH sont autorisées."

    forbidden = (
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "truncate",
        "create",
        "grant",
        "revoke",
        "merge",
        "call",
        "execute",
        "commit",
        "rollback",
        "copy",
    )
    forbidden_pattern = r"\b(" + "|".join(forbidden) + r")\b"
    if re.search(forbidden_pattern, lowered):
        return False, "mot-clé SQL non autorisé détecté."

    return True, ""


@tool
def execute_sql(query: str) -> str:
    """Exécute une requête SQL en lecture seule et retourne un tableau markdown.

    Args:
        query: Requête SQL SELECT ou WITH à exécuter.
    """
    ok, reason = _validate_read_only_sql(query)
    if not ok:
        return f"Erreur : requête SQL refusée ({reason})"

    try:
        executor = SQLExecutor2(connection=SQL_CONNECTION)
        df = executor.query_to_df(query)

        if df.empty:
            return "La requête n'a retourné aucun résultat."

        truncated_msg = ""
        if len(df) > MAX_ROWS:
            df = df.head(MAX_ROWS)
            truncated_msg = f"\n\n(Résultats tronqués aux {MAX_ROWS} premières lignes.)"

        return df.to_markdown(index=False) + truncated_msg

    except Exception as exc:  # Dataiku peut remonter des exceptions propres au connecteur SQL.
        return f"Erreur SQL : {exc}"


# Ajoute ici tes futurs tools : recherche documentaire, API métier, dataset Dataiku, etc.
TOOLS: list[BaseTool] = [execute_sql]
TOOLS_BY_NAME: dict[str, BaseTool] = {t.name: t for t in TOOLS}

# =============================================================================
# ÉTAT LANGGRAPH
# =============================================================================


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    visible_think: str
    tool_loops: int


# =============================================================================
# GRAPHE LANGGRAPH
# =============================================================================


def build_graph(llm: Any, llm_with_tools: Any):
    """Construit le graphe think -> act -> tools -> think."""

    def think_node(state: AgentState) -> dict[str, Any]:
        prompt = [
            SystemMessage(content=VISIBLE_THINK_PROMPT.format(schema=SCHEMA_DESCRIPTION))
        ] + state["messages"]

        response = llm.invoke(prompt)
        return {"visible_think": response.content or ""}

    def act_node(state: AgentState) -> dict[str, Any]:
        prompt = [
            SystemMessage(
                content=ACT_PROMPT.format(
                    schema=SCHEMA_DESCRIPTION,
                    visible_think=state.get("visible_think", ""),
                )
            )
        ] + state["messages"]

        response = llm_with_tools.invoke(prompt)
        return {"messages": [response]}

    def tool_node(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        results: list[ToolMessage] = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args") or {}
            tool_id = tool_call.get("id") or f"{tool_name}-unknown-id"

            tool_obj = TOOLS_BY_NAME.get(tool_name)
            if tool_obj is None:
                output = f"Erreur : outil inconnu `{tool_name}`."
            else:
                try:
                    output = tool_obj.invoke(tool_args)
                except Exception as exc:
                    output = f"Erreur pendant l'exécution de l'outil `{tool_name}` : {exc}"

            results.append(
                ToolMessage(
                    content=str(output),
                    tool_call_id=tool_id,
                    name=tool_name,
                )
            )

        return {
            "messages": results,
            "tool_loops": state.get("tool_loops", 0) + 1,
        }

    def finalize_node(state: AgentState) -> dict[str, Any]:
        prompt = [SystemMessage(content=FINALIZE_PROMPT)] + state["messages"]
        response = llm.invoke(prompt)
        return {"messages": [response]}

    def route_after_act(state: AgentState) -> str:
        last = state["messages"][-1]
        has_tool_calls = bool(getattr(last, "tool_calls", None))

        if not has_tool_calls:
            return END

        if state.get("tool_loops", 0) >= MAX_TOOL_LOOPS:
            return "finalize"

        return "tools"

    builder = StateGraph(AgentState)

    builder.add_node("think", think_node)
    builder.add_node("act", act_node)
    builder.add_node("tools", tool_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "think")
    builder.add_edge("think", "act")
    builder.add_conditional_edges(
        "act",
        route_after_act,
        {
            "tools": "tools",
            "finalize": "finalize",
            END: END,
        },
    )
    builder.add_edge("tools", "think")
    builder.add_edge("finalize", END)

    return builder.compile()


# =============================================================================
# CODE AGENT DATAIKU
# =============================================================================


class MyLLM(BaseLLM):
    def __init__(self):
        client = dataiku.api_client()
        project = client.get_default_project()

        dku_llm = project.get_llm(LLM_ID)
        self.llm = dku_llm.as_langchain_chat_model()
        self.llm_with_tools = self.llm.bind_tools(TOOLS)
        self.graph = build_graph(self.llm, self.llm_with_tools)

    @staticmethod
    def _to_langchain_messages(query: dict[str, Any]) -> list[AnyMessage]:
        """Convertit les messages Dataiku LLM Mesh vers LangChain.

        On garde volontairement seulement user/assistant pour éviter qu'un message
        externe injecte un SystemMessage non maîtrisé dans le graphe.
        """
        lc_messages: list[AnyMessage] = []

        for msg in query.get("messages", []):
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))

        return lc_messages

    def _initial_state(self, query: dict[str, Any]) -> AgentState:
        return {
            "messages": self._to_langchain_messages(query),
            "visible_think": "",
            "tool_loops": 0,
        }

    # -------------------------------------------------------------------------
    # Mode non-streaming
    # -------------------------------------------------------------------------
    def process(self, query, settings, trace):
        state = self._initial_state(query)
        final_state = self.graph.invoke(state)

        messages = final_state.get("messages", [])
        final_text = messages[-1].content if messages else ""

        if not final_text:
            final_text = (
                "Je n'ai pas pu produire de réponse finale exploitable. "
                "Vérifie les logs de l'agent, les droits SQL et la configuration LLM/tools."
            )

        return {"text": final_text}

    # -------------------------------------------------------------------------
    # Mode streaming
    # -------------------------------------------------------------------------
    def process_stream(self, query, settings, trace):
        state = self._initial_state(query)
        current_section: str | None = None

        for msg_chunk, metadata in self.graph.stream(state, stream_mode="messages"):
            node = metadata.get("langgraph_node")

            if (
                node == "think"
                and STREAM_VISIBLE_THINK
                and isinstance(msg_chunk, (AIMessage, AIMessageChunk))
            ):
                if msg_chunk.content:
                    if current_section != "think":
                        yield {"chunk": {"text": "\n\n**🧠 Plan / contrôle**\n\n"}}
                        current_section = "think"
                    yield {"chunk": {"text": msg_chunk.content}}

            elif node == "act" and isinstance(msg_chunk, (AIMessage, AIMessageChunk)):
                has_tool_signal = bool(
                    getattr(msg_chunk, "tool_calls", None)
                    or getattr(msg_chunk, "tool_call_chunks", None)
                )

                # Si le modèle est en train de construire un tool call, on ne streame pas
                # ces fragments comme une réponse finale.
                if msg_chunk.content and not has_tool_signal:
                    if current_section != "act":
                        yield {"chunk": {"text": "\n\n---\n\n**📊 Réponse**\n\n"}}
                        current_section = "act"
                    yield {"chunk": {"text": msg_chunk.content}}

            elif node == "tools" and isinstance(msg_chunk, ToolMessage):
                tool_name = getattr(msg_chunk, "name", "outil") or "outil"
                yield {"chunk": {"text": f"\n\n> 🔧 Outil exécuté : `{tool_name}`.\n"}}

                if STREAM_TOOL_OUTPUT:
                    yield {"chunk": {"text": f"\n```text\n{msg_chunk.content}\n```\n"}}

                current_section = "tools"
