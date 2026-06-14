# -*- coding: utf-8 -*-
"""
Template Agent Dataiku + LangGraph (version amelioree)
======================================================

Pattern conserve :

    START -> think -> act -> tools -> think -> act -> ... -> END

Objectif : un squelette reutilisable pour des agents Dataiku Code Agent
avec Mistral Small 3.2 (mistral-small-2506) ou tout autre modele du LLM Mesh.

Ameliorations par rapport a la v1 (cherchez "AMELIORATION" dans le fichier) :
  1. [BUG] Fermeture des tool_calls en attente avant `finalize` : un message
     assistant avec tool_calls DOIT etre suivi de tool messages, sinon les
     modeles Mistral renvoient une erreur 400.
  2. Conformite format Mistral : un message assistant ne peut pas avoir a la
     fois `content` et `tool_calls` -> on vide le content quand il y a un
     tool call. Les id de tool call doivent etre alphanumeriques sur 9 cars.
  3. Temperature basse (~0.15) recommandee par Mistral pour du SQL reproductible.
  4. LIMIT pousse dans le SQL (au lieu de charger toute la table puis .head()).
  5. Validation SQL robuste : on retire commentaires ET litteraux de chaine
     avant les controles, pour eviter les faux positifs.

A adapter dans chaque projet :
- LLM_ID
- SQL_CONNECTION
- SCHEMA_DESCRIPTION
- TOOLS
- Prompts metier
"""

from __future__ import annotations

import re
import uuid
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

# Mistral Small 3.2 = release 2506. Le nom exact depend de la connexion LLM Mesh
# declaree chez vous ; l'admin peut exposer un alias different.
LLM_ID = "mistral:votre-connexion:mistral-small-2506"

SQL_CONNECTION = "votre_connexion_sql"
MAX_ROWS = 50
MAX_TOOL_LOOPS = 4

# AMELIORATION 3 : Mistral recommande une temperature basse (~0.15). Pour un
# agent SQL, cela rend le SQL genere plus deterministe et reproductible.
TEMPERATURE = 0.15

# Afficher la note du noeud think dans le streaming.
# Conseil : garder True en dev / audit, eventuellement False en production.
STREAM_VISIBLE_THINK = True

# Afficher le resultat brut des tools dans le chat.
# Conseil : garder False si les resultats peuvent contenir des donnees sensibles.
STREAM_TOOL_OUTPUT = False

SCHEMA_DESCRIPTION = """
Tables disponibles :
- customers(customer_id, name, country, created_at)
- orders(order_id, customer_id, amount, order_date, status)

Regles metier utiles :
- `orders.amount` est exprime en euros.
- `orders.status = 'paid'` signifie une commande payee.
"""

VISIBLE_THINK_PROMPT = """Tu es le module de cadrage et de controle d'un agent data analyste.

Important : produis une note visible et concise pour l'utilisateur, pas un raisonnement interne detaille.

{schema}

A chaque passage dans ce noeud :
1. Resume l'objectif utilisateur en une phrase.
2. Indique les donnees ou resultats deja disponibles.
3. Indique la prochaine action logique : requete SQL, correction SQL, ou reponse finale.
4. Apres un resultat de tool, verifie si le resultat suffit ou s'il faut une nouvelle action.

Format attendu :
- 3 a 6 lignes maximum.
- Francais.
- Ne donne pas encore la reponse finale metier.
"""

ACT_PROMPT = """Tu es un assistant data analyste connecte a une base SQL.

{schema}

Note de cadrage/controle produite juste avant :
---
{visible_think}
---

Regles :
- Si une donnee manque, appelle l'outil `execute_sql`.
- Utilise uniquement des requetes SELECT ou WITH.
- Si les resultats disponibles suffisent, reponds clairement en francais.
- Ne fabrique jamais de donnees.
- Si un outil retourne une erreur, corrige la requete ou explique la limite.
- Pour une reponse finale, donne les chiffres cles, un tableau markdown si utile, puis une courte interpretation.
"""

FINALIZE_PROMPT = """Tu es un assistant data analyste.

La limite de boucles d'outils a ete atteinte ou l'agent ne peut plus avancer de facon fiable.
Reponds a l'utilisateur avec ce qui est disponible, sans inventer de donnees.
Explique brievement la limite rencontree et propose une prochaine action concrete.
"""

# =============================================================================
# OUTILS
# =============================================================================


def _strip_sql_comments(query: str) -> str:
    """Supprime les commentaires SQL simples (-- ... et /* ... */)."""
    query = re.sub(r"--.*?$", "", query, flags=re.MULTILINE)
    query = re.sub(r"/\*.*?\*/", "", query, flags=re.DOTALL)
    return query.strip()


def _strip_string_literals(query: str) -> str:
    """Remplace le contenu des litteraux de chaine par une chaine vide.

    AMELIORATION 5 : evite les faux positifs lors des controles de securite.
    Sans cela, `WHERE note = 'pense a update demain'` declencherait la
    blocklist, et `SELECT 'a;b'` serait vu comme du multi-statements.
    Gere les apostrophes doublees SQL ('' a l'interieur d'une chaine).
    """
    query = re.sub(r"'(?:[^']|'')*'", "''", query)
    query = re.sub(r'"(?:[^"]|"")*"', '""', query)
    return query


def _validate_read_only_sql(query: str) -> tuple[bool, str]:
    """Validation defensive : lecture seule, un seul statement.

    Les controles tournent sur une version SANS commentaires ni litteraux ;
    la requete reellement executee reste la requete d'origine.
    """
    checkable = _strip_string_literals(_strip_sql_comments(query))
    if not checkable:
        return False, "requete vide."

    # Autorise un unique point-virgule final, refuse les multi-statements.
    without_final_semicolon = checkable[:-1].strip() if checkable.endswith(";") else checkable
    if ";" in without_final_semicolon:
        return False, "multi-statements refuses."

    lowered = without_final_semicolon.lower()
    if not re.match(r"^(select|with)\b", lowered):
        return False, "seules les requetes SELECT ou CTE WITH sont autorisees."

    forbidden = (
        "insert", "update", "delete", "drop", "alter", "truncate",
        "create", "grant", "revoke", "merge", "call", "execute",
        "commit", "rollback", "copy",
    )
    forbidden_pattern = r"\b(" + "|".join(forbidden) + r")\b"
    if re.search(forbidden_pattern, lowered):
        return False, "mot-cle SQL non autorise detecte."

    return True, ""


@tool
def execute_sql(query: str) -> str:
    """Execute une requete SQL en lecture seule et retourne un tableau markdown.

    Args:
        query: Requete SQL SELECT ou WITH a executer.
    """
    ok, reason = _validate_read_only_sql(query)
    if not ok:
        return f"Erreur : requete SQL refusee ({reason})"

    # AMELIORATION 4 : on pousse le LIMIT dans le SQL pour ne PAS rapatrier
    # toute la table en memoire avant de tronquer. On demande MAX_ROWS + 1
    # lignes pour detecter une troncature. (Compatible Postgres, Snowflake,
    # BigQuery, Redshift... Si votre moteur n'autorise pas LIMIT sur une
    # sous-requete, repliez-vous sur un LIMIT ajoute directement a la requete.)
    inner = query.strip().rstrip(";").strip()
    wrapped = f"SELECT * FROM (\n{inner}\n) AS _agent_sub LIMIT {MAX_ROWS + 1}"

    try:
        executor = SQLExecutor2(connection=SQL_CONNECTION)
        df = executor.query_to_df(wrapped)

        if df.empty:
            return "La requete n'a retourne aucun resultat."

        truncated_msg = ""
        if len(df) > MAX_ROWS:
            df = df.head(MAX_ROWS)
            truncated_msg = f"\n\n(Resultats tronques aux {MAX_ROWS} premieres lignes.)"

        return df.to_markdown(index=False) + truncated_msg

    except Exception as exc:  # Dataiku peut remonter des exceptions du connecteur SQL.
        return f"Erreur SQL : {exc}"


# Ajoutez ici vos futurs tools : recherche documentaire, API metier, dataset Dataiku, etc.
TOOLS: list[BaseTool] = [execute_sql]
TOOLS_BY_NAME: dict[str, BaseTool] = {t.name: t for t in TOOLS}


# =============================================================================
# HELPERS DE CONFORMITE FORMAT MISTRAL
# =============================================================================


def _valid_tool_call_id() -> str:
    """Genere un id de tool call conforme Mistral : alphanumerique, 9 caracteres.

    AMELIORATION 2 : un id avec tirets ou de longueur != 9 provoque une 400.
    """
    return uuid.uuid4().hex[:9]


def _normalize_assistant_message(message: AIMessage) -> AIMessage:
    """Rend un message assistant compatible Mistral.

    AMELIORATION 2 : Mistral refuse un message assistant qui a A LA FOIS du
    `content` et des `tool_calls`. Quand il y a des tool_calls, on vide le
    content. On s'assure aussi que chaque tool_call a un id conforme.
    """
    tool_calls = getattr(message, "tool_calls", None) or []
    if not tool_calls:
        return message

    fixed_calls = []
    for call in tool_calls:
        call = dict(call)
        cid = call.get("id") or ""
        if not (len(cid) == 9 and cid.isalnum()):
            call["id"] = _valid_tool_call_id()
        fixed_calls.append(call)

    return AIMessage(content="", tool_calls=fixed_calls, id=getattr(message, "id", None))


def _close_pending_tool_calls(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Ferme les tool_calls non executes du dernier message assistant.

    AMELIORATION 1 (le bug principal) : avant d'appeler le LLM dans `finalize`,
    si le dernier message est un assistant avec des tool_calls non suivis de
    resultats, on ajoute un ToolMessage synthetique par tool_call. Sans ca,
    Mistral renvoie : "assistant message with tool_calls must be followed by
    tool messages responding to each tool_call_id" (400).
    """
    if not messages:
        return messages

    last = messages[-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    if not isinstance(last, AIMessage) or not tool_calls:
        return messages

    stubs = [
        ToolMessage(
            content="Outil non execute : limite de boucles atteinte.",
            tool_call_id=call.get("id") or _valid_tool_call_id(),
            name=call.get("name", "outil"),
        )
        for call in tool_calls
    ]
    return list(messages) + stubs


# =============================================================================
# ETAT LANGGRAPH
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
        # AMELIORATION 2 : on normalise avant de stocker, pour que les passages
        # ulterieurs dans think/finalize rejouent un historique valide pour Mistral.
        return {"messages": [_normalize_assistant_message(response)]}

    def tool_node(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        results: list[ToolMessage] = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args") or {}
            # AMELIORATION 2 : id de repli conforme Mistral (9 cars alphanum).
            tool_id = tool_call.get("id") or _valid_tool_call_id()

            tool_obj = TOOLS_BY_NAME.get(tool_name)
            if tool_obj is None:
                output = f"Erreur : outil inconnu `{tool_name}`."
            else:
                try:
                    output = tool_obj.invoke(tool_args)
                except Exception as exc:
                    output = f"Erreur pendant l'execution de l'outil `{tool_name}` : {exc}"

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
        # AMELIORATION 1 : on ferme les tool_calls en attente avant l'appel LLM.
        safe_messages = _close_pending_tool_calls(state["messages"])
        prompt = [SystemMessage(content=FINALIZE_PROMPT)] + safe_messages
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
        base_llm = dku_llm.as_langchain_chat_model()

        # AMELIORATION 3 : on applique la temperature via bind().
        # IMPORTANT : bind_tools AVANT bind(), car bind() renvoie un
        # RunnableBinding qui n'expose pas forcement bind_tools.
        # (Selon la version du plugin Dataiku, vous pouvez aussi fixer la
        #  temperature dans les reglages de la connexion LLM Mesh.)
        self.llm = base_llm.bind(temperature=TEMPERATURE)
        self.llm_with_tools = base_llm.bind_tools(TOOLS).bind(temperature=TEMPERATURE)

        self.graph = build_graph(self.llm, self.llm_with_tools)

    @staticmethod
    def _to_langchain_messages(query: dict[str, Any]) -> list[AnyMessage]:
        """Convertit les messages Dataiku LLM Mesh vers LangChain.

        On garde volontairement seulement user/assistant pour eviter qu'un
        message externe injecte un SystemMessage non maitrise dans le graphe.
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

        # AMELIORATION (robustesse) : une erreur du graphe ne doit pas remonter
        # une exception brute a l'utilisateur final.
        try:
            final_state = self.graph.invoke(state)
        except Exception as exc:
            return {
                "text": (
                    "Une erreur est survenue pendant le traitement de l'agent. "
                    f"Detail technique : {exc}"
                )
            }

        messages = final_state.get("messages", [])
        final_text = messages[-1].content if messages else ""

        if not final_text:
            final_text = (
                "Je n'ai pas pu produire de reponse finale exploitable. "
                "Verifie les logs de l'agent, les droits SQL et la configuration LLM/tools."
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
                        yield {"chunk": {"text": "\n\n**Plan / controle**\n\n"}}
                        current_section = "think"
                    yield {"chunk": {"text": msg_chunk.content}}

            elif node in ("act", "finalize") and isinstance(msg_chunk, (AIMessage, AIMessageChunk)):
                has_tool_signal = bool(
                    getattr(msg_chunk, "tool_calls", None)
                    or getattr(msg_chunk, "tool_call_chunks", None)
                )

                # Si le modele construit un tool call, on ne streame pas ces
                # fragments comme une reponse finale.
                if msg_chunk.content and not has_tool_signal:
                    if current_section != "answer":
                        yield {"chunk": {"text": "\n\n---\n\n**Reponse**\n\n"}}
                        current_section = "answer"
                    yield {"chunk": {"text": msg_chunk.content}}

            elif node == "tools" and isinstance(msg_chunk, ToolMessage):
                tool_name = getattr(msg_chunk, "name", "outil") or "outil"
                yield {"chunk": {"text": f"\n\n> Outil execute : `{tool_name}`.\n"}}

                if STREAM_TOOL_OUTPUT:
                    yield {"chunk": {"text": f"\n```text\n{msg_chunk.content}\n```\n"}}

                current_section = "tools"
