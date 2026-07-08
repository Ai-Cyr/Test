# -*- coding: utf-8 -*-
"""
=====================================================================
 AGENT DATA QUALITY — Dataiku Code Agent + LangGraph
 (streaming + graphiques affichés dans Agent Hub + rapport détaillé)
=====================================================================
 Architecture (graphe LangGraph) :

     START ──> [agent] ──(tool_calls?)──> [tools] ──> [agent] ...
                  │
                  └──(fin d'investigation)──> [report] ──> END

 - [agent]  : LLM (LLM Mesh) en boucle ReAct : contrôles SQL + graphiques.
 - [tools]  : run_sql_query (SQLExecutor2, lecture seule),
              get_dataset_info, create_chart.
 - [report] : rédige le RAPPORT DÉTAILLÉ final en Markdown.

 Graphiques dans Agent Hub :
 - create_chart publie des ARTEFACTS dans la réponse de l'agent :
     * un artefact avec une part de type "RECORDS" (colonnes + lignes)
       -> Agent Hub génère nativement le graphique dans le chat
          (activer "Charts Generation" : On Demand ou Auto dans les
          settings du webapp Agent Hub) ;
     * un artefact avec une part "DATA_INLINE" (PNG matplotlib encodé
       en base64) -> image téléchargeable dans l'onglet "Downloads".
 - En streaming, les artefacts sont émis via
   yield {"chunk": {"artifacts": [...]}} (équivalent des chunks
   {"type": "content", "artifacts": [...]} du protocole LLM Mesh).

 Déploiement :
   1. Flow > +Other > Generative AI > Code Agent
      (ou Agents & GenAI Models > New Agent > Code Agent)
   2. Code env Python >= 3.10 avec : langgraph, langchain,
      langchain-core, pandas, matplotlib
   3. Renseigner LLM_ID et DATASET_NAME, sauvegarder, tester.
   4. Exposer l'agent dans Agent Hub (Enterprise Agents) et activer
      "Charts Generation" dans les settings du webapp Agent Hub.
=====================================================================
"""

import base64
import io
import json
import re
import threading
import uuid
from typing import Annotated, List, TypedDict

import dataiku
from dataiku import SQLExecutor2
from dataiku.llm.python import BaseLLM

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# =====================================================================
# 1. CONFIGURATION — à adapter
# =====================================================================
LLM_ID = "openai:VOTRE_CONNEXION:gpt-4o"  # id LLM Mesh (project.list_llms() pour la liste)
DATASET_NAME = "VOTRE_DATASET_SQL"        # dataset SQL du Flow à auditer
MAX_TOOL_ROUNDS = 15                      # nb max de tours agent->outils (anti-boucle)
MAX_ROWS_RETURNED = 50                    # lignes max renvoyées au LLM par requête
MAX_TOOL_RESULT_CHARS = 4000              # troncature des résultats d'outils
MAX_CHART_POINTS = 30                     # points max par graphique
RECURSION_LIMIT = 80                      # limite de pas du graphe LangGraph


# =====================================================================
# 2. HELPERS
# =====================================================================
def _content_to_text(content):
    """Normalise le contenu d'un message LangChain (str ou liste de blocs)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
        return "".join(parts)
    return ""


def _slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:60] or "graphique"


_FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|merge|call|exec)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------
# Registre des artefacts (graphiques) générés par les outils.
# L'outil ne renvoie au LLM qu'un petit JSON avec un chart_id (le base64
# du PNG ne doit surtout pas passer dans le contexte du LLM). Les
# artefacts sont récupérés par chart_id au moment de streamer la réponse.
# ---------------------------------------------------------------------
_ARTIFACT_REGISTRY = {}
_REGISTRY_LOCK = threading.Lock()


def _render_chart_png(chart_type, title, labels, values, series_name):
    """Rend un PNG matplotlib (base64) ; renvoie None si matplotlib absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    try:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
        if chart_type == "pie":
            ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
        elif chart_type == "line":
            ax.plot(labels, values, marker="o")
            ax.set_ylabel(series_name)
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3)
        elif chart_type == "barh":
            ax.barh(labels, values, color="#3B99FC")
            ax.invert_yaxis()
            ax.set_xlabel(series_name)
        else:  # bar
            ax.bar(labels, values, color="#3B99FC")
            ax.set_ylabel(series_name)
            ax.tick_params(axis="x", rotation=45)
        ax.set_title(title)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def _pop_artifacts_for_tool_message(tool_message):
    """Si le résultat d'outil référence un chart_id, récupère (et retire)
    les artefacts correspondants du registre."""
    try:
        payload = json.loads(_content_to_text(tool_message.content))
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    chart_id = payload.get("chart_id")
    if not chart_id:
        return []
    with _REGISTRY_LOCK:
        return _ARTIFACT_REGISTRY.pop(chart_id, [])


# =====================================================================
# 3. OUTILS
# =====================================================================
@tool
def get_dataset_info() -> str:
    """Retourne les métadonnées du dataset audité : nom de la table SQL,
    connexion, et liste des colonnes avec leur type. À appeler EN PREMIER."""
    try:
        ds = dataiku.Dataset(DATASET_NAME)
        columns = [{"name": c["name"], "type": c["type"]} for c in ds.read_schema()]
        loc = ds.get_location_info().get("info", {})
        return json.dumps(
            {
                "dataset": DATASET_NAME,
                "connection": loc.get("connectionName"),
                "sql_table": loc.get("table"),
                "sql_schema": loc.get("schema"),
                "databaseType": loc.get("databaseType"),
                "columns": columns,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def run_sql_query(query: str) -> str:
    """Exécute une requête SQL en LECTURE SEULE (SELECT / WITH uniquement)
    sur la connexion du dataset audité, via SQLExecutor2, et retourne le
    résultat en JSON (max 50 lignes). À utiliser pour tous les contrôles
    qualité : volumétrie, taux de NULL, doublons, min/max, distributions,
    formats, fraîcheur, etc. Toujours agréger, ne jamais faire SELECT *."""
    q = query.strip().rstrip(";").strip()
    if ";" in q:
        return json.dumps({"error": "Une seule instruction SQL à la fois."})
    if not q.lower().startswith(("select", "with")) or _FORBIDDEN_SQL.search(q):
        return json.dumps({"error": "Requête refusée : SELECT/WITH uniquement (lecture seule)."})
    try:
        executor = SQLExecutor2(dataset=dataiku.Dataset(DATASET_NAME))
        df = executor.query_to_df(q)
        rows = json.loads(df.head(MAX_ROWS_RETURNED).to_json(orient="records", date_format="iso"))
        return json.dumps(
            {
                "returned_rows": min(len(df), MAX_ROWS_RETURNED),
                "total_rows": len(df),
                "truncated": len(df) > MAX_ROWS_RETURNED,
                "rows": rows,
            },
            ensure_ascii=False,
            default=str,
        )
    except Exception as e:
        return json.dumps({"error": "Echec de la requête : %s" % str(e)})


@tool
def create_chart(
    title: str,
    chart_type: str,
    labels: List[str],
    values: List[float],
    series_name: str = "valeur",
) -> str:
    """Crée un graphique à partir de données AGRÉGÉES déjà obtenues via SQL
    et le publie dans la conversation (affiché dans Agent Hub).
    - title : titre explicite en français (ex : 'Taux de NULL par colonne (%)').
    - chart_type : 'bar', 'barh', 'line' ou 'pie'.
    - labels : catégories (ex : noms de colonnes), même longueur que values.
    - values : valeurs numériques associées (max ~30 points).
    - series_name : nom de la mesure (ex : 'taux de NULL (%)').
    À utiliser pour illustrer 2 à 4 constats clés de l'audit. Ne JAMAIS
    passer de données brutes ligne à ligne : uniquement des agrégats."""
    try:
        if not labels or not values or len(labels) != len(values):
            return json.dumps({"error": "labels et values doivent être non vides et de même longueur."})
        labels = [str(l) for l in labels[:MAX_CHART_POINTS]]
        values = [float(v) for v in values[:MAX_CHART_POINTS]]
        if chart_type not in ("bar", "barh", "line", "pie"):
            chart_type = "bar"

        chart_id = uuid.uuid4().hex[:12]
        artifacts = []

        # 1) Artefact RECORDS : Agent Hub sait générer nativement un
        #    graphique dans le chat à partir de ce format (si l'option
        #    "Charts Generation" est activée dans les settings du webapp).
        artifacts.append(
            {
                "id": "records-" + chart_id,
                "name": title,
                "description": "Données du graphique « %s » (audit data quality de %s)"
                % (title, DATASET_NAME),
                "parts": [
                    {
                        "type": "RECORDS",
                        "records": {
                            "columns": ["libelle", series_name],
                            "data": [[l, v] for l, v in zip(labels, values)],
                        },
                    }
                ],
            }
        )

        # 2) Artefact DATA_INLINE : image PNG du graphique, disponible
        #    dans l'onglet "Downloads" d'Agent Hub.
        png_b64 = _render_chart_png(chart_type, title, labels, values, series_name)
        if png_b64:
            artifacts.append(
                {
                    "id": "image-" + chart_id,
                    "name": _slug(title) + ".png",
                    "description": title,
                    "parts": [
                        {
                            "type": "DATA_INLINE",
                            "mimeType": "image/png",
                            "dataBase64": png_b64,
                        }
                    ],
                }
            )

        with _REGISTRY_LOCK:
            _ARTIFACT_REGISTRY[chart_id] = artifacts

        return json.dumps(
            {
                "chart_id": chart_id,
                "status": "created",
                "title": title,
                "chart_type": chart_type,
                "points": len(labels),
                "png_generated": bool(png_b64),
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"error": "Echec de création du graphique : %s" % str(e)})


TOOLS = [get_dataset_info, run_sql_query, create_chart]


# =====================================================================
# 4. PROMPTS
# =====================================================================
SYSTEM_PROMPT = """Tu es un expert Data Quality chargé d'auditer un dataset SQL dans Dataiku.

OUTILS :
- get_dataset_info : métadonnées (table SQL, connexion, colonnes + types). À appeler en premier.
- run_sql_query : exécute une requête SELECT (lecture seule) via SQLExecutor2.
- create_chart : publie un graphique (bar, barh, line, pie) dans la conversation
  à partir de données agrégées déjà obtenues.

PLAN D'AUDIT À DÉROULER :
1. get_dataset_info pour connaître la table et les colonnes.
2. Volumétrie : COUNT(*) total.
3. Complétude : taux de NULL de TOUTES les colonnes en UNE seule requête
   (SUM(CASE WHEN "col" IS NULL THEN 1 ELSE 0 END) AS nulls_col, ...).
4. Unicité : lignes entièrement dupliquées + doublons sur les colonnes
   clés probables (COUNT(*) vs COUNT(DISTINCT ...)).
5. Validité par type :
   - numériques : MIN/MAX/AVG, valeurs négatives ou hors plage suspectes ;
   - textes : chaînes vides '', espaces parasites (TRIM), longueurs anormales,
     top valeurs via GROUP BY ... ORDER BY COUNT(*) DESC LIMIT 10 ;
   - dates : MIN/MAX, dates futures ou incohérentes.
6. Fraîcheur : si colonne de date, récence de la donnée la plus récente.
7. Cohérence : 1 ou 2 contrôles croisés pertinents selon le métier des colonnes.
8. VISUALISATION : génère 2 à 4 graphiques des constats majeurs avec create_chart,
   en réutilisant les chiffres déjà obtenus (aucune nouvelle requête SQL pour cela).
   Exemples : taux de NULL par colonne en 'barh', top 10 des valeurs d'une colonne
   suspecte en 'bar', répartition lignes uniques vs doublons en 'pie', évolution
   du volume par mois en 'line'.

RÈGLES :
- SQL standard adapté au dialecte de la connexion ; identifiants entre guillemets doubles ("colonne").
- REGROUPE les contrôles pour minimiser le nombre de requêtes.
- Toujours AGRÉGER : ne rapatrie jamais de gros volumes de lignes brutes.
- Si une requête échoue, corrige la syntaxe et réessaie UNE seule fois.
- Ne modifie JAMAIS les données (SELECT uniquement).
- Annonce brièvement (1 phrase) chaque contrôle ou graphique avant de l'exécuter.
- Quand l'audit et les graphiques sont terminés, réponds SANS appeler d'outil
  par une synthèse de 2-3 phrases. Le rapport détaillé sera rédigé juste après.
- Réponds toujours en français."""

REPORT_PROMPT = """Tu es un expert Data Quality. À partir de la trace d'audit fournie
(appels d'outils et résultats SQL réels), rédige un RAPPORT DÉTAILLÉ en Markdown,
en français, avec EXACTEMENT cette structure :

# 📋 Rapport d'audit Data Quality — {dataset}

## 1. Résumé exécutif
- Score global de qualité sur 100 (justifie le calcul) et verdict en une phrase.
- Les 3 à 5 constats majeurs.

## 2. Périmètre et méthodologie
- Table auditée, connexion, volumétrie, liste des contrôles exécutés.

## 3. Résultats détaillés par dimension
### 3.1 Complétude
Tableau : colonne | nb de NULL | taux (%) | statut (✅ / ⚠️ / 🔴)
### 3.2 Unicité
Doublons détectés (chiffres exacts, clés concernées).
### 3.3 Validité
Anomalies de format / plage / valeurs suspectes, par colonne.
### 3.4 Cohérence
Résultats des contrôles croisés.
### 3.5 Fraîcheur
Récence des données (si applicable).

## 4. Anomalies détectées
Tableau : # | colonne(s) | anomalie | sévérité (🔴 critique / 🟠 majeure / 🟡 mineure) | impact probable

## 5. Recommandations priorisées
Actions correctives concrètes, et pour les plus importantes un exemple de
requête SQL de remédiation ou de contrôle continu.

RÈGLES STRICTES :
- Utilise UNIQUEMENT les chiffres réellement présents dans la trace (aucune invention).
- Lorsqu'un graphique a été généré (appels create_chart), fais-y référence dans la
  section concernée (ex : « voir le graphique 'Taux de NULL par colonne' ci-dessus »).
- Si un contrôle n'a pas pu être réalisé ou a échoué, signale-le explicitement.
- Sois précis, chiffré et actionnable."""


def _build_transcript(messages):
    """Sérialise l'historique (appels d'outils + résultats) pour le nœud rapport.
    Plus robuste que de repasser les messages 'tool' bruts à un LLM sans outils."""
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append("[DEMANDE UTILISATEUR] " + _content_to_text(m.content))
        elif isinstance(m, AIMessage):
            txt = _content_to_text(m.content)
            if txt:
                lines.append("[AGENT] " + txt)
            for tc in (getattr(m, "tool_calls", None) or []):
                args = tc.get("args", {})
                # Ne pas noyer le transcript avec les listes de points des graphiques
                if tc.get("name") == "create_chart" and isinstance(args, dict):
                    args = {k: v for k, v in args.items() if k in ("title", "chart_type", "series_name")}
                lines.append(
                    "[APPEL OUTIL] %s(%s)"
                    % (tc.get("name"), json.dumps(args, ensure_ascii=False))
                )
        elif isinstance(m, ToolMessage):
            txt = _content_to_text(m.content)[:MAX_TOOL_RESULT_CHARS]
            lines.append("[RÉSULTAT %s] %s" % (getattr(m, "name", "outil"), txt))
    return "\n".join(lines)


# =====================================================================
# 5. CONSTRUCTION DU GRAPHE LANGGRAPH
# =====================================================================
class DQState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_rounds: int


def build_graph(settings=None):
    """Construit et compile le graphe. `settings` = CompletionSettings Dataiku
    (température, etc.) transmis au modèle LangChain du LLM Mesh."""
    llm_handle = dataiku.api_client().get_default_project().get_llm(LLM_ID)
    chat = llm_handle.as_langchain_chat_model(completion_settings=settings)
    chat_with_tools = chat.bind_tools(TOOLS)

    def agent_node(state: DQState):
        msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = chat_with_tools.invoke(msgs)
        rounds = state.get("tool_rounds", 0)
        if getattr(response, "tool_calls", None):
            rounds += 1
        return {"messages": [response], "tool_rounds": rounds}

    def route_after_agent(state: DQState):
        last = state["messages"][-1]
        has_tool_calls = bool(getattr(last, "tool_calls", None))
        if has_tool_calls and state.get("tool_rounds", 0) <= MAX_TOOL_ROUNDS:
            return "tools"
        return "report"  # investigation terminée -> rapport détaillé

    def report_node(state: DQState):
        transcript = _build_transcript(state["messages"])
        msgs = [
            SystemMessage(content=REPORT_PROMPT.replace("{dataset}", DATASET_NAME)),
            HumanMessage(
                content="Voici la trace complète de l'audit :\n\n"
                + transcript
                + "\n\nRédige maintenant le rapport détaillé."
            ),
        ]
        response = chat.invoke(msgs)  # sans outils : rédaction pure
        return {"messages": [response]}

    graph = StateGraph(DQState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("report", report_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "report": "report"})
    graph.add_edge("tools", "agent")
    graph.add_edge("report", END)
    return graph.compile()


# =====================================================================
# 6. CODE AGENT DATAIKU (BaseLLM)
# =====================================================================
class MyLLM(BaseLLM):
    def __init__(self):
        pass

    # ---------- conversion query Dataiku -> état LangGraph ----------
    def _initial_state(self, query):
        lc_messages = []
        for m in query.get("messages", []):
            role = m.get("role")
            content = _content_to_text(m.get("content", "")) or str(m.get("content", ""))
            if not content:
                continue
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
        if not lc_messages:
            lc_messages = [
                HumanMessage(content="Réalise un audit qualité complet du dataset %s." % DATASET_NAME)
            ]
        return {"messages": lc_messages, "tool_rounds": 0}

    # ---------- mode synchrone (fallback sans streaming) ----------
    def process(self, query, settings, trace):
        app = build_graph(settings)
        final_state = app.invoke(
            self._initial_state(query),
            config={"recursion_limit": RECURSION_LIMIT},
        )
        # Récupération des graphiques générés pendant l'audit
        artifacts = []
        for m in final_state["messages"]:
            if isinstance(m, ToolMessage):
                artifacts.extend(_pop_artifacts_for_tool_message(m))
        response = {"text": _content_to_text(final_state["messages"][-1].content)}
        if artifacts:
            response["artifacts"] = artifacts
        return response

    # ---------- mode STREAMING ----------
    def process_stream(self, query, settings, trace):
        app = build_graph(settings)
        state = self._initial_state(query)

        yield {"chunk": {"text": "## 🔎 Audit Data Quality — `%s`\n\n" % DATASET_NAME}}

        report_started = False
        # stream_mode combiné :
        #  - "messages" : tokens des LLM des nœuds (agent + report), au fil de l'eau
        #  - "updates"  : fins de nœuds -> notifications d'outils + artefacts (graphiques)
        for mode, payload in app.stream(
            state,
            config={"recursion_limit": RECURSION_LIMIT},
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                msg_chunk, metadata = payload
                node = metadata.get("langgraph_node")
                if node not in ("agent", "report"):
                    continue
                if node == "report" and not report_started:
                    report_started = True
                    yield {"chunk": {"text": "\n\n---\n\n"}}
                text = _content_to_text(getattr(msg_chunk, "content", ""))
                if text:
                    yield {"chunk": {"text": text}}

            elif mode == "updates":
                for node, update in (payload or {}).items():
                    if node != "tools" or not isinstance(update, dict):
                        continue
                    for m in update.get("messages", []):
                        name = getattr(m, "name", "outil")
                        artifacts = _pop_artifacts_for_tool_message(m)
                        if artifacts:
                            # Graphique : émettre les artefacts dans le flux
                            yield {"chunk": {"text": "\n\n> 📊 Graphique publié : %s\n\n"
                                                     % artifacts[0].get("name", "")}}
                            yield {"chunk": {"artifacts": artifacts}}
                        else:
                            yield {"chunk": {"text": "\n\n> 🔧 Contrôle exécuté : `%s`\n\n" % name}}

        yield {"footer": {}}
