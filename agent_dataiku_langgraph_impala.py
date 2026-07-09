# -*- coding: utf-8 -*-
"""
Agent Code Dataiku — Data Steward conversationnel avec LangGraph
================================================================

Ce fichier implémente un Code Agent Dataiku DSS 14.x basé sur BaseLLM.

Fonctionnalités
---------------
- Conversation multi-tours : l'historique du chat est rejoué au LLM.
- Orchestration LangGraph :
    START -> init_completion -> llm_step -> tool_node -> llm_step -> END
- Tool calling via LLM Mesh Dataiku.
- Outils SQL en lecture seule :
    * list_tables
    * run_sql
    * profile_table
    * export_markdown
- Export Markdown compatible Agent Hub Downloads via artefact DATA_INLINE.
- Streaming des étapes : appels outils, requêtes SQL, résultats, réponse finale.

Prérequis
---------
- Dataiku DSS avec Code Agents.
- Code env Python >= 3.10.
- Package Python `langgraph` installé dans le code env.
- Un LLM_ID compatible tool calling.

Installation rapide
-------------------
1. Agents > New agent > Code agent.
2. Sélectionner un code env Python >= 3.10 contenant `langgraph`.
3. Coller ce fichier.
4. Configurer SQL_CONNECTION et, si besoin, LLM_ID / EXPORT_FOLDER_ID.

Exemples
--------
- "Bonjour, tu sais faire quoi ?"
- "Quelles tables sont disponibles ?"
- "Fais le glossaire métier de la table default.customers"
- "Combien y a-t-il de lignes dans cette table ?"
- "Quelles sont les 5 valeurs les plus fréquentes de la colonne country ?"
- "Exporte ce glossaire en Markdown"
"""

import base64
import json
import re
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple, TypedDict

import dataiku
from dataiku import SQLExecutor2
from dataiku.llm.python import BaseLLM

try:
    from langgraph.graph import END, START, StateGraph
except Exception as e:
    raise ImportError(
        "Le package `langgraph` est requis. "
        "Ajoutez-le au code env Python du Code Agent Dataiku, par exemple : "
        "`pip install -U langgraph`."
    ) from e


# =============================================================================
# Configuration à adapter
# =============================================================================

SQL_CONNECTION = "REMPLACER_PAR_VOTRE_CONNEXION"  # nom de la connexion SQL Dataiku

# Exemple : "openai:ma-connexion:gpt-4o"
# Laisser vide pour prendre le premier LLM disponible dans le projet.
# Important : le LLM choisi doit supporter le tool calling.
LLM_ID = ""

# ID ou nom d'un managed folder pour copier les exports Markdown.
# Laisser vide pour désactiver l'export managed folder.
EXPORT_FOLDER_ID = ""

# Impala : les colonnes contenant des caractères spéciaux se quotent avec des backticks.
QUOTE_CHAR = '`'

# Impala : les tables doivent être référencées en base.table.
# DEFAULT_DATABASE peut être renseigné si vous voulez accepter un nom de table seul
# et le transformer automatiquement en DEFAULT_DATABASE.table.
DEFAULT_DATABASE = ""
REQUIRE_DATABASE_IN_TABLE_NAME = True

SAMPLE_QUERY = "SELECT * FROM {table} LIMIT {n}"

SAMPLE_ROWS = 100
MAX_EXAMPLES = 5
MAX_CELL_LEN = 60
WITH_DISTINCT = True

MAX_RESULT_ROWS = 50        # nombre max de lignes renvoyées au LLM par run_sql
MAX_TABLES_FOR_LLM = 200    # nombre max de tables renvoyées au LLM
MAX_SQL_DISPLAY = 700       # troncature dans le chat
MAX_HISTORY_MESSAGES = 16   # fenêtre conversationnelle rejouée au LLM
MAX_STEPS = 8               # nombre max d'itérations LLM <-> outils par tour

LLM_TEMPERATURE = 0.2
LLM_MAX_OUTPUT_TOKENS = 4096


# =============================================================================
# Prompt système
# =============================================================================

SYSTEM_PROMPT = """Tu es « Data Steward », un assistant conversationnel Dataiku connecté
en LECTURE SEULE à une base de données à Impala via la connexion Dataiku « %s ».

Tu aides l'utilisateur à :
- comprendre le contenu de tables SQL ;
- répondre à des questions chiffrées en exécutant du SQL ;
- profiler une table ;
- produire un glossaire métier Markdown ;
- exporter un glossaire ou un document Markdown.

Tu disposes de ces outils :
- list_tables : lister les tables disponibles.
- run_sql : exécuter une requête SELECT/WITH en lecture seule.
- profile_table : profiler une table : colonnes, types, exemples, %% nulls,
  cardinalités, nombre de lignes. Obligatoire avant un glossaire.
- export_markdown : publier un document Markdown comme fichier téléchargeable.

Règles importantes :
- Réponds en français.
- Sois conversationnel, clair et concis.
- Ne donne JAMAIS un chiffre inventé : utilise run_sql pour le calculer.
- N'exécute que des requêtes de lecture : SELECT ou WITH.
- Les tables Impala doivent être nommées avec leur base : `base.table`.
- Si l'utilisateur donne seulement `table`, demande la base Impala ou utilise `DEFAULT_DATABASE` si elle est configurée.
- Si l'utilisateur dit « cette table », « la même table » ou « celle-ci »,
  utilise la dernière table `base.table` connue dans le contexte conversationnel.
- Pour « fais le glossaire métier de la table base.table » :
  1. appelle profile_table ;
  2. rédige le glossaire complet dans ta réponse ;
  3. appelle export_markdown avec le document complet et un filename finissant en .md.
- Format du glossaire :
  # Glossaire métier — `<table>`
  Une à deux phrases de description métier.
  Puis un tableau Markdown :
  | Colonne | Libellé métier | Définition | Type | Exemples | %% nulls | Distincts | Remarques |
- Reste factuel : si le sens métier est incertain, écris-le dans Remarques.
- Signale dans Remarques :
  données personnelles ou sensibles possibles, clés probables,
  colonnes très vides, valeurs ambiguës.
""" % SQL_CONNECTION


# =============================================================================
# État LangGraph
# =============================================================================

class AgentState(TypedDict, total=False):
    query: Dict[str, Any]
    settings: Dict[str, Any]
    trace: Any

    completion: Any
    llm_calls: int

    tool_calls: List[Dict[str, Any]]
    final_text: str
    display_text: str
    display_chunks: List[Dict[str, Any]]
    error_text: str


RouteAfterLLM = Literal["tools", "end"]


# =============================================================================
# Agent Dataiku
# =============================================================================

class MyLLM(BaseLLM):
    """Code Agent Dataiku orchestré avec LangGraph."""

    def __init__(self):
        self.client = dataiku.api_client()
        self.project = self.client.get_default_project()
        self.executor = SQLExecutor2(connection=SQL_CONNECTION)
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ #
    # Mode non-streamé : rejoue le stream et agrège texte + artefacts.
    # ------------------------------------------------------------------ #
    def process(self, query, settings, trace):
        text_parts: List[str] = []
        artifacts: List[Dict[str, Any]] = []

        for event in self.process_stream(query, settings, trace):
            if not isinstance(event, dict):
                continue
            chunk = event.get("chunk", {})
            if chunk.get("text"):
                text_parts.append(chunk["text"])
            if chunk.get("artifacts"):
                artifacts.extend(chunk["artifacts"])

        response: Dict[str, Any] = {"text": "".join(text_parts)}
        if artifacts:
            response["artifacts"] = artifacts
        return response

    # ------------------------------------------------------------------ #
    # Mode streamé : exécute le graphe LangGraph et publie les updates.
    # ------------------------------------------------------------------ #
    def process_stream(self, query, settings, trace):
        state: AgentState = {
            "query": query or {},
            "settings": settings or {},
            "trace": trace,
            "llm_calls": 0,
        }

        try:
            for update in self.graph.stream(state, stream_mode="updates"):
                for node_name, delta in update.items():
                    if not delta:
                        continue

                    if delta.get("error_text"):
                        yield self._txt(delta["error_text"])

                    if delta.get("display_text"):
                        yield self._txt(delta["display_text"])

                    for chunk in delta.get("display_chunks", []) or []:
                        yield chunk

                    if delta.get("final_text"):
                        yield self._txt(delta["final_text"])

        except Exception as e:
            yield self._txt(
                "❌ Erreur pendant l'exécution LangGraph :\n"
                "```\n%s\n```" % str(e)
            )

    # ------------------------------------------------------------------ #
    # Construction du graphe LangGraph
    # ------------------------------------------------------------------ #
    def _build_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node("init_completion", self._node_init_completion)
        graph.add_node("llm_step", self._node_llm_step)
        graph.add_node("tool_node", self._node_tool_node)

        graph.add_edge(START, "init_completion")
        graph.add_edge("init_completion", "llm_step")

        graph.add_conditional_edges(
            "llm_step",
            self._route_after_llm,
            {
                "tools": "tool_node",
                "end": END,
            },
        )

        graph.add_edge("tool_node", "llm_step")

        return graph.compile()

    # ------------------------------------------------------------------ #
    # Nœud 1 : initialisation de la complétion Dataiku LLM Mesh
    # ------------------------------------------------------------------ #
    def _node_init_completion(self, state: AgentState) -> Dict[str, Any]:
        query = state.get("query", {})
        messages = query.get("messages", []) or []

        llm = self.project.get_llm(self._resolve_llm_id())
        completion = llm.new_completion()

        # Paramètres standards.
        try:
            completion.settings["temperature"] = LLM_TEMPERATURE
            completion.settings["maxOutputTokens"] = LLM_MAX_OUTPUT_TOKENS
            completion.settings["tools"] = self._tool_specs()
        except Exception:
            # Certaines versions/connecteurs peuvent exposer les settings différemment.
            # Si cette affectation échoue, l'appel LLM renverra une erreur exploitable.
            pass

        completion.with_message(SYSTEM_PROMPT, role="system")

        context = self._conversation_context(messages)
        if context:
            completion.with_message(context, role="system")

        # Rejoue une fenêtre d'historique afin de rester conversationnel
        # sans faire exploser le prompt.
        for msg in messages[-MAX_HISTORY_MESSAGES:]:
            role = msg.get("role", "user")
            if role not in ("user", "assistant"):
                continue
            text = self._msg_text(msg)
            if text.strip():
                completion.with_message(text, role=role)

        return {"completion": completion, "llm_calls": 0}

    # ------------------------------------------------------------------ #
    # Nœud 2 : appel LLM. Le LLM répond ou demande des outils.
    # ------------------------------------------------------------------ #
    def _node_llm_step(self, state: AgentState) -> Dict[str, Any]:
        completion = state.get("completion")
        trace = state.get("trace")
        llm_calls = int(state.get("llm_calls", 0)) + 1

        if completion is None:
            return {
                "final_text": (
                    "❌ La complétion LLM n'a pas été initialisée. "
                    "Vérifiez la configuration de l'agent."
                ),
                "tool_calls": [],
                "llm_calls": llm_calls,
            }

        if llm_calls > MAX_STEPS:
            return {
                "final_text": (
                    "\n⚠️ Nombre maximal d'étapes atteint (%d). "
                    "Reformulez ou découpez votre demande."
                    % MAX_STEPS
                ),
                "tool_calls": [],
                "llm_calls": llm_calls,
            }

        with self._subspan(trace, "LangGraph — LLM étape %d" % llm_calls) as span:
            resp = completion.execute()
            try:
                if span is not None:
                    span.append_trace(resp.trace)
            except Exception:
                pass

        if not getattr(resp, "success", False):
            return {
                "final_text": (
                    "❌ Échec de l'appel au LLM. "
                    "Vérifiez `LLM_ID` et assurez-vous que le modèle supporte "
                    "le tool calling."
                ),
                "tool_calls": [],
                "llm_calls": llm_calls,
            }

        tool_calls = self._get_tool_calls(resp)

        if not tool_calls:
            return {
                "final_text": getattr(resp, "text", "") or "",
                "tool_calls": [],
                "llm_calls": llm_calls,
            }

        # Important : rattache les tool calls à la complétion avant les tool outputs.
        completion.with_tool_calls(tool_calls)

        display_text = ""
        if getattr(resp, "text", None):
            display_text = resp.text + "\n\n"

        return {
            "completion": completion,
            "tool_calls": tool_calls,
            "display_text": display_text,
            "llm_calls": llm_calls,
        }

    def _route_after_llm(self, state: AgentState) -> RouteAfterLLM:
        return "tools" if state.get("tool_calls") else "end"

    # ------------------------------------------------------------------ #
    # Nœud 3 : exécution des outils demandés par le LLM.
    # ------------------------------------------------------------------ #
    def _node_tool_node(self, state: AgentState) -> Dict[str, Any]:
        completion = state.get("completion")
        trace = state.get("trace")
        tool_calls = state.get("tool_calls", []) or []

        display_chunks: List[Dict[str, Any]] = []

        for tc in tool_calls:
            tool_call_id = tc.get("id")
            name, args = self._parse_tool_call(tc)

            with self._subspan(trace, "LangGraph — outil : %s" % name) as span:
                self._set_span_attr(span, "arguments", args)
                chunks, result = self._exec_tool(name, args)

            display_chunks.extend(chunks)

            if completion is not None:
                completion.with_tool_output(
                    json.dumps(result, ensure_ascii=False, default=str),
                    tool_call_id=tool_call_id,
                )

        return {
            "completion": completion,
            "tool_calls": [],
            "display_chunks": display_chunks,
        }

    # =============================================================================
    # Déclaration des outils LLM Mesh
    # =============================================================================

    def _tool_specs(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_tables",
                    "description": (
                        "Liste les tables disponibles sur la connexion SQL. "
                        "Retourne des noms au format base.table."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_sql",
                    "description": (
                        "Exécute une requête SQL en lecture seule. "
                        "La requête doit être un SELECT ou un WITH unique. "
                        "Retourne au maximum %d lignes au LLM." % MAX_RESULT_ROWS
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Requête SQL SELECT/WITH à exécuter.",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "profile_table",
                    "description": (
                        "Profil complet d'une table SQL : colonnes, types, "
                        "exemples de valeurs, taux de nulls, cardinalités et "
                        "nombre de lignes. À appeler obligatoirement avant "
                        "de rédiger un glossaire métier."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {
                                "type": "string",
                                "description": "Nom de table Impala obligatoire au format base.table, par exemple default.customers.",
                            },
                        },
                        "required": ["table"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "export_markdown",
                    "description": (
                        "Publie un document Markdown comme fichier téléchargeable "
                        "dans Agent Hub, par exemple un glossaire métier."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "Nom du fichier, terminé par .md.",
                            },
                            "content": {
                                "type": "string",
                                "description": "Contenu Markdown complet à exporter.",
                            },
                        },
                        "required": ["filename", "content"],
                    },
                },
            },
        ]

    # =============================================================================
    # Exécution des outils
    # =============================================================================

    def _exec_tool(self, name: str, args: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if name == "list_tables":
            return self._tool_list_tables()

        if name == "run_sql":
            return self._tool_run_sql(str(args.get("query", "")))

        if name == "profile_table":
            return self._tool_profile_table(str(args.get("table", "")))

        if name == "export_markdown":
            return self._tool_export_markdown(
                filename=str(args.get("filename", "")),
                content=str(args.get("content", "")),
            )

        return (
            [self._txt("⚠️ Outil inconnu `%s`.\n\n" % name)],
            {"error": "Outil inconnu : %s" % name},
        )

    def _tool_list_tables(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Liste les tables Impala sous forme base.table.

        Impala ne s'appuie pas toujours sur information_schema selon les versions
        et les droits. On utilise donc SHOW DATABASES puis SHOW TABLES IN <base>.
        Si DEFAULT_DATABASE est renseigné, on limite la recherche à cette base.
        """
        chunks = [self._banner("list_tables", "SHOW DATABASES / SHOW TABLES IN <base>", lang="sql")]

        try:
            if DEFAULT_DATABASE:
                databases = [DEFAULT_DATABASE]
            else:
                db_df = self.executor.query_to_df("SHOW DATABASES")
                databases = self._first_column_as_strings(db_df)

            names: List[str] = []
            errors: Dict[str, str] = {}

            for database in databases:
                if not self._is_safe_identifier(database):
                    continue

                try:
                    tbl_df = self.executor.query_to_df("SHOW TABLES IN %s" % database)
                    tables = self._first_column_as_strings(tbl_df)
                    for table_name in tables:
                        if self._is_safe_identifier(table_name):
                            names.append("%s.%s" % (database, table_name))
                except Exception as e:
                    errors[database] = str(e)

            names = sorted(set(names))
            shown = names[:MAX_TABLES_FOR_LLM]

            chunks.append(
                self._txt(
                    "→ %d table(s) Impala trouvée(s) au format `base.table`%s.\n\n"
                    % (
                        len(names),
                        " ; %d transmise(s) au LLM" % len(shown)
                        if len(names) > len(shown)
                        else "",
                    )
                )
            )

            if errors:
                chunks.append(
                    self._txt(
                        "→ ⚠️ %d base(s) ignorée(s) car non lisible(s) avec cette connexion.\n\n"
                        % len(errors)
                    )
                )

            return chunks, {
                "tables": shown,
                "total_tables": len(names),
                "truncated": len(names) > len(shown),
                "database_count": len(databases),
                "skipped_databases": list(errors.keys())[:20],
                "required_format": "base.table",
            }

        except Exception as e:
            chunks.append(self._txt("→ ❌ erreur lors de la liste des tables Impala.\n\n"))
            return chunks, {"error": str(e), "required_format": "base.table"}

    def _tool_run_sql(self, sql: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        sql = (sql or "").strip()
        chunks = [self._banner("run_sql", sql)]

        clean = sql.rstrip().rstrip(";").strip()

        error = self._validate_readonly_sql(clean)
        if error:
            chunks.append(self._txt("→ ❌ requête refusée : %s\n\n" % error))
            return chunks, {"error": error}

        try:
            df = self.executor.query_to_df(clean)

        except Exception as e:
            chunks.append(self._txt("→ ❌ erreur SQL.\n\n"))
            return chunks, {"error": str(e)}

        truncated = len(df) > MAX_RESULT_ROWS

        try:
            rows = json.loads(
                df.head(MAX_RESULT_ROWS).to_json(orient="records", date_format="iso")
            )
        except Exception:
            rows = df.head(MAX_RESULT_ROWS).astype(str).to_dict(orient="records")

        chunks.append(
            self._txt(
                "→ %d ligne(s)%s.\n\n"
                % (
                    len(df),
                    " ; résultat transmis au LLM tronqué à %d ligne(s)"
                    % MAX_RESULT_ROWS
                    if truncated
                    else "",
                )
            )
        )

        return chunks, {
            "row_count": int(len(df)),
            "columns": [str(c) for c in df.columns],
            "truncated": truncated,
            "rows": rows,
        }

    def _tool_profile_table(self, table: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        raw_table = (table or "").strip()
        chunks = [self._banner("profile_table", raw_table, lang="")]

        normalized, error = self._normalize_table_name(raw_table)
        if error:
            chunks.append(self._txt("→ ❌ %s\n\n" % error))
            return chunks, {
                "error": error,
                "required_format": "base.table",
                "example": "default.customers",
            }

        table = normalized
        sample_sql = SAMPLE_QUERY.format(table=table, n=SAMPLE_ROWS)

        try:
            df = self.executor.query_to_df(sample_sql)

        except Exception as e:
            chunks.append(self._txt("→ ❌ impossible de lire la table Impala `%s`.\n\n" % table))
            return chunks, {
                "table": table,
                "sample_query": sample_sql,
                "error": str(e),
            }

        stats, n_rows = self._run_stats(table, df.columns)
        columns = self._assemble_profile(df, stats)

        chunks.append(
            self._txt(
                "→ `%s` : %d colonne(s) profilée(s), %s ligne(s) au total.\n\n"
                % (
                    table,
                    len(columns),
                    format(n_rows, ",") if n_rows is not None else "?",
                )
            )
        )

        return chunks, {
            "table": table,
            "required_format": "base.table",
            "sample_query": sample_sql,
            "sample_rows": int(len(df)),
            "n_rows": n_rows,
            "columns": columns,
        }

    def _tool_export_markdown(
        self,
        filename: str,
        content: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        filename = self._sanitize_filename(filename or "export.md")
        if not filename.lower().endswith(".md"):
            filename += ".md"

        chunks = [self._banner("export_markdown", filename, lang="")]

        if not content.strip():
            chunks.append(self._txt("→ ❌ contenu Markdown vide.\n\n"))
            return chunks, {"error": "Contenu Markdown vide."}

        artifact = {
            "id": "export-%s" % re.sub(r"\W+", "-", filename),
            "name": filename,
            "description": "Document généré le %s."
            % datetime.now().strftime("%d/%m/%Y %H:%M"),
            "parts": [
                {
                    "type": "DATA_INLINE",
                    "mimeType": "text/markdown",
                    "dataBase64": base64.b64encode(
                        content.encode("utf-8")
                    ).decode("ascii"),
                }
            ],
        }

        chunks.append({"chunk": {"artifacts": [artifact]}})
        chunks.append(
            self._txt(
                "→ 📦 **%s** disponible dans *See details > Downloads*.\n\n"
                % filename
            )
        )

        if EXPORT_FOLDER_ID:
            try:
                dataiku.Folder(EXPORT_FOLDER_ID).upload_data(
                    filename,
                    content.encode("utf-8"),
                )
                chunks.append(
                    self._txt(
                        "→ 💾 copie également écrite dans le managed folder `%s`.\n\n"
                        % EXPORT_FOLDER_ID
                    )
                )
            except Exception as e:
                chunks.append(
                    self._txt(
                        "→ ⚠️ export vers le managed folder impossible : %s\n\n"
                        % str(e)
                    )
                )

        return chunks, {
            "status": "ok",
            "filename": filename,
            "download": "See details > Downloads",
        }

    # =============================================================================
    # SQL helpers
    # =============================================================================

    def _validate_readonly_sql(self, sql: str) -> Optional[str]:
        if not sql:
            return "requête vide."

        if ";" in sql:
            return "une seule requête est autorisée, sans point-virgule."

        if not re.match(r"^\s*(select|with)\b", sql, re.IGNORECASE):
            return "seules les requêtes SELECT ou WITH sont autorisées."

        forbidden = r"\b(insert|update|delete|merge|drop|alter|create|truncate|grant|revoke|call|exec|execute)\b"
        if re.search(forbidden, sql, re.IGNORECASE):
            return "mot-clé SQL non autorisé dans une connexion en lecture seule."

        return None

    def _is_safe_identifier(self, identifier: str) -> bool:
        return bool(re.match(r"^[A-Za-z_][\w$]*$", identifier or ""))

    def _strip_identifier_quotes(self, value: str) -> str:
        value = (value or "").strip()
        value = value.replace("`", "").replace('"', "")
        return value

    def _normalize_table_name(self, table: str) -> Tuple[Optional[str], Optional[str]]:
        """Normalise un nom de table Impala.

        Format recommandé : base.table.
        Si DEFAULT_DATABASE est renseigné, un nom simple `table` devient
        DEFAULT_DATABASE.table. Sinon on renvoie une erreur explicite.
        """
        table = self._strip_identifier_quotes(table)

        if not table:
            return None, "nom de table vide ; indiquez une table Impala au format `base.table`."

        parts = table.split(".")

        if len(parts) == 1:
            if DEFAULT_DATABASE:
                database, table_name = DEFAULT_DATABASE, parts[0]
            elif REQUIRE_DATABASE_IN_TABLE_NAME:
                return None, (
                    "Impala nécessite ici le nom de la base avant la table : "
                    "utilisez `base.table`, par exemple `default.customers`."
                )
            else:
                return table, None

        elif len(parts) == 2:
            database, table_name = parts

        else:
            return None, (
                "nom de table invalide pour Impala : utilisez exactement `base.table`, "
                "sans catalogue supplémentaire."
            )

        if not self._is_safe_identifier(database) or not self._is_safe_identifier(table_name):
            return None, (
                "nom de table invalide : utilisez uniquement lettres, chiffres, `_` ou `$`, "
                "au format `base.table`."
            )

        return "%s.%s" % (database, table_name), None

    def _is_safe_table_name(self, table: str) -> bool:
        normalized, error = self._normalize_table_name(table)
        return normalized is not None and error is None

    def _first_column_as_strings(self, df) -> List[str]:
        if df is None or df.empty:
            return []
        return [str(x) for x in df.iloc[:, 0].dropna().tolist()]

    def _qcol(self, col: str) -> str:
        col = str(col)
        if not QUOTE_CHAR:
            return col
        escaped = col.replace(QUOTE_CHAR, QUOTE_CHAR * 2)
        return QUOTE_CHAR + escaped + QUOTE_CHAR

    def _run_stats(self, table: str, columns) -> Tuple[Dict[str, Any], Optional[int]]:
        aggs = ["COUNT(*) AS n_rows"]

        for i, col in enumerate(columns):
            qcol = self._qcol(str(col))
            aggs.append("COUNT(%s) AS c_%d" % (qcol, i))
            if WITH_DISTINCT:
                aggs.append("COUNT(DISTINCT %s) AS d_%d" % (qcol, i))

        sql = "SELECT %s FROM %s" % (", ".join(aggs), table)

        try:
            sdf = self.executor.query_to_df(sql)
            if sdf.empty:
                return {}, None

            row = {str(k).lower(): v for k, v in sdf.iloc[0].items()}
            n_rows = int(row["n_rows"])

            stats: Dict[str, Any] = {}

            for i, col in enumerate(columns):
                filled = int(row.get("c_%d" % i, 0))
                stats[str(col)] = {
                    "taux_null": round(1 - filled / n_rows, 3) if n_rows else None,
                    "valeurs_distinctes": (
                        int(row["d_%d" % i])
                        if WITH_DISTINCT and ("d_%d" % i) in row
                        else None
                    ),
                }

            return stats, n_rows

        except Exception:
            return {}, None

    def _assemble_profile(self, df, stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        columns: List[Dict[str, Any]] = []

        for col in df.columns:
            serie = df[col]
            info: Dict[str, Any] = {
                "name": str(col),
                "type": str(serie.dtype),
            }

            try:
                exemples = (
                    serie.dropna()
                    .astype(str)
                    .str[:MAX_CELL_LEN]
                    .unique()[:MAX_EXAMPLES]
                    .tolist()
                )
            except Exception:
                exemples = []

            info["exemples"] = exemples
            info.update(stats.get(str(col), {}))
            columns.append(info)

        return columns

    # =============================================================================
    # Conversation helpers
    # =============================================================================

    def _conversation_context(self, messages: List[Dict[str, Any]]) -> str:
        last_table = self._last_table_from_history(messages)

        if not last_table:
            return (
                "Contexte Impala : les tables doivent être référencées au format `base.table`. "
                "Si l'utilisateur donne seulement une table, demande la base Impala, sauf si "
                "DEFAULT_DATABASE est configuré."
            )

        return (
            "Contexte conversationnel Impala : la dernière table SQL connue est `%s`. "
            "Si l'utilisateur dit « cette table », « la même table », "
            "« celle-ci », « cette dernière » ou « le glossaire », "
            "il fait probablement référence à `%s`. Les futurs appels d'outils doivent "
            "conserver le format `base.table`."
            % (last_table, last_table)
        )

    def _last_table_from_history(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        patterns = [
            r"#\s*Glossaire métier\s+—\s+`([^`]+\.[^`]+)`",
            r"\btable\s+[«\\\"'`]?([A-Za-z_][\\w$]*\.[A-Za-z_][\\w$]*)[»\\\"'`]?",
            r"\b([A-Za-z_][\\w$]*\.[A-Za-z_][\\w$]*)\b",
        ]

        for msg in reversed(messages or []):
            text = self._msg_text(msg)
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    table = match.group(1)
                    normalized, error = self._normalize_table_name(table)
                    if normalized and not error:
                        return normalized

        return None

    def _msg_text(self, message: Dict[str, Any]) -> str:
        content = message.get("content", "")

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if "text" in item:
                        parts.append(str(item.get("text", "")))
                    elif item.get("type") == "text":
                        parts.append(str(item.get("content", "")))
                else:
                    parts.append(str(item))
            return "\n".join(p for p in parts if p)

        return str(content)

    # =============================================================================
    # Tool calling helpers
    # =============================================================================

    def _get_tool_calls(self, resp) -> List[Dict[str, Any]]:
        tool_calls = getattr(resp, "tool_calls", None)

        if tool_calls is None:
            raw = getattr(resp, "_raw", None) or {}
            tool_calls = (
                raw.get("toolCalls")
                or raw.get("tool_calls")
                or raw.get("toolCalls".lower())
            )

        normalized: List[Dict[str, Any]] = []

        for tc in tool_calls or []:
            if isinstance(tc, dict):
                normalized.append(tc)
                continue

            # Fallback si le connecteur renvoie des objets.
            item: Dict[str, Any] = {}
            for attr in ("id", "type"):
                if hasattr(tc, attr):
                    item[attr] = getattr(tc, attr)

            fn = getattr(tc, "function", None)
            if fn is not None:
                if isinstance(fn, dict):
                    item["function"] = fn
                else:
                    item["function"] = {
                        "name": getattr(fn, "name", None),
                        "arguments": getattr(fn, "arguments", None),
                    }

            normalized.append(item)

        return normalized

    def _parse_tool_call(self, tc: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        fn = (
            tc.get("function")
            or tc.get("functionCall")
            or tc.get("function_call")
            or {}
        )

        if not isinstance(fn, dict):
            fn = {}

        name = (
            fn.get("name")
            or tc.get("name")
            or tc.get("tool_name")
            or "unknown"
        )

        raw_args = (
            fn.get("arguments")
            or fn.get("args")
            or tc.get("arguments")
            or tc.get("args")
            or {}
        )

        if isinstance(raw_args, dict):
            args = raw_args
        else:
            try:
                args = json.loads(raw_args or "{}")
                if not isinstance(args, dict):
                    args = {}
            except Exception:
                args = {}

        return str(name), args

    # =============================================================================
    # Dataiku / streaming helpers
    # =============================================================================

    def _txt(self, text: str) -> Dict[str, Any]:
        return {"chunk": {"text": text or ""}}

    def _banner(self, tool: str, payload: str, lang: str = "sql") -> Dict[str, Any]:
        payload = payload or ""
        shown = payload if len(payload) <= MAX_SQL_DISPLAY else payload[:MAX_SQL_DISPLAY] + " …"
        return self._txt(
            "🛠️ **Outil `%s`**\n```%s\n%s\n```\n"
            % (tool, lang, shown)
        )

    def _sanitize_filename(self, filename: str) -> str:
        filename = filename.split("/")[-1].split("\\")[-1]
        filename = re.sub(r"[^\w.\-]+", "_", filename).strip("._-")
        return filename or "export.md"

    def _subspan(self, trace, name: str):
        if trace is None:
            return nullcontext()
        try:
            return trace.subspan(name)
        except Exception:
            return nullcontext()

    def _set_span_attr(self, span, key: str, value: Any) -> None:
        try:
            if span is not None:
                span.attributes[key] = value
        except Exception:
            pass

    def _resolve_llm_id(self) -> str:
        if LLM_ID:
            return LLM_ID

        llms = self.project.list_llms()
        if not llms:
            raise Exception("Aucun LLM disponible dans le projet : renseignez LLM_ID.")

        first = llms[0]
        return first.id if hasattr(first, "id") else first["id"]
