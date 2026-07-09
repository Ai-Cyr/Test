# -*- coding: utf-8 -*-
"""
Agent code Dataiku — Glossaire métier d'une table SQL (version STREAMING)
=========================================================================
L'agent streame sa réponse au fur et à mesure (process_stream) et affiche :
  🔍 la table détectée,
  🛠️ chaque appel d'outil SQL (la requête dans un bloc ```sql + résumé du résultat),
  🧠 l'appel au LLM, puis le glossaire rédigé token par token.

Différence avec la version non streamée : le LLM rédige directement le
Markdown final (plus de JSON intermédiaire), ce qui permet un vrai streaming.
Toutes les étapes sont aussi enregistrées dans la trace (subspans),
visibles dans l'onglet Trace / Trace Explorer.

Mise en place :
  - Agents > New agent > Code agent (code env Python >= 3.10), coller ce code.
  - Renseigner SQL_CONNECTION et LLM_ID ; adapter SAMPLE_QUERY / QUOTE_CHAR
    au dialecte si besoin (SQL Server, Oracle, MySQL...).
  - Test : « Fais le glossaire métier de la table public.customers »
           « ... et applique-le » -> ajoute le script COMMENT ON à la fin.
"""

import json
import re

import dataiku
from dataiku import SQLExecutor2
from dataiku.llm.python import BaseLLM
from dataikuapi.dss.llm import (DSSLLMStreamedCompletionChunk,
                                DSSLLMStreamedCompletionFooter)

# --- À configurer ----------------------------------------------------------
SQL_CONNECTION = "REMPLACER_PAR_VOTRE_CONNEXION"  # nom de la connexion SQL Dataiku
LLM_ID = ""          # ex. "openai:ma-connexion:gpt-4o" ; vide = 1er LLM du projet

QUOTE_CHAR = '"'     # quoting des COLONNES : '"' Postgres/Snowflake/Oracle,
                     # '`' MySQL/BigQuery, '' pour désactiver
SAMPLE_QUERY = "SELECT * FROM {table} LIMIT {n}"
# SQL Server : "SELECT TOP {n} * FROM {table}"
# Oracle     : "SELECT * FROM {table} FETCH FIRST {n} ROWS ONLY"

SAMPLE_ROWS = 100
MAX_EXAMPLES = 5
MAX_CELL_LEN = 60
WITH_DISTINCT = True   # COUNT(DISTINCT) exact ; False si table très volumineuse
MAX_SQL_DISPLAY = 500  # troncature des requêtes affichées dans le chat

SYSTEM_PROMPT = """Tu es un data steward senior chargé de documenter un patrimoine de données.
À partir du profil de table fourni (colonnes, types, exemples, statistiques),
tu rédiges DIRECTEMENT le livrable final en Markdown, en français :

1. Un titre : « # Glossaire métier — `<table>` »
2. Une à deux phrases : libellé métier de la table + description.
3. Un tableau Markdown avec les colonnes :
   | Colonne | Libellé métier | Définition | Type | Exemples | % nulls | Distincts | Remarques |
   - Recopie les statistiques fournies (% nulls, valeurs distinctes) sans les recalculer ;
     mets « - » si elles sont absentes.
4. Si « Script COMMENT ON demandé : oui », termine par un bloc ```sql contenant
   COMMENT ON TABLE puis un COMMENT ON COLUMN par colonne (échapper les ' en '').

Règles :
- Déduis le sens métier de chaque colonne (nom, type, valeurs, stats).
- Reste factuel : si le sens est incertain, dis-le dans Remarques plutôt que d'inventer.
- Définitions compréhensibles par un non-technicien (1 à 2 phrases).
- Signale dans Remarques : données personnelles/sensibles (RGPD), clés probables
  (cardinalité = nb de lignes), colonnes très vides.
- Ne produis RIEN d'autre que ce livrable (pas de préambule ni conclusion)."""


class MyLLM(BaseLLM):

    def __init__(self):
        self.client = dataiku.api_client()
        self.project = self.client.get_default_project()
        self.executor = SQLExecutor2(connection=SQL_CONNECTION)

    # ------------------------------------------------------------------ #
    # Mode non streamé : on rejoue simplement le flux streamé.
    def process(self, query, settings, trace):
        text = "".join(c["chunk"]["text"]
                       for c in self.process_stream(query, settings, trace)
                       if isinstance(c, dict) and "chunk" in c)
        return {"text": text}

    # ------------------------------------------------------------------ #
    def process_stream(self, query, settings, trace):
        user_msg = str(query["messages"][-1]["content"])

        table = self._extract_table_name(user_msg)
        if table is None:
            yield self._txt(self._help_message())
            return

        yield self._txt("🔍 **Table détectée :** `%s` (connexion `%s`)\n\n"
                        % (table, SQL_CONNECTION))

        # ---- Outil SQL n°1 : échantillon ------------------------------ #
        sql_sample = SAMPLE_QUERY.format(table=table, n=SAMPLE_ROWS)
        yield self._tool_call("échantillon", sql_sample)
        with trace.subspan("Outil SQL : échantillon") as sp:
            sp.attributes["sql"] = sql_sample
            try:
                df = self.executor.query_to_df(sql_sample)
            except Exception as e:
                yield self._txt("❌ Impossible de lire `%s` :\n```\n%s\n```"
                                % (table, e))
                return
        yield self._txt("→ %d lignes × %d colonnes lues.\n\n"
                        % (len(df), len(df.columns)))

        # ---- Outil SQL n°2 : statistiques sur toute la table ---------- #
        sql_stats = self._build_stats_query(table, df.columns)
        yield self._tool_call("statistiques (table complète)", sql_stats)
        with trace.subspan("Outil SQL : agrégats") as sp:
            sp.attributes["sql"] = sql_stats
            stats, n_rows = self._run_stats(sql_stats, df.columns)
        if n_rows is not None:
            yield self._txt("→ %s lignes au total, taux de nulls et cardinalités "
                            "calculés pour %d colonnes.\n\n"
                            % (format(n_rows, ","), len(df.columns)))
        else:
            yield self._txt("→ ⚠️ Statistiques exactes indisponibles, "
                            "je poursuis avec l'échantillon seul.\n\n")

        profile = self._assemble_profile(df, stats, n_rows)

        # ---- Appel LLM streamé ----------------------------------------- #
        llm_id = self._resolve_llm_id()
        with_comments = bool(re.search(r"\bappli\w*|\bapply\b|comment on",
                                       user_msg, re.IGNORECASE))
        yield self._txt("🧠 **Appel LLM** `%s` — rédaction du glossaire%s...\n\n---\n\n"
                        % (llm_id, " + script COMMENT ON" if with_comments else ""))

        llm = self.project.get_llm(llm_id)
        completion = llm.new_completion()
        completion.with_message(SYSTEM_PROMPT, role="system")
        completion.with_message(
            "Table SQL : %s\nNombre de lignes : %s\n"
            "Script COMMENT ON demandé : %s\nColonnes (profil) :\n%s"
            % (table, n_rows if n_rows is not None else "inconnu",
               "oui" if with_comments else "non",
               json.dumps(profile, ensure_ascii=False, indent=1)),
            role="user")

        with trace.subspan("Génération du glossaire (LLM)") as span:
            for chunk in completion.execute_streamed():
                if isinstance(chunk, DSSLLMStreamedCompletionChunk):
                    t = chunk.data.get("text", "")
                    if t:
                        yield self._txt(t)
                elif isinstance(chunk, DSSLLMStreamedCompletionFooter):
                    try:  # rattache la trace de l'appel LLM à la nôtre
                        span.append_trace(chunk.data["trace"])
                    except Exception:
                        pass

    # ------------------------------------------------------------------ #
    # Helpers de streaming
    def _txt(self, text):
        return {"chunk": {"text": text}}

    def _tool_call(self, label, sql):
        shown = sql if len(sql) <= MAX_SQL_DISPLAY else sql[:MAX_SQL_DISPLAY] + " …"
        return self._txt("🛠️ **Outil SQL — %s**\n```sql\n%s\n```\n" % (label, shown))

    # ------------------------------------------------------------------ #
    # Détection de la table
    def _extract_table_name(self, message):
        m = re.search(r"table\s+[«\"'`]?([\w$]+(?:\.[\w$]+){0,2})",
                      message, re.IGNORECASE)
        if not m:  # à défaut : premier identifiant qualifié « schema.table »
            m = re.search(r"\b([\w$]+\.[\w$]+)\b", message)
        if not m:
            return None
        name = m.group(1)
        return name if re.match(r"^[\w$.]+$", name) else None  # anti-injection

    def _help_message(self):
        tables = ""
        try:
            df = self.executor.query_to_df(
                "SELECT table_schema, table_name FROM information_schema.tables "
                "WHERE table_type = 'BASE TABLE' ORDER BY 1, 2")
            names = (df.iloc[:, 0].astype(str) + "." +
                     df.iloc[:, 1].astype(str)).tolist()[:40]
            tables = "\n\nTables disponibles :\n" + \
                     "\n".join("- `%s`" % n for n in names)
        except Exception:
            pass
        return ("Je n'ai pas identifié la table dans votre message. Précisez-la, "
                "par exemple : « Fais le glossaire métier de la table "
                "public.customers »" + tables)

    # ------------------------------------------------------------------ #
    # Profilage SQL
    def _qcol(self, col):
        return QUOTE_CHAR + col + QUOTE_CHAR if QUOTE_CHAR else col

    def _build_stats_query(self, table, columns):
        aggs = ["COUNT(*) AS n_rows"]
        for i, col in enumerate(columns):
            aggs.append("COUNT(%s) AS c_%d" % (self._qcol(col), i))
            if WITH_DISTINCT:
                aggs.append("COUNT(DISTINCT %s) AS d_%d" % (self._qcol(col), i))
        return "SELECT %s FROM %s" % (", ".join(aggs), table)

    def _run_stats(self, sql, columns):
        try:
            sdf = self.executor.query_to_df(sql)
            row = {str(k).lower(): v for k, v in sdf.iloc[0].items()}
            n_rows = int(row["n_rows"])
            stats = {}
            for i, col in enumerate(columns):
                filled = int(row["c_%d" % i])
                stats[col] = {
                    "taux_null": round(1 - filled / n_rows, 3) if n_rows else None,
                    "valeurs_distinctes": (int(row["d_%d" % i])
                                           if WITH_DISTINCT else None),
                }
            return stats, n_rows
        except Exception:
            return {}, None

    def _assemble_profile(self, df, stats, n_rows):
        columns = []
        for col in df.columns:
            serie = df[col]
            info = {"name": col, "type": str(serie.dtype)}
            info["exemples"] = (serie.dropna().astype(str).str[:MAX_CELL_LEN]
                                .unique()[:MAX_EXAMPLES].tolist())
            info.update(stats.get(col, {}))
            columns.append(info)
        return columns

    # ------------------------------------------------------------------ #
    def _resolve_llm_id(self):
        if LLM_ID:
            return LLM_ID
        llms = self.project.list_llms()
        if not llms:
            raise Exception("Aucun LLM disponible dans le projet : renseignez LLM_ID.")
        first = llms[0]
        return first.id if hasattr(first, "id") else first["id"]
