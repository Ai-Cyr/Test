# -*- coding: utf-8 -*-
"""
Agent code Dataiku — Glossaire métier d'une table (via connexion SQL)
=====================================================================
Variante « push-down SQL » : le profilage est exécuté directement dans la base
via SQLExecutor2, sans dataset Dataiku ni chargement de la table en pandas.

  1. Extrait le nom de la table du message (ex. « la table public.customers »).
  2. Profile la table en SQL :
       - échantillon de lignes (SAMPLE_QUERY),
       - nb de lignes total, taux de nulls et cardinalité (COUNT DISTINCT)
         exacts sur TOUTE la table, en une seule requête d'agrégats.
  3. Génère le glossaire métier via le LLM Mesh (tableau Markdown).
  4. Option « applique » : produit un script SQL COMMENT ON prêt à exécuter.

Mise en place :
  - Agents > New agent > Code agent, coller ce code.
  - Renseigner SQL_CONNECTION (nom de la connexion SQL Dataiku) et LLM_ID.
  - Adapter si besoin SAMPLE_QUERY / QUOTE_CHAR au dialecte (voir ci-dessous).
  - Test : « Fais le glossaire métier de la table public.customers »
"""

import json
import re

import dataiku
from dataiku import SQLExecutor2
from dataiku.llm.python import BaseLLM

# --- À configurer ----------------------------------------------------------
SQL_CONNECTION = "REMPLACER_PAR_VOTRE_CONNEXION"  # nom de la connexion SQL Dataiku
LLM_ID = ""          # ex. "openai:ma-connexion:gpt-4o" ; vide = 1er LLM du projet

QUOTE_CHAR = '"'     # quoting des COLONNES : '"' Postgres/Snowflake/Oracle,
                     # '`' MySQL/BigQuery, '[' non géré -> mettre '' (désactivé)
SAMPLE_QUERY = "SELECT * FROM {table} LIMIT {n}"
# SQL Server : "SELECT TOP {n} * FROM {table}"
# Oracle     : "SELECT * FROM {table} FETCH FIRST {n} ROWS ONLY"

SAMPLE_ROWS = 100    # taille de l'échantillon pour les exemples de valeurs
MAX_EXAMPLES = 5     # nb d'exemples de valeurs par colonne
MAX_CELL_LEN = 60    # troncature des valeurs d'exemple
WITH_DISTINCT = True # COUNT(DISTINCT) exact ; mettre False si table très volumineuse

SYSTEM_PROMPT = """Tu es un data steward senior chargé de documenter un patrimoine de données.
À partir du profil de table fourni (colonnes, types, exemples, statistiques),
tu produis un glossaire métier en français.

Règles :
- Déduis le sens métier de chaque colonne à partir de son nom, son type,
  ses valeurs et ses statistiques (taux de nulls, nb de valeurs distinctes).
- Reste factuel : si le sens est incertain, dis-le dans "remarques" plutôt que d'inventer.
- Les définitions doivent être compréhensibles par un non-technicien (1 à 2 phrases).
- Signale dans "remarques" les données potentiellement personnelles/sensibles (RGPD),
  les clés probables (cardinalité = nb de lignes) et les colonnes très vides.
- Réponds UNIQUEMENT avec un objet JSON valide, sans aucun texte autour, au format :
{
  "table": {"libelle": "...", "description": "..."},
  "colonnes": [
    {
      "colonne": "nom_technique",
      "libelle_metier": "Libellé lisible",
      "definition": "Définition métier claire.",
      "type": "type technique",
      "exemples": "val1, val2",
      "remarques": "sensibilité, qualité, ambiguïté... ou vide"
    }
  ]
}"""


class MyLLM(BaseLLM):

    def __init__(self):
        self.client = dataiku.api_client()
        self.project = self.client.get_default_project()
        self.executor = SQLExecutor2(connection=SQL_CONNECTION)

    # ------------------------------------------------------------------ #
    def process(self, query, settings, trace):
        user_msg = str(query["messages"][-1]["content"])

        table = self._extract_table_name(user_msg)
        if table is None:
            return {"text": self._help_message()}

        with trace.subspan("Profilage SQL de la table"):
            try:
                profile = self._profile_table(table)
            except Exception as e:
                return {"text": "❌ Impossible de lire `%s` sur la connexion `%s` :\n"
                                "```\n%s\n```" % (table, SQL_CONNECTION, e)}

        with trace.subspan("Génération du glossaire (LLM)") as span:
            glossary, raw_text = self._generate_glossary(table, profile, span)

        if glossary is None:
            return {"text": raw_text}  # JSON illisible : réponse brute du LLM

        text = self._render_markdown(table, profile, glossary)

        if re.search(r"\bappli\w*|\bapply\b|comment on", user_msg, re.IGNORECASE):
            text += "\n\n" + self._render_comment_script(table, glossary)

        return {"text": text}

    # ------------------------------------------------------------------ #
    def _extract_table_name(self, message):
        """Repère un nom de table (schema.table accepté) dans le message."""
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
            pass  # information_schema indisponible sur certaines bases
        return ("Je n'ai pas identifié la table dans votre message. Précisez-la, "
                "par exemple : « Fais le glossaire métier de la table "
                "public.customers »" + tables)

    # ------------------------------------------------------------------ #
    def _qcol(self, col):
        return QUOTE_CHAR + col + QUOTE_CHAR if QUOTE_CHAR else col

    def _profile_table(self, table):
        """Échantillon + statistiques exactes calculées dans la base."""
        df = self.executor.query_to_df(
            SAMPLE_QUERY.format(table=table, n=SAMPLE_ROWS))

        # Une seule requête d'agrégats sur toute la table
        stats, n_rows = {}, None
        aggs = ["COUNT(*) AS n_rows"]
        for i, col in enumerate(df.columns):
            aggs.append("COUNT(%s) AS c_%d" % (self._qcol(col), i))
            if WITH_DISTINCT:
                aggs.append("COUNT(DISTINCT %s) AS d_%d" % (self._qcol(col), i))
        try:
            sdf = self.executor.query_to_df(
                "SELECT %s FROM %s" % (", ".join(aggs), table))
            row = {str(k).lower(): v for k, v in sdf.iloc[0].items()}
            n_rows = int(row["n_rows"])
            for i, col in enumerate(df.columns):
                filled = int(row["c_%d" % i])
                stats[col] = {
                    "taux_null": round(1 - filled / n_rows, 3) if n_rows else None,
                    "valeurs_distinctes": int(row["d_%d" % i]) if WITH_DISTINCT else None,
                }
        except Exception:
            pass  # stats exactes indisponibles -> on garde l'échantillon seul

        columns = []
        for col in df.columns:
            serie = df[col]
            info = {"name": col, "type": str(serie.dtype)}
            info["exemples"] = (serie.dropna().astype(str).str[:MAX_CELL_LEN]
                                .unique()[:MAX_EXAMPLES].tolist())
            info.update(stats.get(col, {}))
            columns.append(info)
        return {"n_rows": n_rows, "columns": columns}

    # ------------------------------------------------------------------ #
    def _generate_glossary(self, table, profile, span):
        llm = self.project.get_llm(self._resolve_llm_id())
        completion = llm.new_completion()
        completion.with_message(SYSTEM_PROMPT, role="system")
        completion.with_message(
            "Table SQL : %s\nNombre de lignes : %s\nColonnes (profil) :\n%s"
            % (table, profile.get("n_rows"),
               json.dumps(profile["columns"], ensure_ascii=False, indent=1)),
            role="user")
        resp = completion.execute()
        if not resp.success:
            raise Exception("Échec de l'appel au LLM Mesh (vérifiez LLM_ID).")
        try:
            span.append_trace(resp.trace)
        except Exception:
            pass

        raw = resp.text.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
        try:
            return json.loads(raw), raw
        except Exception:
            return None, raw

    def _resolve_llm_id(self):
        if LLM_ID:
            return LLM_ID
        llms = self.project.list_llms()
        if not llms:
            raise Exception("Aucun LLM disponible dans le projet : renseignez LLM_ID.")
        first = llms[0]
        return first.id if hasattr(first, "id") else first["id"]

    # ------------------------------------------------------------------ #
    def _render_markdown(self, table, profile, glossary):
        tinfo = glossary.get("table", {})
        stats = {c["name"]: c for c in profile["columns"]}
        lines = [
            "# Glossaire métier — `%s`" % table,
            "**%s** : %s" % (tinfo.get("libelle", table),
                             tinfo.get("description", "")),
            "_Volumétrie : %s lignes (connexion `%s`)._"
            % (profile.get("n_rows", "?"), SQL_CONNECTION),
            "",
            "| Colonne | Libellé métier | Définition | Type | Exemples "
            "| % nulls | Distincts | Remarques |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for c in glossary.get("colonnes", []):
            s = stats.get(c.get("colonne"), {})
            null_pct = s.get("taux_null")
            lines.append("| `%s` | %s | %s | %s | %s | %s | %s | %s |" % (
                c.get("colonne", ""), c.get("libelle_metier", ""),
                c.get("definition", ""), c.get("type", ""),
                c.get("exemples", ""),
                "%.0f%%" % (null_pct * 100) if null_pct is not None else "-",
                s.get("valeurs_distinctes", "-"),
                c.get("remarques", "")))
        lines.append("\n_Astuce : ajoutez « applique » à votre message pour obtenir "
                     "le script SQL `COMMENT ON` documentant la table dans la base._")
        return "\n".join(lines)

    def _render_comment_script(self, table, glossary):
        lines = ["📝 Script SQL pour documenter la table dans la base "
                 "(syntaxe Postgres/Oracle/Snowflake ; MySQL : `ALTER TABLE ... "
                 "MODIFY`) :", "```sql"]
        desc = glossary.get("table", {}).get("description")
        if desc:
            lines.append("COMMENT ON TABLE %s IS '%s';"
                         % (table, desc.replace("'", "''")))
        for c in glossary.get("colonnes", []):
            txt = ("%s — %s" % (c.get("libelle_metier", ""),
                                c.get("definition", ""))).strip(" —")
            lines.append("COMMENT ON COLUMN %s.%s IS '%s';"
                         % (table, self._qcol(c.get("colonne", "")),
                            txt.replace("'", "''")))
        lines.append("```")
        return "\n".join(lines)
