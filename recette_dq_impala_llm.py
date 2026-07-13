# -*- coding: utf-8 -*-
"""
Recette Python Dataiku — Data Quality sur une table Impala (v2)
----------------------------------------------------------------
Entrée  : un dataset Dataiku pointant vers la table (connexion Impala)
Sortie  : un dataset de résultats DQ, 1 ligne par colonne analysée
Option  : un 2e dataset de sortie avec le rapport global du LLM

Contrôles SQL (un seul scan de la table) :
  - tous types : nb_rows, nb_null, pct_null, nb_distinct, pct_distinct,
                 min/max, complétude, colonne constante, candidate clé unique
  - numériques : moyenne, écart-type, nb de valeurs négatives, nb de zéros
  - chaînes    : nb de valeurs vides/espaces, longueurs min / max / moyenne
  - dates      : nb de dates dans le futur
  - table      : nb de lignes en doublon strict (optionnel)

Enrichissement LLM Mesh (optionnel, tolérant aux pannes) :
  - llm_severity        : OK / WARNING / CRITICAL par colonne
  - llm_comment         : diagnostic court en français
  - llm_suggested_check : règle de contrôle proposée
  - rapport global de la table (log + dataset optionnel)
"""

import json
import dataiku
import pandas as pd
from datetime import datetime

# ==================================================================
# 1. PARAMÈTRES — à adapter
# ==================================================================
INPUT_DATASET_NAME  = "ma_table_source"   # dataset d'entrée dans le Flow
OUTPUT_DATASET_NAME = "dq_resultats"      # dataset de sortie (détail par colonne)

# Laisser vide pour auto-détection ; sinon forcer "schema.table"
TABLE_OVERRIDE = ""

# True = COUNT(DISTINCT) exact (plus lent) / False = NDV() approché (rapide)
EXACT_DISTINCT = False

# Contrôle table : comptage des lignes en doublon strict (2e requête,
# potentiellement coûteuse sur de très grosses tables)
CHECK_DUPLICATES = True

# ----- LLM Mesh -----
USE_LLM = True
# ID du LLM dans le LLM Mesh, ex. "mistral:ma_connexion_mistral:mistral-large"
# Laisser vide pour afficher la liste des LLM disponibles dans les logs.
LLM_ID = ""
LLM_BATCH_SIZE = 40          # nb de colonnes envoyées par appel LLM
# Dataset optionnel pour le rapport global (à déclarer comme 2e sortie
# de la recette). Laisser vide pour ne pas l'écrire.
REPORT_DATASET_NAME = ""

# Types Dataiku non profilables en SQL Impala
COMPLEX_TYPES = {"array", "map", "object", "geometry", "geopoint"}
NUMERIC_TYPES = {"tinyint", "smallint", "int", "bigint", "float", "double"}
DATE_TYPES    = {"date"}

# ==================================================================
# 2. DATASETS ET LOCALISATION DE LA TABLE
# ==================================================================
input_ds  = dataiku.Dataset(INPUT_DATASET_NAME)
output_ds = dataiku.Dataset(OUTPUT_DATASET_NAME)

loc  = input_ds.get_location_info()
info = loc.get("info", {})

if TABLE_OVERRIDE:
    table_label = TABLE_OVERRIDE
elif info.get("table"):
    schema      = info.get("schema") or info.get("databaseName")
    table_label = "{}.{}".format(schema, info["table"]) if schema else info["table"]
else:
    raise ValueError(
        "Impossible de détecter le nom de la table : renseignez TABLE_OVERRIDE "
        "(ex. 'mon_schema.ma_table')."
    )

full_table = ".".join("`{}`".format(p) for p in table_label.split("."))

# ==================================================================
# 3. EXÉCUTEUR SQL SUR LA CONNEXION IMPALA
# ==================================================================
if loc.get("locationInfoType") == "SQL":
    from dataiku import SQLExecutor2
    executor = SQLExecutor2(dataset=input_ds)
else:
    try:
        from dataiku.core.sql import ImpalaExecutor
    except ImportError:
        from dataiku import ImpalaExecutor
    executor = ImpalaExecutor(dataset=input_ds)

# ==================================================================
# 4. REQUÊTE DE PROFILING (un seul passage sur la table)
# ==================================================================
schema_cols  = input_ds.read_schema()
simple_cols  = [c for c in schema_cols if c["type"].lower() not in COMPLEX_TYPES]
complex_cols = [c for c in schema_cols if c["type"].lower() in COMPLEX_TYPES]

distinct_expr = "COUNT(DISTINCT {q})" if EXACT_DISTINCT else "NDV({q})"

select_parts = ["COUNT(*) AS `__total_rows`"]
for c in simple_cols:
    n, t = c["name"], c["type"].lower()
    q = "`{}`".format(n)

    select_parts.append(
        "SUM(CASE WHEN {q} IS NULL THEN 1 ELSE 0 END) AS `{n}__null`".format(q=q, n=n))
    select_parts.append(distinct_expr.format(q=q) + " AS `{}__dist`".format(n))
    select_parts.append("MIN(CAST({q} AS STRING)) AS `{n}__min`".format(q=q, n=n))
    select_parts.append("MAX(CAST({q} AS STRING)) AS `{n}__max`".format(q=q, n=n))

    if t in NUMERIC_TYPES:
        select_parts.append("AVG({q}) AS `{n}__avg`".format(q=q, n=n))
        select_parts.append("STDDEV({q}) AS `{n}__std`".format(q=q, n=n))
        select_parts.append(
            "SUM(CASE WHEN {q} < 0 THEN 1 ELSE 0 END) AS `{n}__neg`".format(q=q, n=n))
        select_parts.append(
            "SUM(CASE WHEN {q} = 0 THEN 1 ELSE 0 END) AS `{n}__zero`".format(q=q, n=n))

    if t == "string":
        select_parts.append(
            "SUM(CASE WHEN {q} IS NOT NULL AND LENGTH(TRIM({q})) = 0 "
            "THEN 1 ELSE 0 END) AS `{n}__empty`".format(q=q, n=n))
        select_parts.append("MIN(LENGTH({q})) AS `{n}__lmin`".format(q=q, n=n))
        select_parts.append("MAX(LENGTH({q})) AS `{n}__lmax`".format(q=q, n=n))
        select_parts.append("AVG(LENGTH({q})) AS `{n}__lavg`".format(q=q, n=n))

    if t in DATE_TYPES:
        select_parts.append(
            "SUM(CASE WHEN CAST({q} AS TIMESTAMP) > NOW() THEN 1 ELSE 0 END) "
            "AS `{n}__future`".format(q=q, n=n))

query = "SELECT\n    " + ",\n    ".join(select_parts) + "\nFROM " + full_table
print("Requête de profiling envoyée à Impala :\n" + query)

stats      = executor.query_to_df(query)
total_rows = int(stats["__total_rows"].iloc[0])
run_ts     = datetime.now()

# ==================================================================
# 5. CONTRÔLE TABLE : LIGNES EN DOUBLON STRICT
# ==================================================================
nb_duplicate_rows = None
if CHECK_DUPLICATES and simple_cols and total_rows > 0:
    group_by = ", ".join("`{}`".format(c["name"]) for c in simple_cols)
    dup_query = (
        "SELECT COALESCE(CAST(SUM(c) - COUNT(*) AS BIGINT), 0) AS nb_dup "
        "FROM (SELECT COUNT(*) AS c FROM {t} GROUP BY {g} "
        "HAVING COUNT(*) > 1) grp"
    ).format(t=full_table, g=group_by)
    try:
        nb_duplicate_rows = int(executor.query_to_df(dup_query)["nb_dup"].iloc[0])
    except Exception as e:
        print("Contrôle doublons ignoré (erreur) : {}".format(e))

# ==================================================================
# 6. MISE EN FORME + INDICATEURS DÉRIVÉS
# ==================================================================
def _i(v):
    return int(v) if pd.notnull(v) else None

def _f(v, r=4):
    return round(float(v), r) if pd.notnull(v) else None

def _s(name, suffix):
    col = "{}__{}".format(name, suffix)
    return stats[col].iloc[0] if col in stats.columns else None

def pct(x):
    return round(100.0 * x / total_rows, 2) if (total_rows and x is not None) else None

rows = []
for c in simple_cols:
    n, t     = c["name"], c["type"].lower()
    nb_null  = _i(_s(n, "null"))
    nb_dist  = _i(_s(n, "dist"))
    pct_null = pct(nb_null)

    rows.append({
        "table_name":      table_label,
        "column_name":     n,
        "data_type":       c["type"],
        "nb_rows":         total_rows,
        "nb_null":         nb_null,
        "pct_null":        pct_null,
        "nb_empty":        _i(_s(n, "empty")),
        "nb_distinct":     nb_dist,
        "pct_distinct":    pct(nb_dist),
        "completeness_pct": round(100 - pct_null, 2) if pct_null is not None else None,
        "is_constant":     (nb_dist is not None and total_rows > 0 and nb_dist <= 1),
        "is_unique":       (nb_dist is not None and total_rows > 0
                            and nb_dist >= total_rows),
        "min_value":       _s(n, "min"),
        "max_value":       _s(n, "max"),
        "avg_value":       _f(_s(n, "avg")),
        "stddev_value":    _f(_s(n, "std")),
        "nb_negative":     _i(_s(n, "neg")),
        "nb_zero":         _i(_s(n, "zero")),
        "min_length":      _i(_s(n, "lmin")),
        "max_length":      _i(_s(n, "lmax")),
        "avg_length":      _f(_s(n, "lavg"), 2),
        "nb_future_dates": _i(_s(n, "future")),
        "nb_duplicate_rows": nb_duplicate_rows,
        "llm_severity":        None,
        "llm_comment":         None,
        "llm_suggested_check": None,
        "insertion_date":  run_ts,
    })

for c in complex_cols:  # types non profilables : ligne de trace
    rows.append({
        "table_name": table_label, "column_name": c["name"],
        "data_type": c["type"], "nb_rows": total_rows,
        "insertion_date": run_ts,
    })

# ==================================================================
# 7. ENRICHISSEMENT VIA LE LLM MESH (Mistral)
# ==================================================================
llm_report = None
if USE_LLM:
    try:
        client  = dataiku.api_client()
        project = client.get_default_project()

        if not LLM_ID:
            print("LLM_ID non renseigné. LLM disponibles dans le LLM Mesh :")
            for item in project.list_llms():
                try:
                    print("  - id: {} | {}".format(item.id, item.description))
                except AttributeError:
                    print("  - {}".format(item))
            raise ValueError("Renseignez LLM_ID avec l'un des id ci-dessus.")

        llm = project.get_llm(LLM_ID)

        def call_llm(user_prompt, system_prompt=None):
            comp = llm.new_completion()
            try:
                comp.settings["temperature"] = 0
            except Exception:
                pass
            if system_prompt:
                comp.with_message(system_prompt, role="system")
            comp.with_message(user_prompt, role="user")
            resp = comp.execute()
            if not resp.success:
                raise RuntimeError("Echec de l'appel LLM")
            return resp.text

        def extract_json(text):
            text = text.strip()
            if text.startswith("```"):
                text = text.strip("`")
                if text.lower().startswith("json"):
                    text = text[4:]
            start, end = text.find("{"), text.rfind("}")
            return json.loads(text[start:end + 1])

        SYSTEM = ("Tu es un expert en qualité de données. Tu réponds UNIQUEMENT "
                  "avec du JSON valide, sans aucun texte autour.")

        # ---- Analyse par colonne, en lots ----
        analyzable = [r for r in rows if r.get("nb_null") is not None]
        by_name    = {r["column_name"]: r for r in rows}
        LLM_FIELDS = ["column_name", "data_type", "nb_rows", "pct_null", "nb_empty",
                      "nb_distinct", "pct_distinct", "is_constant", "is_unique",
                      "min_value", "max_value", "avg_value", "stddev_value",
                      "nb_negative", "nb_zero", "min_length", "max_length",
                      "nb_future_dates"]

        for i in range(0, len(analyzable), LLM_BATCH_SIZE):
            batch = analyzable[i:i + LLM_BATCH_SIZE]
            payload = [{k: r.get(k) for k in LLM_FIELDS if r.get(k) is not None}
                       for r in batch]
            prompt = (
                "Table analysée : {t} ({n} lignes).\n"
                "Pour chaque colonne ci-dessous, évalue la qualité des données et "
                "réponds en JSON strict au format :\n"
                '{{"columns":[{{"column_name":"...","severity":"OK|WARNING|CRITICAL",'
                '"comment":"...","suggested_check":"..."}}]}}\n'
                "Règles : severity=CRITICAL si problème majeur (ex. taux de nulls "
                "très élevé, colonne constante, valeurs aberrantes), WARNING si "
                "point d'attention, OK sinon. comment : diagnostic bref en français "
                "(25 mots max). suggested_check : une règle de contrôle concrète et "
                "actionnable en français (ex. 'valeurs entre 0 et 150', 'non null', "
                "'format email').\n"
                "Statistiques :\n{stats}"
            ).format(t=table_label, n=total_rows,
                     stats=json.dumps(payload, ensure_ascii=False, default=str))

            try:
                parsed = extract_json(call_llm(prompt, SYSTEM))
                for item in parsed.get("columns", []):
                    r = by_name.get(str(item.get("column_name", "")).strip())
                    if r is not None:
                        r["llm_severity"]        = item.get("severity")
                        r["llm_comment"]         = item.get("comment")
                        r["llm_suggested_check"] = item.get("suggested_check")
            except Exception as e:
                print("Lot LLM {}-{} ignoré (erreur) : {}".format(
                    i, i + len(batch), e))

        # ---- Rapport global ----
        sev = [r.get("llm_severity") for r in rows]
        recap = {
            "table": table_label, "nb_lignes": total_rows,
            "nb_colonnes": len(schema_cols),
            "nb_critical": sev.count("CRITICAL"),
            "nb_warning":  sev.count("WARNING"),
            "nb_doublons": nb_duplicate_rows,
            "colonnes_en_alerte": [
                {"colonne": r["column_name"], "severite": r["llm_severity"],
                 "commentaire": r["llm_comment"]}
                for r in rows if r.get("llm_severity") in ("WARNING", "CRITICAL")],
        }
        try:
            llm_report = call_llm(
                "Rédige en français un rapport de synthèse data quality "
                "(8 lignes max, texte brut sans JSON ni markdown) à partir de : "
                + json.dumps(recap, ensure_ascii=False, default=str))
            print("\n===== RAPPORT DQ (LLM) =====\n{}\n============================"
                  .format(llm_report))
        except Exception as e:
            print("Rapport global LLM ignoré (erreur) : {}".format(e))

    except Exception as e:
        print("Enrichissement LLM désactivé pour ce run : {}".format(e))

# ==================================================================
# 8. ÉCRITURE DES RÉSULTATS
# ==================================================================
OUTPUT_COLS = [
    "table_name", "column_name", "data_type", "nb_rows",
    "nb_null", "pct_null", "nb_empty", "nb_distinct", "pct_distinct",
    "completeness_pct", "is_constant", "is_unique",
    "min_value", "max_value", "avg_value", "stddev_value",
    "nb_negative", "nb_zero", "min_length", "max_length", "avg_length",
    "nb_future_dates", "nb_duplicate_rows",
    "llm_severity", "llm_comment", "llm_suggested_check",
    "insertion_date",
]
df_out = pd.DataFrame(rows).reindex(columns=OUTPUT_COLS)

# Pour historiser : cocher "Append instead of overwrite" sur la sortie
output_ds.write_with_schema(df_out)

if REPORT_DATASET_NAME and llm_report:
    sev = df_out["llm_severity"].fillna("")
    df_report = pd.DataFrame([{
        "table_name":     table_label,
        "insertion_date": run_ts,
        "nb_rows":        total_rows,
        "nb_columns":     len(schema_cols),
        "nb_critical":    int((sev == "CRITICAL").sum()),
        "nb_warning":     int((sev == "WARNING").sum()),
        "rapport_llm":    llm_report,
    }])
    dataiku.Dataset(REPORT_DATASET_NAME).write_with_schema(df_report)

print("OK : {} colonnes analysées pour {} (run du {})".format(
    len(df_out), table_label, run_ts.strftime("%Y-%m-%d %H:%M:%S")))
