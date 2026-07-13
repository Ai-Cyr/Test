# -*- coding: utf-8 -*-
"""
Recette Python Dataiku — Data Quality sur une table Impala
-----------------------------------------------------------
Entrée  : un dataset Dataiku pointant vers la table (connexion Impala)
Sortie  : un dataset de résultats DQ, 1 ligne par colonne analysée

Colonnes produites :
    table_name, column_name, data_type,
    nb_rows, nb_null, pct_null, nb_empty,
    nb_distinct, pct_distinct, min_value, max_value,
    insertion_date

Le profiling est exécuté directement dans Impala (un seul scan de la
table), rien n'est rapatrié en mémoire à part le résultat agrégé.
"""

import dataiku
import pandas as pd
from datetime import datetime

# ==================================================================
# 1. PARAMÈTRES — à adapter
# ==================================================================
INPUT_DATASET_NAME  = "ma_table_source"   # nom du dataset d'entrée dans le Flow
OUTPUT_DATASET_NAME = "dq_resultats"      # nom du dataset de sortie

# Laisser vide pour auto-détection depuis le dataset ;
# sinon forcer le nom physique, ex. "mon_schema.ma_table"
TABLE_OVERRIDE = ""

# True  = COUNT(DISTINCT ...) exact (plus lent)
# False = NDV(...) approché, natif Impala (beaucoup plus rapide)
EXACT_DISTINCT = False

# Types Dataiku non profilables en SQL Impala (ignorés dans la requête)
COMPLEX_TYPES = {"array", "map", "object", "geometry", "geopoint"}

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
        "Impossible de détecter le nom de la table depuis le dataset : "
        "renseignez TABLE_OVERRIDE (ex. 'mon_schema.ma_table')."
    )

# Nom quoté pour la requête : `schema`.`table`
full_table = ".".join("`{}`".format(p) for p in table_label.split("."))

# ==================================================================
# 3. EXÉCUTEUR SQL SUR LA CONNEXION IMPALA
# ==================================================================
if loc.get("locationInfoType") == "SQL":
    # Dataset SQL classique
    from dataiku import SQLExecutor2
    executor = SQLExecutor2(dataset=input_ds)
else:
    # Dataset HDFS interrogé via le moteur Impala
    try:
        from dataiku.core.sql import ImpalaExecutor
    except ImportError:
        from dataiku import ImpalaExecutor
    executor = ImpalaExecutor(dataset=input_ds)

# ==================================================================
# 4. REQUÊTE DE PROFILING (un seul passage sur la table)
# ==================================================================
schema_cols  = input_ds.read_schema()  # [{'name': ..., 'type': ...}, ...]
simple_cols  = [c for c in schema_cols if c["type"].lower() not in COMPLEX_TYPES]
complex_cols = [c for c in schema_cols if c["type"].lower() in COMPLEX_TYPES]

distinct_expr = "COUNT(DISTINCT {q})" if EXACT_DISTINCT else "NDV({q})"

select_parts = ["COUNT(*) AS `__total_rows`"]
for c in simple_cols:
    n = c["name"]
    q = "`{}`".format(n)
    select_parts.append(
        "SUM(CASE WHEN {q} IS NULL THEN 1 ELSE 0 END) AS `{n}__null`".format(q=q, n=n)
    )
    select_parts.append(distinct_expr.format(q=q) + " AS `{}__dist`".format(n))
    select_parts.append("MIN(CAST({q} AS STRING)) AS `{n}__min`".format(q=q, n=n))
    select_parts.append("MAX(CAST({q} AS STRING)) AS `{n}__max`".format(q=q, n=n))
    if c["type"].lower() == "string":
        select_parts.append(
            "SUM(CASE WHEN {q} = '' THEN 1 ELSE 0 END) AS `{n}__empty`".format(q=q, n=n)
        )

query = "SELECT\n    " + ",\n    ".join(select_parts) + "\nFROM " + full_table
print("Requête de profiling envoyée à Impala :\n" + query)

stats      = executor.query_to_df(query)
total_rows = int(stats["__total_rows"].iloc[0])
run_ts     = datetime.now()


def pct(x):
    """Pourcentage arrondi, protégé contre une table vide."""
    return round(100.0 * x / total_rows, 2) if total_rows else None


# ==================================================================
# 5. MISE EN FORME — une ligne par colonne
# ==================================================================
rows = []
for c in simple_cols:
    n        = c["name"]
    nb_null  = int(stats["{}__null".format(n)].iloc[0])
    nb_dist  = int(stats["{}__dist".format(n)].iloc[0])
    empty_col = "{}__empty".format(n)
    nb_empty = int(stats[empty_col].iloc[0]) if empty_col in stats.columns else None

    rows.append({
        "table_name":     table_label,
        "column_name":    n,
        "data_type":      c["type"],
        "nb_rows":        total_rows,
        "nb_null":        nb_null,
        "pct_null":       pct(nb_null),
        "nb_empty":       nb_empty,
        "nb_distinct":    nb_dist,
        "pct_distinct":   pct(nb_dist),
        "min_value":      stats["{}__min".format(n)].iloc[0],
        "max_value":      stats["{}__max".format(n)].iloc[0],
        "insertion_date": run_ts,
    })

# Colonnes de type complexe : ligne de trace sans métriques
for c in complex_cols:
    rows.append({
        "table_name": table_label, "column_name": c["name"],
        "data_type": c["type"], "nb_rows": total_rows,
        "nb_null": None, "pct_null": None, "nb_empty": None,
        "nb_distinct": None, "pct_distinct": None,
        "min_value": None, "max_value": None,
        "insertion_date": run_ts,
    })

df_out = pd.DataFrame(rows, columns=[
    "table_name", "column_name", "data_type", "nb_rows",
    "nb_null", "pct_null", "nb_empty", "nb_distinct", "pct_distinct",
    "min_value", "max_value", "insertion_date",
])

# ==================================================================
# 6. ÉCRITURE DU RÉSULTAT
# ==================================================================
# Pour historiser les runs, cocher "Append instead of overwrite"
# sur la sortie dans l'onglet Inputs/Outputs de la recette.
output_ds.write_with_schema(df_out)
print("OK : {} colonnes analysées pour {} (run du {})".format(
    len(df_out), table_label, run_ts.strftime("%Y-%m-%d %H:%M:%S")))
