"""
Dataiku Code Agent: Data Quality on Impala with LangGraph streaming, using SQLExecutor2 only.

How to use in Dataiku DSS:
1. Create / select a Python 3.10+ code env and install the required packages.
2. Create a Code Agent in your project.
3. Paste this file in the Code tab OR put it in project libraries and import MyLLM.
4. Fill the constants in the CONFIGURATION A RENSEIGNER section below.

The agent performs read-only data-quality work on an Impala-backed SQL table through
`dataiku.SQLExecutor2(connection=...)` only:
- schema inspection
- row count / null-rate / distinct-count / min-max profiling
- duplicate-key checks
- required-column checks
- freshness checks on date/timestamp columns
- safe read-only SQL questions when the user asks for an ad-hoc query
- a detailed Markdown data-quality report with findings, thresholds, SQL audit trail,
  and remediation recommendations

Security note: LLM-generated SQL is validated and restricted to SELECT/WITH/SHOW/DESCRIBE.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Tuple, TypedDict, Union

import dataiku
from dataiku import SQLExecutor2
from dataiku.langchain import LangchainToDKUTracer
from dataiku.llm.python import BaseLLM
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1) CONFIGURATION A RENSEIGNER DANS CE FICHIER
# -----------------------------------------------------------------------------
# Aucune variable d'environnement n'est lue par ce fichier.
# Aucune variable projet Dataiku n'est lue par ce fichier.
# Modifiez directement les constantes ci-dessous avant de coller le code dans
# le Code Agent Dataiku.

# LLM Mesh id Dataiku. Exemple selon votre configuration DSS :
#   "openai:ma_connexion_llm:gpt-4o-mini"
#   "azureopenai:ma_connexion_azure:gpt-4o"
LLM_ID = ""  # TODO: renseigner l'identifiant LLM Mesh Dataiku

# Connexion SQL DSS pointant vers Impala.
# Cette version utilise uniquement SQLExecutor2(connection=...).
SQL_CONNECTION = ""  # TODO: nom de la connexion SQL Dataiku vers Impala
SQL_DATABASE = ""    # optionnel : base/schema Impala, exemple "default"
SQL_TABLE = ""       # TODO: table Impala, exemple "ma_table"

# Règles métier optionnelles.
PRIMARY_KEY_COLUMNS: List[str] = []          # Exemple: ["customer_id"]
REQUIRED_COLUMNS: List[str] = []             # Exemple: ["customer_id", "event_date"]
DATE_COLUMNS: List[str] = []                 # Exemple: ["event_date", "updated_at"]

# Paramètres de volumétrie et de rapport.
MAX_PROFILE_COLUMNS = 80
MAX_REPORT_COLUMNS = 80
MAX_RETURN_ROWS = 100
MAX_ROWS_FOR_LLM = 30

# Seuils qualité par défaut.
DEFAULT_MIN_ROW_COUNT = 1
DEFAULT_MAX_NULL_RATE = 0.20
DEFAULT_MAX_EMPTY_RATE = 0.05
DEFAULT_MAX_DUPLICATE_RATE = 0.0
DEFAULT_MAX_DATA_AGE_DAYS: Optional[int] = None  # Exemple: 2, ou None pour désactiver

# Options d'affichage et de génération.
REVEAL_SQL = True
USE_LLM_FINAL_SUMMARY = True
ENABLE_AGENT_HUB_ARTIFACTS = True
ENABLE_PNG_CHARTS = True
MAX_CHART_ROWS = 30


# -----------------------------------------------------------------------------
# Configuration and helpers
# -----------------------------------------------------------------------------

@dataclass
class AgentConfig:
    llm_id: str = LLM_ID
    sql_connection: str = SQL_CONNECTION
    sql_database: str = SQL_DATABASE
    sql_table: str = SQL_TABLE
    primary_key_columns: List[str] = field(default_factory=lambda: list(PRIMARY_KEY_COLUMNS))
    required_columns: List[str] = field(default_factory=lambda: list(REQUIRED_COLUMNS))
    date_columns: List[str] = field(default_factory=lambda: list(DATE_COLUMNS))
    max_profile_columns: int = MAX_PROFILE_COLUMNS
    max_report_columns: int = MAX_REPORT_COLUMNS
    max_return_rows: int = MAX_RETURN_ROWS
    max_rows_for_llm: int = MAX_ROWS_FOR_LLM
    min_row_count: int = DEFAULT_MIN_ROW_COUNT
    max_null_rate: float = DEFAULT_MAX_NULL_RATE
    max_empty_rate: float = DEFAULT_MAX_EMPTY_RATE
    max_duplicate_rate: float = DEFAULT_MAX_DUPLICATE_RATE
    max_data_age_days: Optional[int] = DEFAULT_MAX_DATA_AGE_DAYS
    reveal_sql: bool = REVEAL_SQL
    use_llm_final_summary: bool = USE_LLM_FINAL_SUMMARY
    enable_agent_hub_artifacts: bool = ENABLE_AGENT_HUB_ARTIFACTS
    enable_png_charts: bool = ENABLE_PNG_CHARTS
    max_chart_rows: int = MAX_CHART_ROWS

    def __post_init__(self) -> None:
        if self.max_data_age_days is not None:
            try:
                self.max_data_age_days = int(self.max_data_age_days)
            except Exception:
                self.max_data_age_days = None

def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "oui"}


def _parse_int(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except Exception:
        return default


def _parse_optional_int(value: Any, default: Optional[int]) -> Optional[int]:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except Exception:
        return default


def _parse_float(value: Any, default: float) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _parse_list(value: Any, default: Optional[List[str]] = None) -> List[str]:
    if default is None:
        default = []
    if value is None or value == "":
        return list(default)
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, tuple):
        return [str(v).strip() for v in value if str(v).strip()]
    raw = str(value).strip()
    if not raw:
        return list(default)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if str(v).strip()]
    except Exception:
        pass
    return [part.strip() for part in raw.split(",") if part.strip()]


def build_config(overrides: Optional[Dict[str, Any]] = None) -> AgentConfig:
    """Build the agent configuration from constants defined in this file only.

    `overrides` is accepted only to keep the Dataiku BaseLLM signature compatible,
    but it is deliberately ignored. This prevents hidden configuration coming from
    environment variables, project variables, or plugin parameters.
    """
    return AgentConfig()

def extract_last_user_prompt(query: Dict[str, Any]) -> str:
    messages = query.get("messages") or []
    for message in reversed(messages):
        if message.get("role") in {"user", "human"}:
            return str(message.get("content", ""))
    if messages:
        return str(messages[-1].get("content", ""))
    return ""


def serialize_json(value: Any, max_chars: int = 16000) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str, indent=2)
    if len(text) > max_chars:
        return text[:max_chars] + "\n... [tronqué]"
    return text


def normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        # pandas / numpy scalars
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass
    try:
        if hasattr(value, "isoformat"):
            return value.isoformat()
    except Exception:
        pass
    try:
        if str(value) == "nan":
            return None
    except Exception:
        pass
    return value


def df_to_records(df: Any, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    if max_rows is not None:
        df = df.head(max_rows)
    records: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        records.append({str(k): normalize_scalar(v) for k, v in row.items()})
    return records


def get_case_insensitive(row: Dict[str, Any], key: str, default: Any = None) -> Any:
    if key in row:
        return row[key]
    key_lower = key.lower()
    for k, v in row.items():
        if str(k).lower() == key_lower:
            return v
    return default


# -----------------------------------------------------------------------------
# SQL safety and quoting
# -----------------------------------------------------------------------------

class UnsafeSQLError(ValueError):
    pass


IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
FORBIDDEN_SQL_WORDS = {
    "alter", "analyze", "compute", "create", "delete", "drop", "grant", "insert",
    "invalidate", "load", "merge", "msck", "refresh", "rename", "replace", "revoke",
    "set", "truncate", "update", "upsert", "use",
}
ALLOWED_FIRST_TOKENS = {"select", "with", "show", "describe", "desc"}


def quote_identifier(name: str) -> str:
    """Quote an Impala identifier with backticks.

    The identifier can contain spaces or accents when it comes from DSS/Impala
    metadata, but it must not contain characters that can break out of quoting.
    """
    name = str(name).strip()
    if not name or any(ch in name for ch in ["`", ";", "\x00", "\n", "\r"]):
        raise UnsafeSQLError(f"Identifiant non autorisé: {name!r}")
    return f"`{name}`"


def safe_table_reference(table: str, database: Optional[str] = None) -> str:
    table = str(table or "").strip()
    database = str(database or "").strip()
    if not table:
        raise ValueError("Aucune table Impala n'est configurée.")
    if database:
        return f"{quote_identifier(database)}.{quote_identifier(table)}"
    parts = [p.strip() for p in table.split(".") if p.strip()]
    if not parts or len(parts) > 3:
        raise UnsafeSQLError(f"Référence de table non autorisée: {table!r}")
    return ".".join(quote_identifier(part) for part in parts)


def sql_strip_markdown(sql: str) -> str:
    sql = str(sql or "").strip()
    sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql)
    return sql.strip()


def remove_sql_comments(sql: str) -> str:
    sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql


def ensure_read_only_sql(sql: str) -> str:
    cleaned = remove_sql_comments(sql_strip_markdown(sql)).strip()
    cleaned = cleaned.rstrip(";").strip()
    if not cleaned:
        raise UnsafeSQLError("La requête SQL est vide.")
    if ";" in cleaned:
        raise UnsafeSQLError("Une seule instruction SQL est autorisée.")
    first_match = re.match(r"^\s*([A-Za-z]+)", cleaned)
    first = first_match.group(1).lower() if first_match else ""
    if first not in ALLOWED_FIRST_TOKENS:
        raise UnsafeSQLError("Seules les requêtes SELECT/WITH/SHOW/DESCRIBE sont autorisées.")
    lower = cleaned.lower()
    for word in FORBIDDEN_SQL_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", lower):
            raise UnsafeSQLError(f"Mot-clé SQL interdit dans ce contexte: {word.upper()}")
    return cleaned


def add_limit_if_needed(sql: str, max_rows: int) -> str:
    cleaned = ensure_read_only_sql(sql)
    first = re.match(r"^\s*([A-Za-z]+)", cleaned).group(1).lower()
    if first not in {"select", "with"}:
        return cleaned
    if re.search(r"\blimit\s+\d+\s*$", cleaned, flags=re.IGNORECASE):
        return cleaned
    return f"{cleaned}\nLIMIT {int(max_rows)}"


# -----------------------------------------------------------------------------
# SQLExecutor2 access layer for Impala-backed SQL connections
# -----------------------------------------------------------------------------

class SQLExecutor2Access:
    """Access layer that deliberately uses dataiku.SQLExecutor2(connection=...) only.

    This standalone version intentionally avoids Dataiku datasets. The agent needs:
    - cfg.sql_connection: DSS SQL connection name, pointing to Impala
    - cfg.sql_table: physical table name
    - cfg.sql_database: optional database/schema used to qualify the table reference
    """

    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.executor: Optional[SQLExecutor2] = None
        self.table_ref = ""
        self.connection_summary = ""
        self._connect()

    def _connect(self) -> None:
        if not self.cfg.sql_connection:
            raise ValueError(
                "SQL_CONNECTION est obligatoire. Renseignez la constante SQL_CONNECTION "
                "dans la section CONFIGURATION A RENSEIGNER DANS CE FICHIER."
            )
        if not self.cfg.sql_table:
            raise ValueError(
                "SQL_TABLE est obligatoire. Renseignez la constante SQL_TABLE "
                "dans la section CONFIGURATION A RENSEIGNER DANS CE FICHIER."
            )

        self.executor = SQLExecutor2(connection=self.cfg.sql_connection)
        self.table_ref = safe_table_reference(self.cfg.sql_table, self.cfg.sql_database or None)
        self.connection_summary = (
            f"executor=SQLExecutor2, connection={self.cfg.sql_connection}, table={self.table_ref}"
        )

    def query_df(self, sql: str) -> Any:
        logger.info("Running SQL through SQLExecutor2: %s", sql)
        if self.executor is None:
            raise ValueError("SQLExecutor2 n'est pas initialisé.")
        return self.executor.query_to_df(sql)

    def describe_columns(self) -> List[Dict[str, Any]]:
        try:
            df = self.query_df(f"DESCRIBE {self.table_ref}")
            records = df_to_records(df)
            columns: List[Dict[str, Any]] = []
            for rec in records:
                values = list(rec.values())
                name = rec.get("name") or rec.get("col_name") or rec.get("column") or (values[0] if values else None)
                dtype = rec.get("type") or rec.get("data_type") or rec.get("datatype") or (values[1] if len(values) > 1 else "")
                comment = rec.get("comment") or (values[2] if len(values) > 2 else "")
                if name is None:
                    continue
                name = str(name).strip()
                if not name or name.startswith("#") or name.lower() in {"partition", "partition information"}:
                    continue
                if name.lower() == "col_name":
                    continue
                columns.append({"name": name, "type": str(dtype or "").strip(), "comment": str(comment or "")})
            if columns:
                return columns
        except Exception as exc:
            logger.warning("DESCRIBE through SQLExecutor2 failed: %s", exc)

        raise ValueError("Impossible de lire le schéma avec SQLExecutor2 via la connexion configurée.")


# -----------------------------------------------------------------------------
# Data-quality SQL generation and rule evaluation
# -----------------------------------------------------------------------------

NUMERIC_TYPES = ("tinyint", "smallint", "int", "bigint", "float", "double", "decimal")
STRING_TYPES = ("string", "varchar", "char")
DATE_TYPES = ("timestamp", "date")
BOOLEAN_TYPES = ("boolean", "bool")


def type_family(dtype: str) -> str:
    d = str(dtype or "").lower()
    if any(t in d for t in NUMERIC_TYPES):
        return "numeric"
    if any(t in d for t in STRING_TYPES):
        return "string"
    if any(t in d for t in DATE_TYPES):
        return "date"
    if any(t in d for t in BOOLEAN_TYPES):
        return "boolean"
    return "other"


def alias_for_column(col: str) -> str:
    alias = re.sub(r"[^A-Za-z0-9_]", "_", col).strip("_").lower()
    return alias or "col"


def build_profile_query(table_ref: str, columns: List[Dict[str, Any]], max_columns: int) -> Tuple[str, Dict[str, str]]:
    selected = columns[:max_columns]
    alias_to_col: Dict[str, str] = {}
    exprs = ["COUNT(*) AS `__row_count`"]
    used_aliases: set[str] = set()

    for col in selected:
        name = str(col["name"])
        q = quote_identifier(name)
        base = alias_for_column(name)
        # Keep aliases unique when column names normalize to the same token.
        alias = base
        i = 2
        while alias in used_aliases:
            alias = f"{base}_{i}"
            i += 1
        used_aliases.add(alias)
        alias_to_col[alias] = name
        family = type_family(str(col.get("type", "")))

        exprs.append(f"SUM(CASE WHEN {q} IS NULL THEN 1 ELSE 0 END) AS `{alias}__nulls`")
        exprs.append(f"NDV({q}) AS `{alias}__approx_distinct`")

        if family == "string":
            exprs.append(
                f"SUM(CASE WHEN {q} IS NOT NULL AND LENGTH(TRIM(CAST({q} AS STRING))) = 0 "
                f"THEN 1 ELSE 0 END) AS `{alias}__empty_strings`"
            )
            exprs.append(f"MIN(LENGTH(CAST({q} AS STRING))) AS `{alias}__min_length`")
            exprs.append(f"MAX(LENGTH(CAST({q} AS STRING))) AS `{alias}__max_length`")
        elif family == "numeric":
            exprs.append(f"MIN({q}) AS `{alias}__min`")
            exprs.append(f"MAX({q}) AS `{alias}__max`")
            exprs.append(f"AVG(CAST({q} AS DOUBLE)) AS `{alias}__avg`")
        elif family == "date":
            exprs.append(f"MIN({q}) AS `{alias}__min`")
            exprs.append(f"MAX({q}) AS `{alias}__max`")

    sql = "SELECT\n  " + ",\n  ".join(exprs) + f"\nFROM {table_ref}"
    return sql, alias_to_col


def profile_table(access: SQLExecutor2Access, columns: List[Dict[str, Any]], cfg: AgentConfig, requested_columns: Optional[List[str]]) -> Dict[str, Any]:
    available = {str(c["name"]).lower(): c for c in columns}
    if requested_columns:
        selected = [available[c.lower()] for c in requested_columns if c.lower() in available]
        if not selected:
            selected = columns
    else:
        selected = columns

    selected = selected[: cfg.max_profile_columns]
    sql, alias_map = build_profile_query(access.table_ref, selected, cfg.max_profile_columns)
    df = access.query_df(sql)
    records = df_to_records(df, max_rows=1)
    row = records[0] if records else {}
    row_count = int(get_case_insensitive(row, "__row_count", 0) or 0)

    by_col_metadata = {str(c["name"]): c for c in selected}
    profiles: List[Dict[str, Any]] = []

    for alias, col_name in alias_map.items():
        meta = by_col_metadata[col_name]
        null_count = int(get_case_insensitive(row, f"{alias}__nulls", 0) or 0)
        empty_strings = get_case_insensitive(row, f"{alias}__empty_strings")
        if empty_strings is not None:
            empty_strings = int(empty_strings or 0)
        approx_distinct = get_case_insensitive(row, f"{alias}__approx_distinct")
        try:
            approx_distinct = int(approx_distinct or 0)
        except Exception:
            approx_distinct = normalize_scalar(approx_distinct)
        profile: Dict[str, Any] = {
            "column": col_name,
            "type": meta.get("type", ""),
            "family": type_family(str(meta.get("type", ""))),
            "null_count": null_count,
            "null_rate": round(null_count / row_count, 6) if row_count else None,
            "approx_distinct": approx_distinct,
        }
        if empty_strings is not None:
            profile["empty_string_count"] = empty_strings
            profile["empty_string_rate"] = round(empty_strings / row_count, 6) if row_count else None
        for metric in ("min", "max", "avg", "min_length", "max_length"):
            val = get_case_insensitive(row, f"{alias}__{metric}")
            if val is not None:
                profile[metric] = normalize_scalar(val)
        profiles.append(profile)

    return {
        "table": access.table_ref,
        "connection": access.connection_summary,
        "row_count": row_count,
        "columns_total": len(columns),
        "columns_profiled": len(profiles),
        "profile_limit_reached": len(columns) > len(profiles),
        "columns": profiles,
        "profile_sql": sql,
    }


def duplicate_key_check(access: SQLExecutor2Access, columns: List[Dict[str, Any]], cfg: AgentConfig) -> Optional[Dict[str, Any]]:
    pk = [c for c in cfg.primary_key_columns if c]
    if not pk:
        return None
    available = {str(c["name"]).lower() for c in columns}
    missing = [c for c in pk if c.lower() not in available]
    if missing:
        return {"status": "ERROR", "message": f"Colonnes de clé primaire absentes: {', '.join(missing)}"}
    key_exprs = ", ".join(quote_identifier(c) for c in pk)
    null_conditions = " OR ".join(f"{quote_identifier(c)} IS NULL" for c in pk)
    sql = f"""
SELECT
  COUNT(*) AS `duplicate_key_groups`,
  COALESCE(SUM(cnt - 1), 0) AS `duplicate_rows`,
  COALESCE(MAX(cnt), 0) AS `max_duplicate_group_size`
FROM (
  SELECT {key_exprs}, COUNT(*) AS cnt
  FROM {access.table_ref}
  GROUP BY {key_exprs}
  HAVING COUNT(*) > 1
) d
""".strip()
    null_sql = f"""
SELECT
  SUM(CASE WHEN {null_conditions} THEN 1 ELSE 0 END) AS `null_key_rows`,
  COUNT(*) AS `row_count`
FROM {access.table_ref}
""".strip()
    dup_row = df_to_records(access.query_df(sql), max_rows=1)[0]
    null_row = df_to_records(access.query_df(null_sql), max_rows=1)[0]
    row_count = int(get_case_insensitive(null_row, "row_count", 0) or 0)
    duplicate_rows = int(get_case_insensitive(dup_row, "duplicate_rows", 0) or 0)
    null_key_rows = int(get_case_insensitive(null_row, "null_key_rows", 0) or 0)
    return {
        "primary_key_columns": pk,
        "duplicate_key_groups": int(get_case_insensitive(dup_row, "duplicate_key_groups", 0) or 0),
        "duplicate_rows": duplicate_rows,
        "duplicate_rate": round(duplicate_rows / row_count, 6) if row_count else None,
        "max_duplicate_group_size": int(get_case_insensitive(dup_row, "max_duplicate_group_size", 0) or 0),
        "null_key_rows": null_key_rows,
        "null_key_rate": round(null_key_rows / row_count, 6) if row_count else None,
        "sql": sql,
        "null_key_sql": null_sql,
    }


def parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def evaluate_rules(profile: Dict[str, Any], columns: List[Dict[str, Any]], duplicate_check: Optional[Dict[str, Any]], cfg: AgentConfig, plan: Dict[str, Any]) -> Dict[str, Any]:
    thresholds = plan.get("thresholds") or {}
    max_null_rate = _parse_float(thresholds.get("max_null_rate"), cfg.max_null_rate)
    max_empty_rate = _parse_float(thresholds.get("max_empty_rate"), cfg.max_empty_rate)
    min_row_count = _parse_int(thresholds.get("min_row_count"), cfg.min_row_count)
    max_duplicate_rate = _parse_float(thresholds.get("max_duplicate_rate"), cfg.max_duplicate_rate)
    max_data_age_days = _parse_optional_int(thresholds.get("max_data_age_days"), cfg.max_data_age_days)

    results: List[Dict[str, Any]] = []

    def add_rule(name: str, status: str, details: str, metric: Any = None, threshold: Any = None) -> None:
        results.append({
            "rule": name,
            "status": status,
            "metric": normalize_scalar(metric),
            "threshold": normalize_scalar(threshold),
            "details": details,
        })

    row_count = int(profile.get("row_count") or 0)
    add_rule(
        "row_count_minimum",
        "OK" if row_count >= min_row_count else "ERROR",
        f"Nombre de lignes = {row_count}",
        row_count,
        f">= {min_row_count}",
    )

    available_columns = {str(c.get("name", "")).lower() for c in columns}
    for required in cfg.required_columns:
        add_rule(
            f"required_column:{required}",
            "OK" if required.lower() in available_columns else "ERROR",
            f"Colonne obligatoire {required!r}",
            required,
            "présente",
        )

    for col in profile.get("columns", []):
        col_name = col["column"]
        null_rate = col.get("null_rate")
        if null_rate is not None:
            add_rule(
                f"null_rate:{col_name}",
                "OK" if null_rate <= max_null_rate else "ERROR",
                f"Taux de NULL pour {col_name}",
                null_rate,
                f"<= {max_null_rate}",
            )
        empty_rate = col.get("empty_string_rate")
        if empty_rate is not None:
            add_rule(
                f"empty_string_rate:{col_name}",
                "OK" if empty_rate <= max_empty_rate else "WARNING",
                f"Taux de chaînes vides pour {col_name}",
                empty_rate,
                f"<= {max_empty_rate}",
            )

        if max_data_age_days is not None and col_name in cfg.date_columns and col.get("max") is not None:
            dt = parse_datetime(col.get("max"))
            if dt is not None:
                age_days = (datetime.now(timezone.utc) - dt).days
                add_rule(
                    f"freshness:{col_name}",
                    "OK" if age_days <= max_data_age_days else "WARNING",
                    f"Fraîcheur selon MAX({col_name})",
                    f"{age_days} jours",
                    f"<= {max_data_age_days} jours",
                )

    if duplicate_check:
        if duplicate_check.get("status") == "ERROR":
            add_rule("duplicate_key", "ERROR", str(duplicate_check.get("message")))
        else:
            dup_rate = duplicate_check.get("duplicate_rate") or 0
            null_key_rate = duplicate_check.get("null_key_rate") or 0
            add_rule(
                "duplicate_key_rate",
                "OK" if dup_rate <= max_duplicate_rate else "ERROR",
                f"Doublons sur clé: {', '.join(duplicate_check.get('primary_key_columns', []))}",
                dup_rate,
                f"<= {max_duplicate_rate}",
            )
            add_rule(
                "primary_key_null_rate",
                "OK" if null_key_rate == 0 else "ERROR",
                "Lignes avec clé primaire partiellement ou totalement NULL",
                null_key_rate,
                "= 0",
            )

    if any(r["status"] == "ERROR" for r in results):
        overall = "ERROR"
    elif any(r["status"] == "WARNING" for r in results):
        overall = "WARNING"
    else:
        overall = "OK"

    return {
        "overall_status": overall,
        "thresholds": {
            "min_row_count": min_row_count,
            "max_null_rate": max_null_rate,
            "max_empty_rate": max_empty_rate,
            "max_duplicate_rate": max_duplicate_rate,
            "max_data_age_days": max_data_age_days,
        },
        "rules": results,
    }


# -----------------------------------------------------------------------------
# LangGraph state and graph construction
# -----------------------------------------------------------------------------

class DQState(TypedDict, total=False):
    user_query: str
    plan: Dict[str, Any]
    table_ref: str
    connection: str
    columns: List[Dict[str, Any]]
    profile: Dict[str, Any]
    duplicate_check: Optional[Dict[str, Any]]
    rule_evaluation: Dict[str, Any]
    sql_query: str
    query_result: List[Dict[str, Any]]
    response: str
    artifacts: List[Dict[str, Any]]
    error: str


def parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = sql_strip_markdown(text)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}


def fallback_plan(user_query: str) -> Dict[str, Any]:
    text = user_query.lower()
    if any(token in text for token in ["schéma", "schema", "colonnes", "columns", "describe"]):
        return {"action": "schema", "columns": [], "thresholds": {}, "custom_sql": "", "intent": "inspection du schéma"}
    if any(token in text for token in ["sql", "select", "requête", "query", "combien", "count", "top", "moyenne"]):
        return {"action": "sql", "columns": [], "thresholds": {}, "custom_sql": "", "intent": "question SQL ad hoc"}
    if any(token in text for token in ["profil", "profile", "stat", "distribution", "null", "distinct"]):
        return {"action": "profile", "columns": [], "thresholds": {}, "custom_sql": "", "intent": "profiling"}
    if any(token in text for token in ["aide", "help", "capable", "peux-tu"]):
        return {"action": "help", "columns": [], "thresholds": {}, "custom_sql": "", "intent": "aide"}
    return {"action": "checks", "columns": [], "thresholds": {}, "custom_sql": "", "intent": "contrôles qualité"}


def normalize_plan(plan: Dict[str, Any], user_query: str, cfg: AgentConfig) -> Dict[str, Any]:
    if not plan:
        plan = fallback_plan(user_query)
    action = str(plan.get("action") or "checks").lower().strip()
    if action not in {"schema", "profile", "checks", "sql", "help"}:
        action = fallback_plan(user_query).get("action", "checks")
    plan["action"] = action
    plan["columns"] = _parse_list(plan.get("columns"), [])
    thresholds = plan.get("thresholds") if isinstance(plan.get("thresholds"), dict) else {}
    cleaned_thresholds: Dict[str, Any] = {}
    for key in ("max_null_rate", "max_empty_rate", "min_row_count", "max_duplicate_rate", "max_data_age_days"):
        if key in thresholds and thresholds[key] not in (None, ""):
            cleaned_thresholds[key] = thresholds[key]
    plan["thresholds"] = cleaned_thresholds
    custom_sql = str(plan.get("custom_sql") or "").strip()
    if custom_sql:
        try:
            plan["custom_sql"] = add_limit_if_needed(custom_sql, cfg.max_return_rows)
            plan["action"] = "sql"
        except Exception as exc:
            plan["custom_sql"] = ""
            plan["sql_error"] = str(exc)
            if action == "sql":
                plan["action"] = "checks"
    return plan


def make_planning_prompt(cfg: AgentConfig) -> ChatPromptTemplate:
    system_prompt = (
        "Tu es un planificateur pour un agent de data quality connecté à Impala via SQLExecutor2. "
        "Retourne uniquement un objet JSON valide, sans Markdown.\n\n"
        "Schéma JSON attendu:\n"
        "{{\n"
        "  \"action\": \"schema|profile|checks|sql|help\",\n"
        "  \"columns\": [\"colonne_optionnelle\"],\n"
        "  \"thresholds\": {{\"max_null_rate\": 0.2, \"max_empty_rate\": 0.05, \"min_row_count\": 1, \"max_duplicate_rate\": 0}},\n"
        "  \"custom_sql\": \"SELECT ... ou chaîne vide\",\n"
        "  \"intent\": \"résumé court de la demande\"\n"
        "}}\n\n"
        "Règles de sécurité:\n"
        "- custom_sql doit être vide sauf si l'utilisateur demande explicitement un résultat SQL ad hoc.\n"
        "- custom_sql doit être lecture seule: SELECT/WITH/SHOW/DESCRIBE uniquement.\n"
        "- Ne propose jamais INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, REFRESH, INVALIDATE, COMPUTE STATS, SET ou USE.\n"
        "- Pour un audit qualité général, choisis action=checks.\n"
        "- Pour un simple profilage statistique, choisis action=profile.\n"
        "- Pour une liste de colonnes ou le schéma, choisis action=schema."
    )
    human_prompt = (
        "Configuration disponible:\n"
        f"- sql_connection={cfg.sql_connection or ''}\n"
        f"- sql_table={cfg.sql_table or ''}\n"
        f"- primary_key_columns={cfg.primary_key_columns}\n"
        f"- required_columns={cfg.required_columns}\n"
        f"- date_columns={cfg.date_columns}\n\n"
        "Demande utilisateur:\n{user_query}"
    )
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])

def _fmt_int(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{int(value):,}".replace(",", " ")
    except Exception:
        return str(value)


def _fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return str(value)


def _fmt_metric(value: Any) -> str:
    if isinstance(value, float):
        if 0 <= value <= 1:
            return _fmt_pct(value)
        return _fmt_float(value)
    return str(value)



# -----------------------------------------------------------------------------
# Agent Hub artifacts and chart helpers
# -----------------------------------------------------------------------------

def _safe_artifact_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value or "").strip())
    return cleaned.strip("_") or "artifact"


def _records_from_rows(columns: List[str], rows: List[List[Any]]) -> Dict[str, Any]:
    return {
        "columns": [str(c) for c in columns],
        "data": [[normalize_scalar(cell) for cell in row] for row in rows],
    }


def make_records_artifact(
    artifact_id: str,
    name: str,
    description: str,
    columns: List[str],
    rows: List[List[Any]],
) -> Dict[str, Any]:
    """Build a chartable Agent Hub artifact.

    Agent Hub uses `RECORDS` artifacts as chartable in-chat data. The payload keeps
    the records both on the top-level artifact and in a part to be compatible with
    the generic Dataiku agent artifact structure.
    """
    records = _records_from_rows(columns, rows)
    return {
        "id": _safe_artifact_id(artifact_id),
        "type": "RECORDS",
        "name": name,
        "description": description,
        "parts": [
            {
                "type": "RECORDS",
                "index": 0,
                "records": records,
            }
        ],
        # Agent Hub documentation examples sometimes show `records` directly on
        # the returned object. Keeping it here improves compatibility across Hub
        # and LLM Mesh consumers.
        "records": records,
    }


def make_data_inline_artifact(
    artifact_id: str,
    name: str,
    description: str,
    mime_type: str,
    data_base64: str,
    artifact_type: str = "DATA_INLINE",
) -> Dict[str, Any]:
    return {
        "id": _safe_artifact_id(artifact_id),
        "type": artifact_type,
        "name": name,
        "description": description,
        "parts": [
            {
                "type": "DATA_INLINE",
                "index": 0,
                "mimeType": mime_type,
                "dataBase64": data_base64,
            }
        ],
    }


def make_generated_sql_artifact(sql: str) -> Dict[str, Any]:
    return {
        "id": "dq_generated_sql_query",
        "type": "GENERATED_SQL_QUERY",
        "name": "Requête SQL exécutée",
        "description": "Requête lecture seule exécutée par l’agent via SQLExecutor2.",
        "parts": [
            {
                "type": "GENERATED_SQL_QUERY",
                "index": 0,
                "performedQuery": sql,
            }
        ],
    }


def _try_make_bar_png_base64(title: str, labels: List[str], values: List[float], y_label: str) -> Optional[str]:
    """Return a base64 PNG bar chart, or None if matplotlib is unavailable."""
    clean_pairs = [(str(label), float(value or 0)) for label, value in zip(labels, values)]
    if not clean_pairs:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.info("matplotlib unavailable; skipping PNG chart artifact: %s", exc)
        return None

    width = max(7.0, min(14.0, 0.45 * len(clean_pairs) + 4.0))
    fig, ax = plt.subplots(figsize=(width, 4.8))
    x_positions = list(range(len(clean_pairs)))
    ax.bar(x_positions, [v for _, v in clean_pairs])
    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for label, _ in clean_pairs], rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _rows_to_records_artifact_from_dicts(
    artifact_id: str,
    name: str,
    description: str,
    rows: List[Dict[str, Any]],
    max_rows: int,
) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    columns: List[str] = []
    for row in rows[:max_rows]:
        for key in row.keys():
            key_s = str(key)
            if key_s not in columns:
                columns.append(key_s)
    if not columns:
        return None
    matrix = [[normalize_scalar(row.get(col)) for col in columns] for row in rows[:max_rows]]
    return make_records_artifact(artifact_id, name, description, columns, matrix)


def _top_profile_rows(profile: Dict[str, Any], metric: str, cfg: AgentConfig) -> List[Dict[str, Any]]:
    cols = [c for c in profile.get("columns", []) if c.get(metric) is not None]
    cols = sorted(cols, key=lambda c: float(c.get(metric) or 0), reverse=True)
    return cols[: max(1, cfg.max_chart_rows)]


def build_agent_hub_artifacts(state: DQState, cfg: AgentConfig) -> List[Dict[str, Any]]:
    """Create Dataiku Agent Hub artifacts for charts and downloads.

    The most important artifacts are `RECORDS`, because Agent Hub treats them as
    chartable data in the chat interface. Optional PNG artifacts are included as
    `DATA_INLINE` images for clients that preview inline image files.
    """
    if not cfg.enable_agent_hub_artifacts or state.get("error"):
        return []

    artifacts: List[Dict[str, Any]] = []

    # SQL ad-hoc answers: return the query result as chartable records.
    if state.get("query_result") is not None:
        rec_artifact = _rows_to_records_artifact_from_dicts(
            "dq_sql_result_records",
            "Résultat SQL — données chartables",
            "Résultat de la requête lecture seule, exploitable par l’Agent Hub pour générer un graphique.",
            state.get("query_result", []) or [],
            cfg.max_chart_rows,
        )
        if rec_artifact:
            artifacts.append(rec_artifact)
        if cfg.reveal_sql and state.get("sql_query"):
            artifacts.append(make_generated_sql_artifact(state["sql_query"]))
        return artifacts

    profile = state.get("profile", {}) or {}
    evaluation = state.get("rule_evaluation", {}) or {}
    rules = evaluation.get("rules", []) if evaluation else []

    # KPI overview.
    score_value = quality_score(evaluation) if evaluation else None
    status_counts = {
        "ERROR": sum(1 for r in rules if r.get("status") == "ERROR"),
        "WARNING": sum(1 for r in rules if r.get("status") == "WARNING"),
        "OK": sum(1 for r in rules if r.get("status") == "OK"),
    }
    kpi_rows = [
        ["row_count", int(profile.get("row_count") or 0)],
        ["columns_profiled", int(profile.get("columns_profiled") or 0)],
        ["columns_total", int(profile.get("columns_total") or 0)],
        ["quality_score", score_value if score_value is not None else 0],
        ["rules_error", status_counts["ERROR"]],
        ["rules_warning", status_counts["WARNING"]],
        ["rules_ok", status_counts["OK"]],
    ]
    artifacts.append(make_records_artifact(
        "dq_quality_kpis",
        "KPI qualité — données chartables",
        "Indicateurs globaux du contrôle qualité.",
        ["metric", "value"],
        kpi_rows,
    ))

    # Rule status chart.
    rule_status_rows = [[status, count] for status, count in status_counts.items()]
    artifacts.append(make_records_artifact(
        "dq_rule_status_counts",
        "Statut des règles qualité — données chartables",
        "Nombre de règles OK, WARNING et ERROR.",
        ["status", "count"],
        rule_status_rows,
    ))
    if cfg.enable_png_charts:
        png = _try_make_bar_png_base64(
            "Statut des règles qualité",
            [r[0] for r in rule_status_rows],
            [float(r[1]) for r in rule_status_rows],
            "Nombre de règles",
        )
        if png:
            artifacts.append(make_data_inline_artifact(
                "dq_rule_status_counts_png",
                "statut_regles_qualite.png",
                "Graphique PNG du statut des règles qualité.",
                "image/png",
                png,
                artifact_type="IMAGE",
            ))

    # Null rate chart.
    null_cols = _top_profile_rows(profile, "null_rate", cfg)
    if null_cols:
        rows = [
            [c.get("column"), round(float(c.get("null_rate") or 0) * 100, 4), int(c.get("null_count") or 0), c.get("type", "")]
            for c in null_cols
        ]
        artifacts.append(make_records_artifact(
            "dq_null_rate_by_column",
            "Taux de NULL par colonne — données chartables",
            "Colonnes triées par taux de NULL décroissant.",
            ["column", "null_rate_pct", "null_count", "data_type"],
            rows,
        ))
        if cfg.enable_png_charts:
            png = _try_make_bar_png_base64(
                "Taux de NULL par colonne",
                [str(r[0]) for r in rows],
                [float(r[1]) for r in rows],
                "% NULL",
            )
            if png:
                artifacts.append(make_data_inline_artifact(
                    "dq_null_rate_by_column_png",
                    "taux_null_par_colonne.png",
                    "Graphique PNG du taux de NULL par colonne.",
                    "image/png",
                    png,
                    artifact_type="IMAGE",
                ))

    # Empty-string rate chart for string columns.
    empty_cols = _top_profile_rows(profile, "empty_string_rate", cfg)
    if empty_cols:
        rows = [
            [c.get("column"), round(float(c.get("empty_string_rate") or 0) * 100, 4), int(c.get("empty_string_count") or 0), c.get("type", "")]
            for c in empty_cols
        ]
        artifacts.append(make_records_artifact(
            "dq_empty_string_rate_by_column",
            "Taux de chaînes vides par colonne — données chartables",
            "Colonnes texte triées par taux de chaînes vides décroissant.",
            ["column", "empty_string_rate_pct", "empty_string_count", "data_type"],
            rows,
        ))
        if cfg.enable_png_charts:
            png = _try_make_bar_png_base64(
                "Taux de chaînes vides par colonne",
                [str(r[0]) for r in rows],
                [float(r[1]) for r in rows],
                "% chaînes vides",
            )
            if png:
                artifacts.append(make_data_inline_artifact(
                    "dq_empty_string_rate_by_column_png",
                    "taux_chaines_vides_par_colonne.png",
                    "Graphique PNG du taux de chaînes vides par colonne.",
                    "image/png",
                    png,
                    artifact_type="IMAGE",
                ))

    # Approximate cardinality chart.
    distinct_cols = _top_profile_rows(profile, "approx_distinct", cfg)
    if distinct_cols:
        rows = [
            [c.get("column"), int(c.get("approx_distinct") or 0), c.get("type", "")]
            for c in distinct_cols
        ]
        artifacts.append(make_records_artifact(
            "dq_approx_distinct_by_column",
            "Cardinalité approximative par colonne — données chartables",
            "Colonnes triées par nombre approximatif de valeurs distinctes.",
            ["column", "approx_distinct", "data_type"],
            rows,
        ))
        if cfg.enable_png_charts:
            png = _try_make_bar_png_base64(
                "Cardinalité approximative par colonne",
                [str(r[0]) for r in rows],
                [float(r[1]) for r in rows],
                "Valeurs distinctes approx.",
            )
            if png:
                artifacts.append(make_data_inline_artifact(
                    "dq_approx_distinct_by_column_png",
                    "cardinalite_approximative_par_colonne.png",
                    "Graphique PNG de la cardinalité approximative par colonne.",
                    "image/png",
                    png,
                    artifact_type="IMAGE",
                ))

    # Duplicate-key summary if a primary key is configured.
    duplicate_check = state.get("duplicate_check") or {}
    if duplicate_check and duplicate_check.get("status") != "ERROR":
        dup_rows = [
            ["duplicate_key_groups", int(duplicate_check.get("duplicate_key_groups") or 0)],
            ["duplicate_rows", int(duplicate_check.get("duplicate_rows") or 0)],
            ["duplicate_rate_pct", round(float(duplicate_check.get("duplicate_rate") or 0) * 100, 4)],
            ["max_duplicate_group_size", int(duplicate_check.get("max_duplicate_group_size") or 0)],
            ["null_key_rows", int(duplicate_check.get("null_key_rows") or 0)],
            ["null_key_rate_pct", round(float(duplicate_check.get("null_key_rate") or 0) * 100, 4)],
        ]
        artifacts.append(make_records_artifact(
            "dq_duplicate_key_summary",
            "Doublons de clé — données chartables",
            "Résumé du contrôle d’unicité sur la clé configurée.",
            ["metric", "value"],
            dup_rows,
        ))

    if cfg.reveal_sql and profile.get("profile_sql"):
        artifacts.append(make_generated_sql_artifact(profile["profile_sql"]))

    return artifacts

def markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    def esc(value: Any) -> str:
        text = "" if value is None else str(value)
        text = text.replace("|", "\\|").replace("\n", "<br>")
        return text

    if not rows:
        return "_Aucune donnée._"
    header = "| " + " | ".join(esc(h) for h in headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(esc(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def status_icon(status: str) -> str:
    return {"OK": "✅", "WARNING": "⚠️", "ERROR": "❌"}.get(str(status or "").upper(), "ℹ️")


def quality_score(evaluation: Dict[str, Any]) -> int:
    rules = evaluation.get("rules", []) if evaluation else []
    if not rules:
        return 0
    errors = sum(1 for r in rules if r.get("status") == "ERROR")
    warnings = sum(1 for r in rules if r.get("status") == "WARNING")
    # Simple deterministic score for communication, not a statistical guarantee.
    return max(0, min(100, 100 - errors * 18 - warnings * 7))


def compact_rules(rules: List[Dict[str, Any]], status: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    filtered = [r for r in rules if status is None or r.get("status") == status]
    return filtered if limit is None else filtered[:limit]


def profile_rows_for_report(profile: Dict[str, Any], cfg: AgentConfig) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for col in profile.get("columns", [])[: cfg.max_report_columns]:
        rows.append([
            f"`{col.get('column')}`",
            col.get("type", ""),
            col.get("family", ""),
            _fmt_int(col.get("null_count")),
            _fmt_pct(col.get("null_rate")),
            _fmt_int(col.get("empty_string_count")) if "empty_string_count" in col else "n/a",
            _fmt_pct(col.get("empty_string_rate")) if "empty_string_rate" in col else "n/a",
            _fmt_int(col.get("approx_distinct")),
            col.get("min", col.get("min_length", "n/a")),
            col.get("max", col.get("max_length", "n/a")),
            _fmt_float(col.get("avg")) if col.get("avg") is not None else "n/a",
        ])
    return rows


def top_column_findings(profile: Dict[str, Any]) -> List[str]:
    cols = profile.get("columns", [])
    findings: List[str] = []
    row_count = int(profile.get("row_count") or 0)

    high_nulls = [c for c in cols if (c.get("null_rate") or 0) > 0]
    high_nulls = sorted(high_nulls, key=lambda c: c.get("null_rate") or 0, reverse=True)[:5]
    for c in high_nulls:
        findings.append(
            f"`{c.get('column')}` contient {_fmt_pct(c.get('null_rate'))} de NULL "
            f"({_fmt_int(c.get('null_count'))} lignes)."
        )

    high_empty = [c for c in cols if (c.get("empty_string_rate") or 0) > 0]
    high_empty = sorted(high_empty, key=lambda c: c.get("empty_string_rate") or 0, reverse=True)[:5]
    for c in high_empty:
        findings.append(
            f"`{c.get('column')}` contient {_fmt_pct(c.get('empty_string_rate'))} de chaînes vides "
            f"({_fmt_int(c.get('empty_string_count'))} lignes)."
        )

    if row_count:
        constants = [c for c in cols if c.get("approx_distinct") == 1 and c.get("null_count") != row_count]
        for c in constants[:5]:
            findings.append(f"`{c.get('column')}` semble constante sur les lignes non-nulles.")

    return findings


def remediation_recommendations(state: DQState) -> List[str]:
    recommendations: List[str] = []
    evaluation = state.get("rule_evaluation", {})
    rules = evaluation.get("rules", []) if evaluation else []
    errors = compact_rules(rules, "ERROR")
    warnings = compact_rules(rules, "WARNING")

    if any(r.get("rule") == "row_count_minimum" for r in errors):
        recommendations.append("Vérifier l’alimentation amont: la table est vide ou sous le volume minimal attendu.")
    if any(str(r.get("rule", "")).startswith("required_column:") for r in errors):
        recommendations.append("Aligner le contrat de schéma entre producteurs et consommateurs, puis corriger les colonnes obligatoires manquantes.")
    if any(str(r.get("rule", "")).startswith("null_rate:") for r in errors):
        recommendations.append("Identifier les colonnes au-dessus du seuil de NULL, puis arbitrer entre correction source, règle métier de défaut, ou assouplissement documenté du seuil.")
    if any(r.get("rule") == "duplicate_key_rate" for r in errors):
        recommendations.append("Dédupliquer les clés avant consommation analytique et ajouter un contrôle d’unicité en amont.")
    if any(r.get("rule") == "primary_key_null_rate" for r in errors):
        recommendations.append("Rejeter ou corriger les lignes dont la clé primaire est partiellement ou totalement NULL.")
    if any(str(r.get("rule", "")).startswith("empty_string_rate:") for r in warnings):
        recommendations.append("Normaliser les chaînes vides en NULL ou en valeur métier explicite, puis surveiller le taux par source.")
    if any(str(r.get("rule", "")).startswith("freshness:") for r in warnings):
        recommendations.append("Contrôler la fraîcheur des partitions ou du job d’ingestion associé à la colonne date configurée.")

    if not recommendations and not errors and not warnings:
        recommendations.append("Maintenir les contrôles actuels dans un scénario Dataiku planifié et historiser les métriques pour détecter les dérives.")
    elif not recommendations:
        recommendations.append("Prioriser les règles en ERROR, puis décider si les WARNING doivent devenir bloquants selon le SLA métier.")
    return recommendations


def format_schema_report(state: DQState) -> str:
    columns = state.get("columns", [])
    rows = [[f"`{c.get('name')}`", c.get("type", ""), c.get("comment", "")] for c in columns[:300]]
    return "\n".join([
        "# Rapport de schéma",
        "",
        f"**Table** : `{state.get('table_ref', '')}`",
        f"**Connexion** : {state.get('connection', '')}",
        f"**Nombre de colonnes lues** : {_fmt_int(len(columns))}",
        "",
        "## Colonnes",
        markdown_table(["Colonne", "Type", "Commentaire"], rows),
    ])


def format_sql_result_report(state: DQState, cfg: AgentConfig) -> str:
    rows = state.get("query_result", []) or []
    output = [
        "# Rapport SQL lecture seule",
        "",
        f"**Table** : `{state.get('table_ref', '')}`",
        f"**Connexion** : {state.get('connection', '')}",
        f"**Lignes retournées** : {_fmt_int(len(rows))}",
        "**Graphiques Agent Hub** : le résultat est renvoyé en artefact `RECORDS` chartable.",
    ]
    if cfg.reveal_sql and state.get("sql_query"):
        output.extend(["", "## Requête exécutée", f"```sql\n{state['sql_query']}\n```"])
    output.extend(["", "## Résultat", "```json", serialize_json(rows, max_chars=20000), "```"])
    return "\n".join(output)


def format_error_report(state: DQState, cfg: AgentConfig) -> str:
    return "\n".join([
        "# Rapport data quality — erreur",
        "",
        "**Statut** : ❌ ERROR",
        f"**Erreur** : {state.get('error', 'Erreur inconnue')}",
        f"**Table** : `{state.get('table_ref', '')}`" if state.get("table_ref") else "**Table** : non résolue",
        f"**Connexion** : {state.get('connection', '')}" if state.get("connection") else "**Connexion** : non résolue",
        "",
        "## Actions recommandées",
        "- Vérifier `SQL_CONNECTION` + `SQL_TABLE`.",
        "- Confirmer que la connexion DSS pointe bien vers Impala et que l’utilisateur Dataiku a les droits de lecture.",
        "- Tester manuellement `DESCRIBE <table>` puis une requête `SELECT COUNT(*)` dans un SQL Notebook Dataiku.",
    ])


def format_detailed_report(state: DQState, cfg: AgentConfig) -> str:
    if state.get("error"):
        return format_error_report(state, cfg)

    action = state.get("plan", {}).get("action", "checks")
    if action == "help":
        return (
            "# Agent data quality Dataiku\n\n"
            "Je peux produire un rapport détaillé de qualité de données sur une table Impala via `SQLExecutor2` : "
            "schéma, volumétrie, taux de NULL, chaînes vides, cardinalité approximative, min/max, contrôles de clé, "
            "fraîcheur, SQL exécuté et recommandations.\n\n"
            "Exemples : `Fais un audit qualité complet`, `Montre le schéma`, "
            "`Quel est le taux de null par colonne ?`, `Vérifie les doublons sur la clé configurée`."
        )

    if state.get("query_result") is not None:
        return format_sql_result_report(state, cfg)

    if state.get("columns") and not state.get("profile"):
        return format_schema_report(state)

    profile = state.get("profile", {})
    evaluation = state.get("rule_evaluation", {})
    rules = evaluation.get("rules", []) if evaluation else []
    errors = compact_rules(rules, "ERROR")
    warnings = compact_rules(rules, "WARNING")
    ok_rules = compact_rules(rules, "OK")
    status = evaluation.get("overall_status", "UNKNOWN") if evaluation else "PROFILE_ONLY"
    score = quality_score(evaluation) if evaluation else "n/a"
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    output: List[str] = [
        "# Rapport détaillé de data quality",
        "",
        "## 1. Synthèse exécutive",
        "",
        f"**Statut global** : {status_icon(status)} **{status}**",
        f"**Score qualité indicatif** : **{score}/100**" if isinstance(score, int) else "**Score qualité indicatif** : n/a",
        f"**Table auditée** : `{profile.get('table', state.get('table_ref', ''))}`",
        f"**Connexion** : {profile.get('connection', state.get('connection', ''))}",
        f"**Date de génération** : {generated_at}",
        f"**Lignes analysées** : {_fmt_int(profile.get('row_count'))}",
        f"**Colonnes profilées** : {_fmt_int(profile.get('columns_profiled'))} / {_fmt_int(profile.get('columns_total'))}",
        f"**Règles en erreur** : {_fmt_int(len(errors))} | **Warnings** : {_fmt_int(len(warnings))} | **OK** : {_fmt_int(len(ok_rules))}",
        "**Graphiques Agent Hub** : artefacts `RECORDS` chartables générés pour KPI, règles, NULL, chaînes vides, cardinalité et doublons si configurés.",
    ]

    if profile.get("profile_limit_reached"):
        output.append(
            f"⚠️ Le profiling a été limité à {cfg.max_profile_columns} colonnes. "
            "Augmentez `max_profile_columns` pour couvrir toute la table."
        )

    output.extend(["", "## 2. Périmètre et seuils", ""])
    threshold_rows = []
    thresholds = evaluation.get("thresholds", {}) if evaluation else {}
    for key in ["min_row_count", "max_null_rate", "max_empty_rate", "max_duplicate_rate", "max_data_age_days"]:
        threshold_rows.append([key, thresholds.get(key, "n/a")])
    output.append(markdown_table(["Paramètre", "Valeur"], threshold_rows))

    if rules:
        output.extend(["", "## 3. Résultat détaillé des règles", ""])
        rule_rows = [
            [
                f"{status_icon(r.get('status'))} {r.get('status')}",
                r.get("rule"),
                _fmt_metric(r.get("metric")),
                r.get("threshold"),
                r.get("details"),
            ]
            for r in rules[:300]
        ]
        output.append(markdown_table(["Statut", "Règle", "Métrique", "Seuil", "Détail"], rule_rows))

    output.extend(["", "## 4. Anomalies et points d’attention", ""])
    anomaly_lines: List[str] = []
    for r in errors[:20]:
        anomaly_lines.append(f"- ❌ **{r.get('rule')}** — {r.get('details')} ; métrique={_fmt_metric(r.get('metric'))}, seuil={r.get('threshold')}.")
    for r in warnings[:20]:
        anomaly_lines.append(f"- ⚠️ **{r.get('rule')}** — {r.get('details')} ; métrique={_fmt_metric(r.get('metric'))}, seuil={r.get('threshold')}.")
    for finding in top_column_findings(profile):
        anomaly_lines.append(f"- {finding}")
    if not anomaly_lines:
        anomaly_lines.append("- Aucune anomalie détectée avec les seuils configurés.")
    output.extend(anomaly_lines)

    if state.get("duplicate_check"):
        dup = state["duplicate_check"] or {}
        output.extend(["", "## 5. Contrôle de clé / unicité", ""])
        if dup.get("status") == "ERROR":
            output.append(f"❌ {dup.get('message')}")
        else:
            output.append(markdown_table(
                ["Métrique", "Valeur"],
                [
                    ["Colonnes de clé", ", ".join(dup.get("primary_key_columns", []))],
                    ["Groupes de doublons", _fmt_int(dup.get("duplicate_key_groups"))],
                    ["Lignes en doublon", _fmt_int(dup.get("duplicate_rows"))],
                    ["Taux de doublons", _fmt_pct(dup.get("duplicate_rate"))],
                    ["Taille max d’un groupe doublonné", _fmt_int(dup.get("max_duplicate_group_size"))],
                    ["Lignes avec clé NULL", _fmt_int(dup.get("null_key_rows"))],
                    ["Taux de clé NULL", _fmt_pct(dup.get("null_key_rate"))],
                ],
            ))

    output.extend(["", "## 6. Profiling détaillé par colonne", ""])
    output.append(markdown_table(
        ["Colonne", "Type", "Famille", "NULL", "% NULL", "Vides", "% vides", "Distinct approx", "Min / min len", "Max / max len", "Avg"],
        profile_rows_for_report(profile, cfg),
    ))
    if len(profile.get("columns", [])) > cfg.max_report_columns:
        output.append(f"\n_Rapport colonnes tronqué à {cfg.max_report_columns} colonnes._")

    output.extend(["", "## 7. Recommandations", ""])
    for rec in remediation_recommendations(state):
        output.append(f"- {rec}")

    if cfg.reveal_sql:
        output.extend(["", "## 8. Audit trail SQL", ""])
        if profile.get("profile_sql"):
            output.extend(["### Requête de profiling", f"```sql\n{profile['profile_sql']}\n```"])
        dup = state.get("duplicate_check") or {}
        if dup.get("sql"):
            output.extend(["### Requête de détection des doublons", f"```sql\n{dup['sql']}\n```"])
        if dup.get("null_key_sql"):
            output.extend(["### Requête de contrôle des clés NULL", f"```sql\n{dup['null_key_sql']}\n```"])

    output.extend(["", "## 9. Données techniques synthétiques", ""])
    output.extend([
        f"- Action détectée : `{state.get('plan', {}).get('action', '')}`",
        f"- Intention : {state.get('plan', {}).get('intent', '')}",
        f"- Colonnes demandées : {', '.join(state.get('plan', {}).get('columns', []) or []) or 'toutes / non spécifié'}",
        "- Exécuteur : `dataiku.SQLExecutor2` uniquement",
    ])

    return "\n".join(output)


def insert_llm_summary(report: str, summary: str) -> str:
    summary = str(summary or "").strip()
    if not summary:
        return report
    marker = "## 2. Périmètre et seuils"
    insertion = "\n\n### Lecture experte générée par le LLM\n\n" + summary + "\n"
    if marker in report:
        return report.replace(marker, insertion + "\n" + marker, 1)
    return report + insertion


def format_manual_response(state: DQState, cfg: AgentConfig) -> str:
    return format_detailed_report(state, cfg)

def build_graph(cfg: AgentConfig, llm: Any):
    def plan_request(state: DQState) -> DQState:
        user_query = state["user_query"]
        try:
            prompt = make_planning_prompt(cfg).format_messages(user_query=user_query)
            planning_llm = llm.with_config({"tags": ["dq_planner"], "metadata": {"stage": "planning"}})
            raw = planning_llm.invoke(prompt).content
            plan = parse_json_object(raw)
        except Exception as exc:
            logger.warning("Planning LLM failed; using heuristic plan: %s", exc)
            plan = fallback_plan(user_query)
        return {"plan": normalize_plan(plan, user_query, cfg)}

    def inspect_schema(state: DQState) -> DQState:
        try:
            access = SQLExecutor2Access(cfg)
            columns = access.describe_columns()
            return {"columns": columns, "table_ref": access.table_ref, "connection": access.connection_summary}
        except Exception as exc:
            return {"error": f"Erreur pendant la lecture du schéma via SQLExecutor2: {exc}"}

    def run_profile(state: DQState) -> DQState:
        if state.get("error"):
            return {}
        try:
            access = SQLExecutor2Access(cfg)
            profile = profile_table(access, state.get("columns", []), cfg, state.get("plan", {}).get("columns") or None)
            dup = duplicate_key_check(access, state.get("columns", []), cfg)
            return {"profile": profile, "duplicate_check": dup}
        except Exception as exc:
            return {"error": f"Erreur pendant le profiling via SQLExecutor2: {exc}"}

    def run_checks(state: DQState) -> DQState:
        if state.get("error"):
            return {}
        try:
            evaluation = evaluate_rules(
                state.get("profile", {}),
                state.get("columns", []),
                state.get("duplicate_check"),
                cfg,
                state.get("plan", {}),
            )
            return {"rule_evaluation": evaluation}
        except Exception as exc:
            return {"error": f"Erreur pendant l'évaluation des règles qualité: {exc}"}

    def run_read_only_sql(state: DQState) -> DQState:
        try:
            access = SQLExecutor2Access(cfg)
            plan = state.get("plan", {})
            custom_sql = str(plan.get("custom_sql") or "").strip()
            if not custom_sql:
                # Safe default for common SQL-like questions when the planner did not generate SQL.
                custom_sql = f"SELECT COUNT(*) AS row_count FROM {access.table_ref}"
            sql = add_limit_if_needed(custom_sql, cfg.max_return_rows)
            df = access.query_df(sql)
            return {"sql_query": sql, "query_result": df_to_records(df, max_rows=cfg.max_return_rows), "table_ref": access.table_ref, "connection": access.connection_summary}
        except Exception as exc:
            return {"error": f"Erreur SQL lecture seule: {exc}"}

    def compose_answer(state: DQState) -> DQState:
        report = format_detailed_report(state, cfg)
        artifacts = build_agent_hub_artifacts(state, cfg)
        if not cfg.use_llm_final_summary or state.get("error"):
            return {"response": report, "artifacts": artifacts}
        try:
            summarizer = llm.with_config({"tags": ["final_response"], "metadata": {"stage": "final_response"}})
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "Tu es un expert data quality Dataiku/Impala. Rédige en français une lecture experte courte, "
                    "orientée risques et plan d'action. Ne fabrique aucune métrique et n'enlève aucune section du rapport: "
                    "ton texte sera inséré dans un rapport détaillé déterministe. Maximum 8 puces."
                ),
                (
                    "human",
                    "Demande utilisateur:\n{user_query}\n\n"
                    "Plan:\n{plan}\n\n"
                    "Résultats JSON disponibles:\n{state_json}"
                ),
            ]).format_messages(
                user_query=state.get("user_query", ""),
                plan=serialize_json(state.get("plan", {}), max_chars=4000),
                state_json=serialize_json({k: v for k, v in state.items() if k not in {"response"}}, max_chars=14000),
            )
            summary = str(summarizer.invoke(prompt).content or "").strip()
            return {"response": insert_llm_summary(report, summary), "artifacts": artifacts}
        except Exception as exc:
            logger.warning("Final LLM summary failed; using deterministic detailed report: %s", exc)
            return {"response": report, "artifacts": artifacts}

    def route_after_plan(state: DQState) -> str:
        if state.get("error"):
            return "compose_answer"
        action = state.get("plan", {}).get("action", "checks")
        if action == "sql":
            return "run_read_only_sql"
        if action == "help":
            return "compose_answer"
        return "inspect_schema"

    def route_after_schema(state: DQState) -> str:
        if state.get("error"):
            return "compose_answer"
        action = state.get("plan", {}).get("action", "checks")
        if action == "schema":
            return "compose_answer"
        return "run_profile"

    def route_after_profile(state: DQState) -> str:
        if state.get("error"):
            return "compose_answer"
        action = state.get("plan", {}).get("action", "checks")
        if action == "checks":
            return "run_checks"
        return "compose_answer"

    graph = StateGraph(DQState)
    graph.add_node("plan_request", plan_request)
    graph.add_node("inspect_schema", inspect_schema)
    graph.add_node("run_profile", run_profile)
    graph.add_node("run_checks", run_checks)
    graph.add_node("run_read_only_sql", run_read_only_sql)
    graph.add_node("compose_answer", compose_answer)

    graph.add_edge(START, "plan_request")
    graph.add_conditional_edges(
        "plan_request",
        route_after_plan,
        {
            "inspect_schema": "inspect_schema",
            "run_read_only_sql": "run_read_only_sql",
            "compose_answer": "compose_answer",
        },
    )
    graph.add_conditional_edges(
        "inspect_schema",
        route_after_schema,
        {
            "run_profile": "run_profile",
            "compose_answer": "compose_answer",
        },
    )
    graph.add_conditional_edges(
        "run_profile",
        route_after_profile,
        {
            "run_checks": "run_checks",
            "compose_answer": "compose_answer",
        },
    )
    graph.add_edge("run_checks", "compose_answer")
    graph.add_edge("run_read_only_sql", "compose_answer")
    graph.add_edge("compose_answer", END)
    return graph.compile()


# -----------------------------------------------------------------------------
# Streaming adapters for Dataiku BaseLLM
# -----------------------------------------------------------------------------

STAGE_LABELS = {
    "plan_request": "Analyse de la demande et préparation du plan",
    "inspect_schema": "Lecture du schéma via SQLExecutor2",
    "run_profile": "Calcul des métriques via SQLExecutor2",
    "run_checks": "Évaluation des règles qualité",
    "run_read_only_sql": "Exécution d'une requête SQL lecture seule",
    "compose_answer": "Rédaction de la réponse finale",
}


def normalize_stream_item(item: Any) -> Tuple[str, Any]:
    """Normalize LangGraph stream outputs across versions."""
    if isinstance(item, dict) and "type" in item and "data" in item:
        return str(item["type"]), item["data"]
    if isinstance(item, tuple) and len(item) == 2:
        first, second = item
        if isinstance(first, str) and first in {"updates", "values", "messages", "custom", "debug", "checkpoints", "tasks"}:
            return first, second
        return "messages", item
    return "updates", item


def chunk_text(text: str, size: int = 60) -> Iterable[str]:
    # Keep Markdown readable by chunking on spaces where possible.
    text = text or ""
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start + 15:
                end = space + 1
        yield text[start:end]
        start = end


def message_chunk_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: List[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") in {"text", "output_text"} and block.get("text"):
                    pieces.append(str(block["text"]))
                elif block.get("content"):
                    pieces.append(str(block["content"]))
            elif isinstance(block, str):
                pieces.append(block)
        return "".join(pieces)
    return ""


def stream_event(stage: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "chunk": {
            "type": "event",
            "eventKind": "progress",
            "eventData": {
                "stage": stage,
                "message": STAGE_LABELS.get(stage, stage),
                **(data or {}),
            },
        }
    }


def get_llm_from_mesh(cfg: AgentConfig, settings: Any) -> Any:
    if not cfg.llm_id:
        raise ValueError("LLM_ID n'est pas configuré. Renseignez la constante LLM_ID en haut du fichier. Exemple: openai:myconnection:gpt-4o-mini")
    project = dataiku.api_client().get_default_project()
    # completion_settings=settings lets DSS propagate the caller's temperature,
    # max output tokens, and other compatible LLM Mesh settings.
    return project.get_llm(cfg.llm_id).as_langchain_chat_model(completion_settings=settings)


class MyLLM(BaseLLM):
    """Dataiku Code Agent / Custom Agent entry point."""

    def __init__(self):
        pass

    def set_config(self, config, plugin_config=None):
        # Intentionnellement ignoré: cette version standalone est configurée
        # uniquement par les constantes en haut du fichier.
        self.config_override = {}

    def _config(self) -> AgentConfig:
        return build_config()

    def process(self, query, settings, trace):
        """Non-streaming fallback used by clients that do not request streaming."""
        prompt = extract_last_user_prompt(query)
        cfg = self._config()
        llm = get_llm_from_mesh(cfg, settings)
        graph = build_graph(cfg, llm)
        tracer = LangchainToDKUTracer(dku_trace=trace)
        result = graph.invoke({"user_query": prompt}, config={"callbacks": [tracer]})
        return {
            "text": result.get("response") or format_manual_response(result, cfg),
            "artifacts": result.get("artifacts", []),
        }

    def process_stream(self, query, settings, trace):
        """Synchronous streaming fallback. Prefer aprocess_stream when available."""
        completion = self.process(query, settings, trace)
        response = completion.get("text", "")
        for part in chunk_text(response):
            yield {"chunk": {"text": part}}
        artifacts = completion.get("artifacts") or []
        if artifacts:
            yield {"chunk": {"artifacts": artifacts}}

    async def aprocess_stream(self, query, settings, trace) -> AsyncIterator[Dict[str, Any]]:
        """Async streaming implementation for Dataiku Code Agents."""
        prompt = extract_last_user_prompt(query)
        cfg = self._config()
        llm = get_llm_from_mesh(cfg, settings)
        graph = build_graph(cfg, llm)
        tracer = LangchainToDKUTracer(dku_trace=trace)

        final_text: Optional[str] = None
        final_artifacts: List[Dict[str, Any]] = []
        emitted_final_tokens = False

        yield stream_event("plan_request")
        try:
            async for item in graph.astream(
                {"user_query": prompt},
                stream_mode=["updates", "messages"],
                config={"callbacks": [tracer]},
            ):
                mode, data = normalize_stream_item(item)

                if mode == "updates":
                    if isinstance(data, dict):
                        for node_name, update in data.items():
                            if node_name in STAGE_LABELS:
                                event_data: Dict[str, Any] = {}
                                if isinstance(update, dict):
                                    if update.get("table_ref"):
                                        event_data["table"] = update.get("table_ref")
                                    if update.get("sql_query"):
                                        event_data["sql"] = update.get("sql_query")
                                    if update.get("rule_evaluation"):
                                        event_data["status"] = update["rule_evaluation"].get("overall_status")
                                    if update.get("response"):
                                        final_text = str(update["response"])
                                    if update.get("artifacts"):
                                        final_artifacts = list(update.get("artifacts") or [])
                                yield stream_event(str(node_name), event_data)

                elif mode == "messages":
                    if isinstance(data, tuple) and len(data) == 2:
                        message, metadata = data
                    else:
                        message, metadata = data, {}
                    metadata = metadata or {}
                    node_name = metadata.get("langgraph_node") or metadata.get("node")
                    tags = set(metadata.get("tags") or [])
                    # Do not stream planner JSON; only stream the final answer.
                    if node_name == "compose_answer" or "final_response" in tags:
                        content = message_chunk_text(message)
                        if content:
                            emitted_final_tokens = True
                            yield {"chunk": {"text": content}}

            # If the underlying LLM does not stream token chunks, stream the final
            # response in small pieces so Dataiku clients still get incremental UX.
            if final_text and not emitted_final_tokens:
                for part in chunk_text(final_text):
                    yield {"chunk": {"text": part}}
                    await asyncio.sleep(0)

            if final_artifacts:
                yield {"chunk": {"artifacts": final_artifacts}}

        except Exception as exc:
            logger.exception("Streaming agent failed")
            text = f"\n\n### Erreur pendant l'exécution de l'agent\n{exc}"
            for part in chunk_text(text):
                yield {"chunk": {"text": part}}
                await asyncio.sleep(0)
