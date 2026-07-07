# -*- coding: utf-8 -*-
"""
Agent Dataiku + LangGraph pour Text-to-SQL Impala avec Mistral Small 3.2
========================================================================

VERSION 3 — durcie pour couvrir le jeu de test (54 questions).

Améliorations vs version 2 :

1. REFORMULATION DE QUESTION AUTONOME (follow-ups)
   Le routeur produit une "standalone_question" qui intègre les tours
   précédents ("Donne-moi par famille valeur", "ces obligations",
   "Non, première date dispo de février 2026"...). C'est cette question
   reformulée qui alimente plan / SQL / juge / réponse.

2. RÉSOLUTION DES DATES CONTRE LA BASE
   Les dates extraites (formats FR : 30/03/2026, "30 juin 2025",
   "début février", "fin mai", "cette semaine", "le mois dernier",
   "depuis le début d'année", date du jour déclarée par l'utilisateur)
   sont RÉSOLUES en vraies dates présentes en base (plus proche <=,
   première/dernière du mois ou de l'année, bornes de période).
   Le SQL final utilise exclusivement ces dates résolues.

3. RÉSOLUTION DU VOCABULAIRE MÉTIER
   Variantes automatiques (minuscules, sans accents, singulier) +
   synonymes FR->EN (défense->defense/aerospace, luxe->luxury,
   banque->bank, italiennes->italy...). Si le lookup LIKE ne trouve
   rien, on remonte un échantillon DISTINCT de la colonne pour que le
   modèle choisisse la valeur EXACTE de la base. Groupes de pays
   ("Union européenne") gérés par liste des 27 + échantillon.

4. RÉPONSE DIRECTE VIA LLM DÉDIÉ
   Définitions ("C'est quoi une obligation ?"), pédagogie
   (taux vs valorisation), opinions/conseils ("bonne idée d'investir
   dans la défense ?") -> réponse prudente, sans conseil personnalisé,
   avec proposition d'analyser l'exposition réelle du portefeuille.

5. PATTERNS SQL IMPALA DANS LES PROMPTS
   Comparaison multi-dates (IN + GROUP BY date), part du total
   (SUM() OVER ()), segment vs reste (CASE WHEN), top/flop (ORDER BY
   ... LIMIT), analyse multi-axes (UNION ALL), NULLIF pour les
   divisions, year()/month(), interdiction GROUPING SETS/ROLLUP.

6. HONNÊTETÉ SUR LES DONNÉES ABSENTES
   SEDOL, NAV officielle, maturité, duration, notation, fonds propres
   réglementaires : l'agent doit dire que la donnée n'existe pas dans
   la table et proposer le proxy le plus proche, jamais inventer.

7. JUGE RENFORCÉ
   Reçoit le contexte dates/filtres + numéro de tentative. Résultat
   vide, date non résolue, filtre non matché -> consigne de correction
   concrète (autre orthographe vue dans l'échantillon, date résolue...).

8. RÉPONSE FINALE AMÉLIORÉE
   Mentionne la date réellement utilisée si différente de la date
   demandée, chiffres lisibles, limites explicites, réponse concise.

Principe de streaming (inchangé) :
- graph.stream(state, stream_mode=["updates", "messages", "custom"]).
- Évènements custom : thinking / tool_call / tool_result / section.
- PLAN, RÉPONSE FINALE et RÉPONSE DIRECTE streamés token par token.

A adapter dans chaque projet :
- LLM_ID
- SQL_CONNECTION
- TABLE_NAME
- DATE_EXPR_TEMPLATE si date_valorisation est stockée en STRING
- BUSINESS_SYNONYMS / règles métier dans SCHEMA_DESCRIPTION
"""

from __future__ import annotations

import calendar
import json
import re
import unicodedata
from datetime import date
from operator import add as list_add
from typing import Annotated, Any, Iterator, TypedDict

import dataiku
from dataiku import SQLExecutor2
from dataiku.llm.python import BaseLLM

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# get_stream_writer permet d'émettre des évènements custom depuis un noeud.
# Disponible dans les versions récentes de LangGraph (>= 0.2.34 environ).
try:
    from langgraph.config import get_stream_writer
except Exception:  # pragma: no cover - compat versions plus anciennes
    get_stream_writer = None  # type: ignore[assignment]

# =============================================================================
# CONFIGURATION PROJET
# =============================================================================

# Mistral Small 3.2 = release 2506.
# Le nom exact dépend de la connexion LLM Mesh déclarée chez vous.
LLM_ID = "mistral:votre-connexion:mistral-small-2506"

SQL_CONNECTION = "votre_connexion_sql"

# Nom de la table Impala à interroger.
TABLE_NAME = "votre_table_impala"

MAX_ROWS = 50
MAX_SQL_ATTEMPTS = 3
TEMPERATURE = 0.1

# Expression SQL pour normaliser une colonne date en 'yyyy-MM-dd'.
# - Colonne TIMESTAMP ou DATE : "to_date({col})" fonctionne.
# - Colonne STRING déjà au format 'yyyy-MM-dd' : mettre simplement "{col}".
DATE_EXPR_TEMPLATE = "to_date({col})"

# Nombre max de lignes remontées par un échantillon DISTINCT de colonne.
DISTINCT_SAMPLE_LIMIT = 60

# Nombre max de requêtes de vérification (lookups) par noeud de contexte.
MAX_LOOKUPS_PER_NODE = 6

# En streaming, affiche des messages d'avancement par noeud.
STREAM_PROGRESS = True

# En streaming, affiche le raisonnement (thinking) et les outils (tools).
STREAM_THINKING = True
STREAM_TOOLS = True

# Longueur max d'un résultat d'outil affiché en direct (évite de noyer le flux).
TOOL_RESULT_PREVIEW_CHARS = 1500

# =============================================================================
# DESCRIPTION DU SCHEMA
# =============================================================================

PORTFOLIO_COLUMNS = [
    "id_portefeuille",
    "classification_portefeuille_l7",
    "type_ptf_cpt_french_l6",
    "code_niveau_6",
    "code_niveau_7",
]

DATE_COLUMNS = [
    "date_valorisation",
    "date_insertion",
]

FILTERABLE_COLUMNS = [
    "id_valeur",
    "id_contrat_mx",
    "libelle_famille_valeur",
    "code_tiers_emetteur",
    "emetteur",
    "code_direction",
    "secteur_eco_2_lib",
    "libelle_famille",
    "secteur_eco_icb_n3_libelle",
    "secteur_eco_bics_n3_libelle",
    "devise_engagement",
    "cod_pays_implant",
    "relation_couverture",
    "type_risques_l4",
    "source",
    "reference_grappage",
    "num_lot",
    "id_valeur_sous_jacent",
    "libelle_sous_famille_valeur",
    "type_taux",
    "type_produits_l3",
    "sens_contrat_position",
    "sens_contrat",
    "section",
    "cmgp_groupe",
    "cmgp_sous_groupe",
    "nom_societe",
    "secteur_activite",
    "nom_foret",
    "df",
    "typologie_principale",
    "libelle_pays_emetteur",
    "secteur_eco",
]

# Colonnes "pays" : utilisées pour les groupes de pays (UE, zone euro...).
COUNTRY_COLUMNS = ["libelle_pays_emetteur", "cod_pays_implant"]

NUMERIC_MEASURES = [
    "valeur_de_marche_ctv",
    "valeur_bilan_brute_ctv",
    "pmv_latente_exteriorisables",
    "provisions_ctv",
    "pvl_mvl_ccs_hors_nomin_resid_cmgp",
    "soulte_inflation",
    "variation_indexation",
    "pvl_mvl_cmgp",
    "nominal_residuel_ctv",
    "icne_ctv",
    "valeur_boursiere_plc_ctv",
    "valeur_actuelle_ctv",
    "soulte_amortissable_initiale_ctv",
    "amortissement_soulte",
    "xx_flux_nom",
    "montant_brut_dev_val",
    "nb_titres",
    "pmvl_devise_cotation",
    "pmvl_ctv",
    "strike",
    "provisions_stock_ctv",
    "pmv_latente_ctv",
    "provisions_eoy",
    "total_encours",
]

ALL_COLUMNS = [
    "date_valorisation",
    "id_portefeuille",
    "id_valeur",
    "id_contrat_mx",
    "libelle_famille_valeur",
    "code_tiers_emetteur",
    "emetteur",
    "code_direction",
    "secteur_eco_2_lib",
    "libelle_famille",
    "secteur_eco_icb_n3_libelle",
    "secteur_eco_bics_n3_libelle",
    "classification_portefeuille_l7",
    "code_niveau_6",
    "code_niveau_7",
    "devise_engagement",
    "cod_pays_implant",
    "valeur_de_marche_ctv",
    "valeur_bilan_brute_ctv",
    "pmv_latente_exteriorisables",
    "provisions_ctv",
    "relation_couverture",
    "type_risques_l4",
    "source",
    "reference_grappage",
    "num_lot",
    "id_valeur_sous_jacent",
    "type_ptf_cpt_french_l6",
    "libelle_sous_famille_valeur",
    "pvl_mvl_ccs_hors_nomin_resid_cmgp",
    "type_taux",
    "soulte_inflation",
    "variation_indexation",
    "pvl_mvl_cmgp",
    "type_produits_l3",
    "nominal_residuel_ctv",
    "icne_ctv",
    "valeur_boursiere_plc_ctv",
    "valeur_actuelle_ctv",
    "soulte_amortissable_initiale_ctv",
    "amortissement_soulte",
    "xx_flux_nom",
    "crs_devise_cotation_dteva10",
    "sens_contrat_position",
    "montant_brut_dev_val",
    "nb_titres",
    "pmvl_devise_cotation",
    "pmvl_ctv",
    "strike",
    "sens_contrat",
    "provisions_stock_ctv",
    "section",
    "cmgp_groupe",
    "cmgp_sous_groupe",
    "pmv_latente_ctv",
    "nom_societe",
    "secteur_activite",
    "nom_foret",
    "df",
    "typologie_principale",
    "libelle_pays_emetteur",
    "provisions_eoy",
    "secteur_eco",
    "total_encours",
    "date_insertion",
]

# Synonymes / traductions métier pour la recherche de valeurs en base.
# Clés en minuscules SANS accents. À enrichir au fil des tests avec les
# libellés réellement présents dans votre table.
BUSINESS_SYNONYMS: dict[str, list[str]] = {
    "defense": ["defense", "defence", "aerospace", "armement", "militaire"],
    "luxe": ["luxury", "lux"],
    "banque": ["bank", "banking"],
    "banques": ["bank", "banking"],
    "bancaire": ["bank", "banking"],
    "sante": ["health", "healthcare", "health care", "pharma"],
    "assurance": ["insurance"],
    "energie": ["energy", "oil", "gas", "utilities"],
    "technologie": ["technology", "tech", "software"],
    "immobilier": ["real estate", "property", "reit"],
    "automobile": ["auto", "car"],
    "italien": ["ital", "italy", "italie"],
    "italienne": ["ital", "italy", "italie"],
    "italiens": ["ital", "italy", "italie"],
    "italiennes": ["ital", "italy", "italie"],
    "italie": ["ital", "italy"],
    "francais": ["france", "french", "franc"],
    "francaise": ["france", "french", "franc"],
    "francaises": ["france", "french", "franc"],
    "allemand": ["german", "germany", "allemagne"],
    "allemande": ["german", "germany", "allemagne"],
    "espagnol": ["spain", "spanish", "espagne"],
    "americain": ["united states", "usa", "us", "etats-unis"],
    "americaines": ["united states", "usa", "us", "etats-unis"],
    "etats-unis": ["united states", "usa", "us"],
    "usa": ["united states", "us"],
    "obligation": ["oblig", "bond"],
    "obligations": ["oblig", "bond"],
    "obligataire": ["oblig", "bond"],
    "obligataires": ["oblig", "bond"],
    "action": ["action", "equity", "stock"],
    "actions": ["action", "equity", "stock"],
    "etat": ["etat", "state", "government", "sovereign", "souverain"],
    "tresorerie": ["treso", "treasury"],
    "discretionnaire": ["discretionnaire", "discretion", "discretionary"],
    "discretionnaires": ["discretionnaire", "discretion", "discretionary"],
    "monetaire": ["monetaire", "money market"],
    "fonds": ["fonds", "fund", "opc", "sicav", "fcp"],
    "taux": ["taux", "rate", "fixed income"],
}

# Groupes de pays reconnus dans les questions.
COUNTRY_GROUP_TERMS = {
    "union europeenne",
    "l'union europeenne",
    "ue",
    "eu",
    "europe",
    "zone euro",
    "europeenne",
}

# Référentiel pays.
# La table stocke :
#   - cod_pays_implant     : code ISO 3166-1 alpha-3 (FRA, ITA, USA, DEU...)
#                            -> SANS accent ni casse : filtre le PLUS FIABLE.
#   - libelle_pays_emetteur: nom FRANÇAIS, 1re lettre en majuscule, accents
#                            conservés (États-Unis, Suède, Israël, Pays-Bas,
#                            République tchèque, Corée du Sud, Mexique...).
# Comme LOWER() sous Impala NE supprime PAS les accents, on privilégie le
# filtre sur cod_pays_implant (code ISO3) et on garde le libellé français
# EXACT (accentué) comme filtre secondaire.
#
# Format de chaque entrée : (nom_français_exact, code_iso3, [alias normalisés]).
# Les alias servent à reconnaître le pays quel que soit le libellé utilisateur
# (anglais, adjectif de nationalité, code...). Ils sont comparés après
# normalisation (minuscules + suppression des accents).
_COUNTRY_TABLE: list[tuple[str, str, list[str]]] = [
    # --- Union européenne (27) ---
    ("Allemagne", "DEU", ["germany", "german", "allemand", "deutschland"]),
    ("Autriche", "AUT", ["austria", "austrian", "autrichien"]),
    ("Belgique", "BEL", ["belgium", "belgian", "belge"]),
    ("Bulgarie", "BGR", ["bulgaria", "bulgarian", "bulgare"]),
    ("Chypre", "CYP", ["cyprus", "cypriot", "chypriote"]),
    ("Croatie", "HRV", ["croatia", "croatian", "croate"]),
    ("Danemark", "DNK", ["denmark", "danish", "danois"]),
    ("Espagne", "ESP", ["spain", "spanish", "espagnol"]),
    ("Estonie", "EST", ["estonia", "estonian", "estonien"]),
    ("Finlande", "FIN", ["finland", "finnish", "finlandais"]),
    ("France", "FRA", ["french", "francais", "francaise"]),
    ("Grèce", "GRC", ["greece", "greek", "grec", "grecque"]),
    ("Hongrie", "HUN", ["hungary", "hungarian", "hongrois"]),
    ("Irlande", "IRL", ["ireland", "irish", "irlandais"]),
    ("Italie", "ITA", ["italy", "italian", "italien", "ital"]),
    ("Lettonie", "LVA", ["latvia", "latvian", "letton"]),
    ("Lituanie", "LTU", ["lithuania", "lithuanian", "lituanien"]),
    ("Luxembourg", "LUX", ["luxembourgish", "luxembourgeois"]),
    ("Malte", "MLT", ["malta", "maltese", "maltais"]),
    ("Pays-Bas", "NLD", ["netherlands", "dutch", "hollande", "neerlandais", "holland"]),
    ("Pologne", "POL", ["poland", "polish", "polonais"]),
    ("Portugal", "PRT", ["portuguese", "portugais"]),
    ("République tchèque", "CZE", ["czech republic", "czechia", "czech", "tcheque", "tchequie"]),
    ("Roumanie", "ROU", ["romania", "romanian", "roumain"]),
    ("Slovaquie", "SVK", ["slovakia", "slovak", "slovaque"]),
    ("Slovénie", "SVN", ["slovenia", "slovenian", "slovene"]),
    ("Suède", "SWE", ["sweden", "swedish", "suedois"]),
    # --- Hors UE fréquents ---
    ("États-Unis", "USA", ["united states", "usa", "us", "u.s.", "american", "americain", "americaine", "amerique", "etats unis"]),
    ("Royaume-Uni", "GBR", ["united kingdom", "uk", "britain", "great britain", "british", "britannique", "angleterre", "england", "royaume uni"]),
    ("Suisse", "CHE", ["switzerland", "swiss"]),
    ("Norvège", "NOR", ["norway", "norwegian", "norvegien"]),
    ("Japon", "JPN", ["japan", "japanese", "japonais"]),
    ("Chine", "CHN", ["china", "chinese", "chinois"]),
    ("Canada", "CAN", ["canadian", "canadien"]),
    ("Mexique", "MEX", ["mexico", "mexican", "mexicain"]),
    ("Brésil", "BRA", ["brazil", "brazilian", "bresil", "bresilien"]),
    ("Inde", "IND", ["india", "indian", "indien"]),
    ("Australie", "AUS", ["australia", "australian", "australien"]),
    ("Corée du Sud", "KOR", ["south korea", "korea", "korean", "coree du sud", "coreen", "coree"]),
    ("Israël", "ISR", ["israel", "israeli", "israelien"]),
    ("Turquie", "TUR", ["turkey", "turkish", "turc", "turque"]),
    ("Russie", "RUS", ["russia", "russian", "russe"]),
    ("Singapour", "SGP", ["singapore", "singaporean"]),
    ("Hong Kong", "HKG", ["hongkong"]),
    ("Afrique du Sud", "ZAF", ["south africa", "afrique du sud", "sud-africain"]),
]


def _normalize_country_key(text: str) -> str:
    """Minuscule + sans accents + espaces compactés, pour comparer des libellés pays."""
    normalized = unicodedata.normalize("NFD", str(text))
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    normalized = normalized.lower().strip()
    normalized = re.sub(r"[\s]+", " ", normalized)
    return normalized


# Index de recherche : clé normalisée -> (nom_français, code_iso3).
COUNTRY_LOOKUP: dict[str, tuple[str, str]] = {}
for _fr_name, _iso3, _aliases in _COUNTRY_TABLE:
    _keys = {_normalize_country_key(_fr_name), _iso3.lower()}
    for _alias in _aliases:
        _keys.add(_normalize_country_key(_alias))
    for _k in _keys:
        if _k:
            COUNTRY_LOOKUP.setdefault(_k, (_fr_name, _iso3))

# Les 27 de l'UE (dans l'ordre du tableau ci-dessus).
_EU_SET = {
    "Allemagne", "Autriche", "Belgique", "Bulgarie", "Chypre", "Croatie",
    "Danemark", "Espagne", "Estonie", "Finlande", "France", "Grèce",
    "Hongrie", "Irlande", "Italie", "Lettonie", "Lituanie", "Luxembourg",
    "Malte", "Pays-Bas", "Pologne", "Portugal", "République tchèque",
    "Roumanie", "Slovaquie", "Slovénie", "Suède",
}
EU_NAMES_FR: list[str] = [n for (n, _i, _a) in _COUNTRY_TABLE if n in _EU_SET]
EU_ISO3: list[str] = [i for (n, i, _a) in _COUNTRY_TABLE if n in _EU_SET]

EU_COUNTRIES_HINT = ", ".join(
    f"{name} ({iso3})"
    for (name, iso3, _a) in _COUNTRY_TABLE
    if name in _EU_SET
)


def _sql_string(value: str) -> str:
    """Quote SQL simple pour des valeurs littérales."""
    return "'" + str(value).replace("'", "''") + "'"


# Snippets SQL prêts à l'emploi pour l'Union européenne.
# Le filtre par codes ISO3 est le plus fiable (aucun accent, aucune casse).
EU_IN_ISO3 = "(" + ", ".join(_sql_string(code) for code in EU_ISO3) + ")"
EU_IN_NAMES_FR = "(" + ", ".join(_sql_string(name) for name in EU_NAMES_FR) + ")"


def _resolve_country_term(term: str) -> tuple[str, str] | None:
    """Résout un libellé pays utilisateur en (nom_français_exact, code_iso3).

    1) correspondance exacte sur la clé normalisée,
    2) sinon correspondance par sous-chaîne (alias de longueur >= 4), la plus
       longue d'abord (gère "banques italiennes" -> Italie).
    Retourne None si aucun pays n'est reconnu.
    """
    key = _normalize_country_key(term)
    if not key:
        return None

    if key in COUNTRY_LOOKUP:
        return COUNTRY_LOOKUP[key]

    best: tuple[int, tuple[str, str]] | None = None
    for alias_key, resolved in COUNTRY_LOOKUP.items():
        if len(alias_key) >= 4 and alias_key in key:
            if best is None or len(alias_key) > best[0]:
                best = (len(alias_key), resolved)
    return best[1] if best else None


def _country_where_snippet(fr_name: str, iso3: str) -> str:
    """WHERE fiable pour un pays : priorité au code ISO3, libellé français en secours."""
    return (
        f"(cod_pays_implant = {_sql_string(iso3)} "
        f"OR libelle_pays_emetteur = {_sql_string(fr_name)})"
    )


# =============================================================================
# RÈGLES MÉTIER SPÉCIFIQUES (termes -> filtre exact)
# =============================================================================
# Certains termes du métier ne se retrouvent pas tels quels en base : ils
# correspondent à un filtre précis sur une colonne donnée. Ces règles sont
# appliquées de façon DÉTERMINISTE (elles priment sur la recherche libre du LLM).

# --- "banques" = secteur bancaire ---
BANK_SECTOR_COLUMN = "secteur_eco_2_lib"
# Valeurs d'énumération fournies par le métier.
# On inclut aussi l'orthographe "INSTITUTION_FINANCIERE" (avec N) par sécurité :
# une valeur absente dans un IN est sans effet, donc le filtre reste correct
# quelle que soit l'orthographe réellement stockée en base.
BANK_SECTOR_VALUES = [
    "INSTITUTION_FINACIERE",
    "INSTITUTION_FINANCIERE",
    "ETABLISSEMENT_CREDIT",
]

# --- "portefeuille de taux" = classification niveau 7 commençant par TV ou TF ---
RATE_PORTFOLIO_COLUMN = "classification_portefeuille_l7"
RATE_PORTFOLIO_PREFIXES = ["TV", "TF"]

# --- "secteur" -> colonne principale de secteur ---
# D'après le métier, le secteur se trouve PRINCIPALEMENT dans cette colonne.
# Toute mention d'un secteur dans la question doit être cherchée ici en priorité.
SECTOR_MAIN_COLUMN = "secteur_eco_bics_n3_libelle"
# Colonnes secteur secondaires (fallback si la valeur n'est pas trouvée dans
# la colonne principale). L'ordre reflète la priorité.
SECTOR_COLUMNS = [
    SECTOR_MAIN_COLUMN,
    "secteur_eco_icb_n3_libelle",
    "secteur_eco_2_lib",
    "secteur_eco",
    "secteur_activite",
]


def _bank_where_snippet() -> str:
    """WHERE exact pour le secteur bancaire."""
    values = ", ".join(_sql_string(v) for v in BANK_SECTOR_VALUES)
    return f"{BANK_SECTOR_COLUMN} IN ({values})"


def _rate_portfolio_where_snippet() -> str:
    """WHERE exact pour un portefeuille de taux (préfixe TV/TF)."""
    prefixes = ", ".join(_sql_string(p.upper()) for p in RATE_PORTFOLIO_PREFIXES)
    return f"LEFT(UPPER({RATE_PORTFOLIO_COLUMN}), 2) IN ({prefixes})"


BANK_WHERE = _bank_where_snippet()
RATE_PORTFOLIO_WHERE = _rate_portfolio_where_snippet()

# Règles détectables par mots-clés dans la question.
# kind = "filter" -> injecté dans le contexte filtres (noeud get_filter)
# kind = "portfolio" -> injecté dans le contexte portefeuille (noeud search_portfolio)
BUSINESS_TERM_RULES: list[dict[str, Any]] = [
    {
        "name": "banques (secteur bancaire)",
        "kind": "filter",
        "column": BANK_SECTOR_COLUMN,
        "where": BANK_WHERE,
        "aliases": [
            "banque", "banques", "bancaire", "bancaires",
            "secteur bancaire", "secteur des banques",
            "etablissement de credit", "etablissements de credit",
            "institution financiere", "institutions financieres",
        ],
        "explanation": (
            "Le terme « banques » désigne le SECTEUR bancaire : il se traduit par "
            f"un filtre {BANK_WHERE} "
            "(et NON par une recherche texte libre). "
            "S'applique quand « banque(s) » désigne le secteur, pas un émetteur "
            "précis nommé « Banque … »."
        ),
    },
    {
        "name": "portefeuille de taux",
        "kind": "portfolio",
        "column": RATE_PORTFOLIO_COLUMN,
        "where": RATE_PORTFOLIO_WHERE,
        "aliases": [
            "portefeuille de taux", "portefeuilles de taux",
            "portefeuille taux", "portefeuilles taux",
            "ptf de taux", "ptf taux",
            "poche de taux", "poche taux",
        ],
        "explanation": (
            "Un « portefeuille de taux » est identifié par "
            f"{RATE_PORTFOLIO_COLUMN} commençant par TV ou TF, soit le filtre "
            f"{RATE_PORTFOLIO_WHERE}."
        ),
    },
]


def _detect_business_rules(question: str, kind: str) -> list[dict[str, Any]]:
    """Détecte les règles métier déclenchées par des mots-clés de la question.

    Correspondance par mot entier (word boundary) sur le texte normalisé
    (minuscules, sans accents), pour éviter les faux positifs internes aux mots.
    """
    norm = _normalize_country_key(question)
    if not norm:
        return []

    fired: list[dict[str, Any]] = []
    for rule in BUSINESS_TERM_RULES:
        if rule.get("kind") != kind:
            continue
        for alias in rule.get("aliases", []):
            alias_norm = _normalize_country_key(alias)
            if not alias_norm:
                continue
            if re.search(r"\b" + re.escape(alias_norm) + r"\b", norm):
                fired.append(rule)
                break
    return fired


def _business_rule_aliases(kind: str) -> set[str]:
    """Ensemble d'alias normalisés (pour repérer un filtre LLM couvert par une règle)."""
    aliases: set[str] = set()
    for rule in BUSINESS_TERM_RULES:
        if rule.get("kind") != kind:
            continue
        for alias in rule.get("aliases", []):
            a = _normalize_country_key(alias)
            if a:
                aliases.add(a)
    return aliases


def _date_expr(col: str) -> str:
    """Expression SQL normalisant une colonne date en 'yyyy-MM-dd'."""
    return DATE_EXPR_TEMPLATE.format(col=col)


SCHEMA_DESCRIPTION = f"""
Table principale Impala :
- {TABLE_NAME}

Colonnes disponibles :
{", ".join(ALL_COLUMNS)}

Colonnes portefeuille importantes :
{", ".join(PORTFOLIO_COLUMNS)}

Colonnes date importantes :
{", ".join(DATE_COLUMNS)}

Colonnes de mesures numériques fréquentes :
{", ".join(NUMERIC_MEASURES)}

Glossaire des mesures (mapper le vocabulaire utilisateur vers les colonnes) :
- exposition / position / valorisation / valo / valeur de marché / stock -> SUM(valeur_de_marche_ctv)
- encours -> SUM(total_encours) si pertinent, sinon SUM(valeur_de_marche_ctv)
- plus-value / moins-value latente / PMVL / latent -> pmv_latente_ctv
  (variante "extériorisable" : pmv_latente_exteriorisables ; pmvl_ctv existe aussi)
- provisions -> provisions_ctv (stock : provisions_stock_ctv ; fin d'année : provisions_eoy)
- valeur bilan / valeur au bilan / bilan brut -> valeur_bilan_brute_ctv
- nombre de titres -> nb_titres ; nominal -> nominal_residuel_ctv ; coupon couru -> icne_ctv

Dimensions usuelles :
- famille valeur -> libelle_famille_valeur (sous-famille : libelle_sous_famille_valeur ; famille : libelle_famille)
- émetteur -> emetteur (code : code_tiers_emetteur)
- titre / instrument -> id_valeur (identifiant du titre, correspond en général au code ISIN)
- pays -> cod_pays_implant (code ISO 3166-1 alpha-3 : FRA, ITA, USA, DEU...)
  ou libelle_pays_emetteur (nom FRANÇAIS, 1re lettre majuscule, ACCENTS
  conservés : États-Unis, Suède, Israël, Pays-Bas, République tchèque,
  Corée du Sud, Mexique...).
  IMPORTANT : filtrer de préférence sur cod_pays_implant avec le code ISO3
  (aucun accent, aucune casse -> fiable). LOWER() ne supprime PAS les accents
  sous Impala, donc un LIKE sans accent échouerait sur libelle_pays_emetteur.
  Si l'on filtre sur libelle_pays_emetteur, utiliser le nom français EXACT et
  accentué avec = (ex. libelle_pays_emetteur = 'États-Unis').
- secteur -> COLONNE PRINCIPALE secteur_eco_bics_n3_libelle (BICS N3).
  Toute mention d'un secteur dans la question se cherche EN PRIORITÉ dans cette
  colonne. Colonnes secondaires (fallback) : secteur_eco_icb_n3_libelle (ICB N3),
  secteur_eco_2_lib, secteur_eco, secteur_activite.
- portefeuille -> id_portefeuille (+ classification_portefeuille_l7, type_ptf_cpt_french_l6, code_niveau_6/7)
- devise -> devise_engagement ; direction -> code_direction ; df (ex. valeurs DFFE, DFIN)

Groupes de pays :
- "Union européenne" / "UE" = les 27 pays membres (nom + code ISO3) :
  {EU_COUNTRIES_HINT}.
- Pour filtrer l'UE, utiliser de préférence les codes ISO3 :
  cod_pays_implant IN {EU_IN_ISO3}.
  (Filtre équivalent, moins fiable, par libellés français exacts :
  libelle_pays_emetteur IN {EU_IN_NAMES_FR}.)

Règles métier spécifiques (termes -> filtre EXACT, à appliquer tel quel) :
- « banques » / secteur bancaire -> {BANK_WHERE}
  (ne PAS faire de recherche texte libre pour ce terme).
- « portefeuille de taux » -> {RATE_PORTFOLIO_WHERE}
  (classification_portefeuille_l7 commençant par TV ou TF).
- « secteur » : toute mention d'un secteur (ex. Défense, Santé, Luxe, Banques...)
  se filtre EN PRIORITÉ sur secteur_eco_bics_n3_libelle. N'utiliser une autre
  colonne secteur que si la valeur n'y est pas trouvée.
Quand la question déclenche une de ces règles, le contexte (filtres ou
portefeuille) fournit le filtre exact à réutiliser dans le WHERE.

Données NON disponibles dans la table (le dire clairement à l'utilisateur,
proposer le proxy le plus proche, ne JAMAIS inventer de valeur) :
- code SEDOL (seul id_valeur, en général l'ISIN, est disponible)
- NAV officielle / NAV ajustée (proxys : valeur_de_marche_ctv, valeur_actuelle_ctv)
- maturité / échéance des titres, duration, sensibilité aux taux
- notation de crédit, consommation réglementaire de fonds propres
- corrélations de marché, historiques de cours hors dates de valorisation

Règles métier et SQL :
- Moteur SQL : Impala.
- Utiliser uniquement SELECT ou WITH.
- Ne jamais inventer une colonne.
- Filtrer les dates avec {_date_expr("date_valorisation")} = 'YYYY-MM-DD'
  (ou IN ('YYYY-MM-DD', ...) pour plusieurs dates).
  Année : year(date_valorisation) = 2026 ; mois : month(date_valorisation) = 3.
- Pour "dernier", "récent", "à date", utiliser en priorité date_valorisation
  sauf si l'utilisateur parle explicitement de date_insertion.
- Toujours utiliser les dates RÉSOLUES fournies dans le contexte dates
  (ce sont des dates réellement présentes en base).
- Si la question demande un montant total, utiliser SUM sur la mesure appropriée.
- Si la question demande une répartition, utiliser GROUP BY sur la dimension demandée.
- Si la question demande une liste de lignes détaillées, ajouter un LIMIT raisonnable.
"""

# =============================================================================
# PROMPTS
# =============================================================================

INITIAL_THINK_PROMPT = """Tu es le routeur initial d'un agent text-to-SQL pour une table de positions financières.

Tu dois : 1) reformuler la question en question AUTONOME, 2) décider quels noeuds appeler.
Réponds uniquement en JSON valide, sans markdown.

Date du jour : {today}

Schéma :
{schema}

Reformulation (standalone_question) :
- Si la question dépend des tours précédents (follow-up), réécris-la en une question
  complète et autonome qui intègre le sujet, les filtres, les dates et les corrections
  des messages précédents.
  Exemples :
  * Précédent : "exposition totale en valeur de marché au 30/03/2026" ; question : "Donne-moi par famille valeur."
    -> "Quelle est l'exposition en valeur de marché par famille de valeur au 30/03/2026 ?"
  * Précédent : liste d'obligations à VM <= 0 au 31/12/2025 ; question : "Quel est l'émetteur de ces obligations ?"
    -> "Quels sont les émetteurs des titres obligataires ayant une valeur de marché <= 0 au 31/12/2025 ?"
  * Précédent : réponse sur début février ; question : "Non, première date dispo de février 2026."
    -> "Quelle est la valeur de marché à la première date de valorisation disponible de février 2026 ?"
- Si la question est déjà autonome, recopie-la telle quelle.
- Conserve la langue française et tous les identifiants (GF595, ISIN, noms...).

Règles de routage :
- direct_answer=true UNIQUEMENT si aucune donnée de la table n'est nécessaire :
  salutations, aide, définitions ("c'est quoi une obligation ?"), pédagogie
  (lien taux/valorisation), opinions ou conseils d'investissement généraux.
  Laisse direct_answer_text vide : un noeud dédié rédigera la réponse.
- Si la question demande des données, agrégats, montants, expositions, encours,
  provisions, PMVL, portefeuille, secteur, émetteur, pays, devise, date, listes,
  comparaisons, analyses du portefeuille : direct_answer=false.
- Si la question mélange concept ET données du portefeuille, direct_answer=false.
- needs_portfolio_search=true si la question parle d'un portefeuille précis,
  id_portefeuille (ex. GF595, GC575), classification, niveau 6/7, type portefeuille,
  ou d'une famille de portefeuilles. En particulier, « portefeuille(s) de taux »
  déclenche needs_portfolio_search=true (mets "portefeuille de taux" dans
  portfolio_terms) : c'est une règle métier gérée par un noeud dédié.
  Mets les termes dans portfolio_terms.
- needs_filter_extraction=true si la question contient des filtres métier hors
  date/portefeuille : émetteur, titre, secteur, pays, zone (Union européenne),
  famille de valeur, devise, direction, df... Mets les termes dans filter_terms.
  En particulier, « banque(s) » / « secteur bancaire » déclenche
  needs_filter_extraction=true (mets "banques" dans filter_terms) : c'est une
  règle métier (secteur) gérée par un noeud dédié.
- needs_date_extraction=true si la question contient une date, période, année,
  mois, semaine, "dernier", "à fin", "au", "entre", "depuis", "début", "fin",
  "aujourd'hui", "cette semaine", "le mois dernier", ou si l'utilisateur déclare
  la date du jour.
- Si une requête SQL est nécessaire sans contexte particulier : needs_sql=true.

Format JSON attendu :
{{
  "standalone_question": "",
  "direct_answer": false,
  "direct_answer_text": "",
  "needs_sql": true,
  "needs_portfolio_search": false,
  "needs_filter_extraction": false,
  "needs_date_extraction": false,
  "portfolio_terms": [],
  "filter_terms": [],
  "date_terms": [],
  "reason_short": ""
}}
"""

DIRECT_ANSWER_PROMPT = """Tu es un assistant francophone spécialisé en finance de marché,
adossé à une base de positions de portefeuille.

Réponds directement à l'utilisateur, en français, de façon claire et concise.

Règles :
- Définitions et pédagogie (obligation, lien taux/prix, duration...) : réponse
  exacte, simple, avec un petit exemple chiffré si utile.
- Opinions de marché ou conseils d'investissement ("est-ce une bonne idée
  d'investir dans X ?") : reste prudent et factuel. Donne les éléments à
  considérer (valorisation, risques, diversification, horizon), rappelle que
  tu ne fournis pas de conseil d'investissement personnalisé, et propose
  d'analyser l'exposition actuelle du portefeuille en base si c'est pertinent.
- Ne fabrique jamais de données de marché récentes ni de chiffres de la base.
- Pas de SQL dans la réponse.
"""

GET_FILTER_PROMPT = """Tu extrais les filtres métier d'une question utilisateur pour préparer une requête SQL.

Réponds uniquement en JSON valide, sans markdown.

Question autonome :
{standalone_question}

Schéma :
{schema}

Colonnes filtrables possibles :
{filterable_columns}

Règles :
- Ignore les filtres portefeuille et date, traités par des noeuds dédiés.
- Ignore les conditions sur des mesures numériques (ex. valeur de marché <= 0) :
  elles seront gérées directement dans le SQL.
- Pour chaque filtre, propose dans "alternatives" 1 à 3 variantes plausibles du
  libellé tel qu'il pourrait exister en base (traduction anglaise, libellé
  sectoriel standard...). Ex : valeur "Défense" ->
  alternatives ["Defense", "Aerospace & Defense"] ; valeur "luxe" ->
  ["Luxury", "Luxury Goods"].
- SECTEUR : pour toute mention d'un secteur (Défense, Santé, Luxe, Énergie,
  Banques...), mets column="secteur_eco_bics_n3_libelle" (colonne principale du
  secteur). N'emploie une autre colonne secteur (secteur_eco_icb_n3_libelle...)
  que si l'utilisateur précise explicitement une autre nomenclature (ex. ICB).
- Choisis la colonne la plus plausible pour les autres dimensions
  (émetteur -> emetteur, titre/ISIN -> id_valeur, famille -> libelle_famille_valeur).
- PAYS : mets column="cod_pays_implant" (code ISO 3166-1 alpha-3, le plus
  fiable). Garde dans "value" le pays tel que formulé par l'utilisateur
  (ex. "Italie", "italiennes", "États-Unis") : un traitement dédié le convertit
  automatiquement en code ISO3 + nom français exact. Tu n'as pas besoin de
  connaître le code toi-même.
- Groupes de pays ("Union européenne", "UE", "zone euro") : column="cod_pays_implant",
  garde la valeur telle quelle ; un traitement dédié fournit la liste des codes.

Format JSON attendu :
{{
  "filters": [
    {{
      "column": "nom_colonne",
      "operator": "=",
      "value": "valeur",
      "alternatives": [],
      "confidence": 0.0
    }}
  ],
  "notes": ""
}}
"""

GET_DATE_PROMPT = """Tu extrais les contraintes temporelles d'une question utilisateur pour préparer une requête SQL Impala.

Réponds uniquement en JSON valide, sans markdown.

Date du jour réelle : {today}

Question autonome :
{standalone_question}

Schéma :
{schema}

Colonnes de date possibles :
{date_columns}

Règles :
- Si l'utilisateur DÉCLARE une autre date du jour ("nous sommes le 27 mars 2026"),
  remplis current_date_override avec cette date au format YYYY-MM-DD et calcule
  toutes les expressions relatives à partir d'elle.
- Convertis les formats français : "30/03/2026" -> 2026-03-30 ;
  "30 juin 2025" -> 2025-06-30 ; "premier février" -> YYYY-02-01 (année du contexte).
  Si l'année manque, utilise l'année évoquée dans la conversation, sinon celle
  de la date du jour.
- kind :
  * "latest"  : "dernier", "plus récent", "à date", "aujourd'hui", "dernière
    date de valorisation connue".
  * "exact"   : une date précise -> value=YYYY-MM-DD.
  * "month"   : un mois précis -> value=YYYY-MM.
  * "year"    : une année -> value=YYYY.
  * "between" : une période -> start / end en YYYY-MM-DD.
    "depuis le début d'année" -> start=YYYY-01-01, end="" (= dernière date dispo).
    "cette semaine" -> start=lundi de la semaine, end=date du jour.
  * "multi"   : comparaison de plusieurs dates ou mois -> values=[...]
    (chaque élément YYYY-MM-DD ou YYYY-MM).
    Ex : "janvier et mars 2026" -> values=["2026-01","2026-03"] ;
    "entre fin janvier et fin mars 2026" (comparaison de photos)
    -> values=["2026-01","2026-03"], prefer="last".
  * "min"     : "première date disponible" (toutes périodes) -> kind="min".
    "première date disponible en 2026" -> kind="year", value="2026", prefer="first".
- prefer :
  * "first" pour "début" (début février, début d'année, première date dispo).
  * "last" pour "fin" (fin mai, à fin janvier, fin du mois dernier, end of month).
  * "nearest" (défaut) pour une date exacte : la date disponible la plus
    proche <= la date demandée.
- "le mois dernier" / "fin du mois dernier" -> month du mois précédent, prefer="last".
- "début de semaine" -> exact sur le lundi de la semaine courante, prefer="nearest".
- Une seule photo de portefeuille demandée sans précision sur un mois
  ("la valo en mars") -> kind="month", prefer="last".
- Si aucune contrainte de date n'est exprimée mais qu'une valorisation est
  demandée -> kind="latest".
- Ne fabrique pas de date non déductible.

Format JSON attendu :
{{
  "current_date_override": "",
  "date_filters": [
    {{
      "column": "date_valorisation",
      "kind": "latest | exact | month | year | between | multi | min",
      "value": "",
      "start": "",
      "end": "",
      "values": [],
      "prefer": "nearest | first | last",
      "label": ""
    }}
  ],
  "notes": ""
}}
"""

PLAN_PROMPT = """Tu es un planificateur SQL pour une table Impala de positions financières.

Tu dois préparer un plan détaillé de requête, mais tu ne dois PAS écrire le SQL final.

Schéma :
{schema}

Contexte récent de conversation :
{conversation_context}

Question utilisateur (reformulée autonome) :
{question}

Contexte portefeuille :
{portfolio_context}

Contexte filtres :
{filter_context}

Contexte dates :
{date_context}

Consignes :
- Décris l'objectif métier en une phrase.
- Identifie les colonnes à sélectionner, les mesures à agréger (SUM/AVG/COUNT),
  les filtres WHERE, les GROUP BY, l'ordre de tri, le LIMIT éventuel.
- Utilise les dates RÉSOLUES du contexte dates (dates réellement en base) et
  les valeurs EXACTES trouvées dans le contexte filtres/portefeuille. Si un
  libellé exact figure dans un échantillon, choisis-le.
- Comparaison entre plusieurs dates : UNE seule requête, filtre IN sur les
  dates résolues + GROUP BY sur la date (et la dimension éventuelle).
- Part / poids / % du total : agrégat + SUM(...) OVER () pour le total.
- "X vs le reste" : CASE WHEN pour créer un segment 'X' / 'Reste'.
- Plus forte / plus faible position, top N, pire perte latente :
  ORDER BY mesure DESC ou ASC + LIMIT.
- Analyse multi-axes (risques, concentration, devise, secteur) : UNE requête
  UNION ALL de blocs agrégés, avec une colonne littérale 'axe' pour distinguer
  chaque bloc (ex. 'par devise', 'par secteur', 'top emetteurs').
- Question sur une donnée ABSENTE de la table (SEDOL, NAV, maturité, duration,
  fonds propres...) : le plan doit le dire et, si possible, prévoir la requête
  sur le proxy le plus proche pour rester utile.
- Question analytique ou de scénario (stress test, signaux faibles) : prévois
  UNE requête qui remonte les agrégats nécessaires (répartitions par famille /
  devise / secteur / dates), l'analyse qualitative sera faite dans la réponse.
- Ne pas inventer de colonne.
- Reste bref : 5 à 12 lignes.
"""

GENERATE_SQL_PROMPT = """Tu es un générateur SQL Impala.

Réponds uniquement avec une requête SQL SELECT ou WITH.
Ne mets pas de markdown.
Ne mets pas d'explication.

Schéma :
{schema}

Contexte récent de conversation :
{conversation_context}

Question utilisateur (reformulée autonome) :
{question}

Plan SQL :
{sql_plan}

Contexte portefeuille :
{portfolio_context}

Contexte filtres :
{filter_context}

Contexte dates :
{date_context}

Tentative numéro :
{attempt}

SQL précédent :
{previous_sql}

Résultat ou erreur précédente :
{previous_result}

Instruction de correction :
{fix_instruction}

Règles :
- Utilise uniquement la table {table_name}.
- Utilise uniquement des colonnes du schéma.
- Requête en lecture seule : SELECT ou WITH uniquement. UNE seule requête.
- Compatible Impala. Pas de GROUPING SETS, ROLLUP ni CUBE.
- Ne termine pas par un point-virgule.
- Dates : {date_expr_hint} = 'YYYY-MM-DD' (ou IN (...) pour plusieurs dates),
  year(date_valorisation) = ..., month(date_valorisation) = ... .
  Utilise EXACTEMENT les dates résolues du contexte dates. N'invente aucune date.
- Filtres texte : si une valeur exacte a été confirmée par une vérification ou
  un échantillon, utilise-la avec = (ou IN). Sinon,
  LOWER(CAST(col AS STRING)) LIKE '%valeur%' avec les variantes plausibles (OR).
- PAYS : filtrer sur cod_pays_implant avec le code ISO 3166-1 alpha-3
  (ex. cod_pays_implant = 'ITA'). Le contexte filtres fournit le code exact et
  le nom français. NE PAS faire de LIKE sans accent sur libelle_pays_emetteur
  (LOWER ne retire pas les accents sous Impala). Si tu filtres par nom, emploie
  le nom français EXACT accentué avec = (ex. libelle_pays_emetteur = 'États-Unis').
- Union européenne : cod_pays_implant IN (...codes ISO3 fournis dans le contexte...).
- Part du total : 100 * SUM(mesure) / NULLIF(SUM(SUM(mesure)) OVER (), 0).
- Comparaison de dates : GROUP BY {date_expr_hint} avec filtre IN, ORDER BY 1.
- Segment vs reste : CASE WHEN condition THEN 'X' ELSE 'Reste' END AS segment.
- Top/flop : ORDER BY mesure DESC (top) ou ASC (pires pertes) + LIMIT.
- Analyse multi-axes : UNION ALL de sous-requêtes agrégées partageant les mêmes
  colonnes (axe STRING, libelle STRING, montant DOUBLE).
- Toute division : protéger avec NULLIF(denominateur, 0).
- Ajoute un LIMIT raisonnable pour les listes détaillées.
- N'ajoute pas de LIMIT pour un agrégat global sauf si nécessaire.
"""

JUDGE_PROMPT = """Tu es un contrôleur qualité pour un agent text-to-SQL.

Tu dois dire si le résultat SQL répond à la question utilisateur.
Réponds uniquement en JSON valide, sans markdown.

Question utilisateur (reformulée autonome) :
{question}

Contexte dates (dates résolues, réellement disponibles) :
{date_context}

Contexte filtres (valeurs vérifiées / échantillons) :
{filter_context}

Plan SQL :
{sql_plan}

SQL exécuté :
{sql}

Résultat SQL :
{sql_result}

Tentative {attempt} sur {max_attempts}.

Format JSON attendu :
{{
  "is_sufficient": true,
  "reason": "",
  "fix_instruction": ""
}}

Règles :
- is_sufficient=true si le résultat permet de répondre à la question.
- is_sufficient=false si le SQL est faux, incomplet, hors sujet, s'il manque un
  filtre important, ou si les dates du SQL ne correspondent pas aux dates
  résolues du contexte.
- Résultat vide : presque toujours insuffisant. Propose une correction concrète
  dans fix_instruction : utiliser la date résolue du contexte, élargir le filtre
  texte en LIKE, ou reprendre un libellé exact vu dans les échantillons.
  Exception : si un résultat vide est une réponse plausible à la question
  (vérification d'existence), is_sufficient=true.
- Si la question porte sur une donnée absente de la table et que la requête
  fournit le proxy prévu au plan, is_sufficient=true.
- Si correction nécessaire, fix_instruction doit être une consigne courte et
  actionnable pour régénérer le SQL.
"""

FORMULATE_PROMPT = """Tu es un assistant data analyste financier.

Réponds en français naturel à l'utilisateur à partir du résultat SQL.

Question utilisateur (reformulée autonome) :
{question}

Contexte dates (dates demandées vs dates résolues) :
{date_context}

Plan SQL :
{sql_plan}

SQL exécuté :
{sql}

Résultat SQL :
{sql_result}

Contrôle qualité :
{judge}

Règles :
- Ne fabrique jamais de données : uniquement les chiffres du résultat SQL.
- Commence par la réponse directe à la question (le chiffre ou la liste).
- Si la date réellement utilisée diffère de la date demandée (ex. "01/02/2026"
  demandé, données au 2026-02-02), dis-le explicitement.
- Formate les montants lisiblement (séparateur de milliers, 2 décimales max,
  préciser qu'il s'agit de montants en contre-valeur CTV).
- Si le résultat est vide, dis-le clairement et donne l'explication la plus
  probable (date sans données, libellé différent en base...).
- Si le résultat est tronqué, mentionne-le.
- Si le résultat contient une erreur et qu'on ne peut plus corriger, explique la limite.
- Si la question portait sur une donnée absente de la table (SEDOL, NAV,
  maturité, duration, fonds propres réglementaires...), dis clairement que la
  donnée n'existe pas dans la table et présente le proxy fourni.
- Pour les questions analytiques (risques, concentration, stress, signaux
  faibles) : appuie chaque affirmation sur les chiffres retournés, explicite
  les hypothèses et rappelle les limites (pas de données de sensibilité ni de
  corrélations en base).
- Si utile, affiche le résultat sous forme de tableau markdown.
- Termine par une courte interprétation métier (1 à 2 phrases).
"""

# =============================================================================
# STREAMING : émission d'évènements custom depuis les noeuds
# =============================================================================


def _emit_event(event_type: str, **payload: Any) -> None:
    """Émet un évènement custom vers le flux LangGraph s'il existe.

    En mode non-streaming (graph.invoke), get_stream_writer lève une exception :
    on l'ignore pour que les noeuds restent utilisables dans les deux modes.
    """
    if get_stream_writer is None:
        return
    try:
        writer = get_stream_writer()
    except Exception:
        return
    if writer is None:
        return
    try:
        writer({"type": event_type, **payload})
    except Exception:
        pass


def _emit_thinking(title: str, text: str) -> None:
    if STREAM_THINKING:
        _emit_event("thinking", title=title, text=text)


def _emit_tool_call(name: str, sql: str) -> None:
    if STREAM_TOOLS:
        _emit_event("tool_call", name=name, sql=sql)


def _emit_tool_result(name: str, result: str) -> None:
    if STREAM_TOOLS:
        _emit_event("tool_result", name=name, result=result)


def _emit_section(title: str) -> None:
    _emit_event("section", title=title)


# =============================================================================
# HELPERS LLM / JSON / SQL
# =============================================================================


def _extract_json(text: str, default: dict[str, Any]) -> dict[str, Any]:
    """Extrait un objet JSON depuis une réponse LLM.

    Le modèle doit répondre en JSON pur, mais ce helper tolère un peu de bruit.
    """
    if not text:
        return dict(default)

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else dict(default)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return dict(default)

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else dict(default)
    except Exception:
        return dict(default)


def _invoke_json(llm: Any, prompt: list[AnyMessage], default: dict[str, Any]) -> dict[str, Any]:
    response = llm.invoke(prompt)
    parsed = _extract_json(response.content or "", default)

    # On fusionne avec le défaut pour garantir la présence des clés attendues.
    merged = dict(default)
    merged.update(parsed)
    return merged


def _extract_sql(text: str) -> str:
    """Nettoie une réponse LLM pour récupérer uniquement le SQL."""
    if not text:
        return ""

    match = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1)

    match = re.search(r"\b(select|with)\b.*", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(0)

    return text.strip().rstrip(";").strip()


def _strip_sql_comments(query: str) -> str:
    """Supprime les commentaires SQL simples (-- ... et /* ... */)."""
    query = re.sub(r"--.*?$", "", query, flags=re.MULTILINE)
    query = re.sub(r"/\*.*?\*/", "", query, flags=re.DOTALL)
    return query.strip()


def _strip_string_literals(query: str) -> str:
    """Remplace le contenu des littéraux de chaîne par une chaîne vide.

    Cela évite des faux positifs de validation sur des mots comme update/drop
    présents dans une valeur texte.
    """
    query = re.sub(r"'(?:[^']|'')*'", "''", query)
    query = re.sub(r'"(?:[^"]|"")*"', '""', query)
    return query


def _validate_read_only_sql(query: str) -> tuple[bool, str]:
    """Validation défensive : lecture seule, un seul statement."""
    checkable = _strip_string_literals(_strip_sql_comments(query))
    if not checkable:
        return False, "requête vide."

    without_final_semicolon = checkable[:-1].strip() if checkable.endswith(";") else checkable
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


def _strip_accents(text: str) -> str:
    """Supprime les accents (Défense -> Defense) pour les recherches LIKE."""
    normalized = unicodedata.normalize("NFD", str(text))
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _term_variants(term: str, max_variants: int = 4) -> list[str]:
    """Variantes de recherche d'un terme : brut, sans accents, singulier, synonymes."""
    base = str(term).strip().lower()
    variants: list[str] = []

    def add(candidate: str) -> None:
        candidate = candidate.strip().lower()
        if candidate and candidate not in variants:
            variants.append(candidate)

    add(base)
    unaccented = _strip_accents(base)
    add(unaccented)

    if len(unaccented) > 3 and unaccented.endswith("s"):
        add(unaccented[:-1])

    for key in (unaccented, unaccented.rstrip("s")):
        for synonym in BUSINESS_SYNONYMS.get(key, []):
            add(synonym)

    return variants[:max_variants]


def _like_where(
    columns: list[str],
    terms: list[str],
    max_terms: int = 3,
    max_variants: int = 3,
) -> str:
    """WHERE de recherche approximative (variantes incluses) sur plusieurs colonnes."""
    conditions: list[str] = []

    for term in terms[:max_terms]:
        term = str(term).strip()
        if not term:
            continue

        for variant in _term_variants(term, max_variants=max_variants):
            like_value = _sql_string(f"%{variant}%")
            for col in columns:
                conditions.append(f"LOWER(CAST({col} AS STRING)) LIKE {like_value}")

    return " OR ".join(conditions) if conditions else "1 = 0"


def _has_final_limit(query: str) -> bool:
    checkable = _strip_string_literals(_strip_sql_comments(query)).strip().rstrip(";").strip()
    return bool(re.search(r"\blimit\s+\d+\s*$", checkable, flags=re.IGNORECASE))


def _cap_or_add_impala_limit(query: str) -> str:
    """Ajoute ou plafonne un LIMIT final pour éviter de rapatrier trop de lignes.

    On évite le wrapping SELECT * FROM (...) car certains moteurs/versions Impala
    tolèrent mal certains WITH imbriqués.
    """
    inner = query.strip().rstrip(";").strip()

    if not _has_final_limit(inner):
        return f"{inner}\nLIMIT {MAX_ROWS + 1}"

    # Si le LLM a déjà mis un LIMIT trop grand en fin de requête, on le plafonne.
    def repl(match: re.Match[str]) -> str:
        try:
            requested = int(match.group(1))
        except Exception:
            return match.group(0)
        if requested > MAX_ROWS + 1:
            return f"LIMIT {MAX_ROWS + 1}"
        return match.group(0)

    return re.sub(r"\blimit\s+(\d+)\s*$", repl, inner, flags=re.IGNORECASE)


def _execute_sql_markdown(query: str) -> str:
    """Exécute une requête SQL en lecture seule et retourne un tableau markdown."""
    ok, reason = _validate_read_only_sql(query)
    if not ok:
        return f"Erreur : requête SQL refusée ({reason})"

    executable_sql = _cap_or_add_impala_limit(query)

    try:
        executor = SQLExecutor2(connection=SQL_CONNECTION)
        df = executor.query_to_df(executable_sql)

        if df.empty:
            return "La requête n'a retourné aucun résultat."

        truncated_msg = ""
        if len(df) > MAX_ROWS:
            df = df.head(MAX_ROWS)
            truncated_msg = f"\n\n(Résultats tronqués aux {MAX_ROWS} premières lignes.)"

        return df.to_markdown(index=False) + truncated_msg

    except Exception as exc:
        return f"Erreur SQL : {exc}"


def _execute_sql_scalar(query: str) -> tuple[str | None, str]:
    """Exécute une requête et retourne la première cellule (ou None + message)."""
    ok, reason = _validate_read_only_sql(query)
    if not ok:
        return None, f"Erreur : requête SQL refusée ({reason})"

    try:
        executor = SQLExecutor2(connection=SQL_CONNECTION)
        df = executor.query_to_df(query)

        if df.empty or df.iloc[0, 0] is None:
            return None, "aucune valeur"

        value = str(df.iloc[0, 0]).strip()
        # Normalise un éventuel timestamp '2026-03-30 00:00:00' en date.
        if re.match(r"^\d{4}-\d{2}-\d{2}", value):
            value = value[:10]
        if not value or value.lower() in ("nan", "nat", "none"):
            return None, "aucune valeur"
        return value, ""

    except Exception as exc:
        return None, f"Erreur SQL : {exc}"


def _scalar_lookup(label: str, sql: str) -> str | None:
    """Exécute un lookup scalaire en l'exposant dans le flux d'outils."""
    _emit_tool_call(label, sql)
    value, message = _execute_sql_scalar(sql)
    _emit_tool_result(label, value if value is not None else message)
    return value


def _distinct_sample(column: str, limit: int = DISTINCT_SAMPLE_LIMIT) -> str:
    """Échantillon DISTINCT d'une colonne, pour que le LLM voie le vocabulaire réel."""
    sql = (
        f"SELECT DISTINCT {column} FROM {TABLE_NAME} "
        f"WHERE {column} IS NOT NULL ORDER BY 1 LIMIT {limit}"
    )
    _emit_tool_call(f"Échantillon « {column} »", sql)
    result = _execute_sql_markdown(sql)
    _emit_tool_result(f"Échantillon « {column} »", result)
    return result


def _is_empty_result(result: str) -> bool:
    return result.strip().startswith("La requête n'a retourné aucun résultat")


def _month_bounds(year_month: str) -> tuple[str, str] | None:
    """'2026-02' -> ('2026-02-01', '2026-02-28')."""
    match = re.match(r"^(\d{4})-(\d{2})$", str(year_month).strip())
    if not match:
        return None
    year, month = int(match.group(1)), int(match.group(2))
    if not 1 <= month <= 12:
        return None
    last_day = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{last_day:02d}"


def _last_user_question(messages: list[AnyMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content or "")
    return ""


def _render_recent_conversation(messages: list[AnyMessage], max_messages: int = 8) -> str:
    """Rend les derniers tours utiles pour les prompts hors historique LangChain."""
    rendered: list[str] = []

    for msg in messages[-max_messages:]:
        if isinstance(msg, HumanMessage):
            role = "Utilisateur"
        elif isinstance(msg, AIMessage):
            role = "Assistant"
        else:
            continue

        content = str(msg.content or "").strip()
        if len(content) > 2000:
            content = content[:2000] + "..."
        if content:
            rendered.append(f"{role}: {content}")

    return "\n".join(rendered)


def _selected_context_branches(route: dict[str, Any]) -> list[str]:
    branches: list[str] = []

    if route.get("needs_portfolio_search"):
        branches.append("search_portfolio")

    if route.get("needs_filter_extraction"):
        branches.append("get_filter")

    if route.get("needs_date_extraction"):
        branches.append("get_date")

    return branches


# =============================================================================
# RÉSOLUTION DES DATES CONTRE LA BASE
# =============================================================================


def _resolve_date_filters(date_filters: list[dict[str, Any]]) -> list[str]:
    """Transforme les contraintes extraites en dates RÉELLEMENT disponibles.

    Retourne des lignes de texte à injecter dans le contexte dates.
    Chaque résolution effectue au plus 2 petites requêtes MIN/MAX.
    """
    lines: list[str] = []
    lookups = 0

    def nearest_before(expr: str, value: str) -> str | None:
        return _scalar_lookup(
            f"Date la plus proche <= {value}",
            f"SELECT MAX({expr}) FROM {TABLE_NAME} WHERE {expr} <= {_sql_string(value)}",
        )

    def nearest_after(expr: str, value: str) -> str | None:
        return _scalar_lookup(
            f"Date la plus proche >= {value}",
            f"SELECT MIN({expr}) FROM {TABLE_NAME} WHERE {expr} >= {_sql_string(value)}",
        )

    def bounds(expr: str, start: str, end: str) -> tuple[str | None, str | None]:
        first = _scalar_lookup(
            f"Première date entre {start} et {end}",
            f"SELECT MIN({expr}) FROM {TABLE_NAME} "
            f"WHERE {expr} >= {_sql_string(start)} AND {expr} <= {_sql_string(end)}",
        )
        last = _scalar_lookup(
            f"Dernière date entre {start} et {end}",
            f"SELECT MAX({expr}) FROM {TABLE_NAME} "
            f"WHERE {expr} >= {_sql_string(start)} AND {expr} <= {_sql_string(end)}",
        )
        return first, last

    for item in date_filters:
        if lookups >= MAX_LOOKUPS_PER_NODE:
            lines.append("(Nombre max de contrôles de dates atteint.)")
            break
        if not isinstance(item, dict):
            continue

        column = str(item.get("column", "") or "date_valorisation").strip()
        if column not in DATE_COLUMNS:
            column = "date_valorisation"
        expr = _date_expr(column)

        kind = str(item.get("kind", "") or "").strip().lower()
        prefer = str(item.get("prefer", "") or "").strip().lower()
        value = str(item.get("value", "") or "").strip()
        start = str(item.get("start", "") or "").strip()
        end = str(item.get("end", "") or "").strip()
        label = str(item.get("label", "") or "").strip()
        values = item.get("values", []) or []
        if not isinstance(values, list):
            values = []

        prefix = f"[{label}] " if label else ""

        if kind == "latest":
            lookups += 1
            resolved = _scalar_lookup(
                f"Dernière date « {column} »", f"SELECT MAX({expr}) FROM {TABLE_NAME}"
            )
            lines.append(
                f"{prefix}Dernière date disponible ({column}) : {resolved or 'introuvable'}."
            )

        elif kind == "min":
            lookups += 1
            resolved = _scalar_lookup(
                f"Première date « {column} »", f"SELECT MIN({expr}) FROM {TABLE_NAME}"
            )
            lines.append(
                f"{prefix}Première date disponible ({column}) : {resolved or 'introuvable'}."
            )

        elif kind == "exact" and value:
            lookups += 1
            if prefer == "first":
                resolved = nearest_after(expr, value) or nearest_before(expr, value)
            else:
                resolved = nearest_before(expr, value) or nearest_after(expr, value)
            if resolved is None:
                lines.append(f"{prefix}Aucune date disponible autour de {value}.")
            elif resolved == value:
                lines.append(f"{prefix}Date {value} disponible en base : utiliser {resolved}.")
            else:
                lines.append(
                    f"{prefix}Date {value} ABSENTE de la base ; date disponible la plus "
                    f"proche : {resolved}. Utiliser {resolved} et le signaler."
                )

        elif kind == "month" and value:
            month_bounds = _month_bounds(value)
            if month_bounds is None:
                lines.append(f"{prefix}Mois non interprétable : {value}.")
                continue
            lookups += 1
            first, last = bounds(expr, month_bounds[0], month_bounds[1])
            if first is None and last is None:
                fallback = nearest_before(expr, month_bounds[1])
                lines.append(
                    f"{prefix}Aucune date disponible sur le mois {value}. "
                    f"Dernière date antérieure disponible : {fallback or 'introuvable'}."
                )
            else:
                chosen = first if prefer == "first" else last
                lines.append(
                    f"{prefix}Mois {value} : première date dispo = {first}, dernière = {last}. "
                    f"Date à utiliser ({'début' if prefer == 'first' else 'fin'} de mois) : {chosen}."
                )

        elif kind == "year" and value:
            year_value = re.sub(r"\D", "", value)[:4]
            if len(year_value) != 4:
                lines.append(f"{prefix}Année non interprétable : {value}.")
                continue
            lookups += 1
            first, last = bounds(expr, f"{year_value}-01-01", f"{year_value}-12-31")
            if first is None and last is None:
                lines.append(f"{prefix}Aucune date disponible sur l'année {year_value}.")
            else:
                chosen = first if prefer == "first" else last
                lines.append(
                    f"{prefix}Année {year_value} : première date dispo = {first}, "
                    f"dernière = {last}. Date à privilégier : {chosen}."
                )

        elif kind == "between":
            if not start and not end:
                continue
            lookups += 1
            if not end:
                latest = _scalar_lookup(
                    f"Dernière date « {column} »", f"SELECT MAX({expr}) FROM {TABLE_NAME}"
                )
                end = latest or date.today().isoformat()
            if not start:
                start = "1900-01-01"
            first, last = bounds(expr, start, end)
            if first is None and last is None:
                fallback = nearest_before(expr, end)
                lines.append(
                    f"{prefix}Aucune date disponible entre {start} et {end}. "
                    f"Dernière date antérieure disponible : {fallback or 'introuvable'}."
                )
            else:
                lines.append(
                    f"{prefix}Période {start} -> {end} : première date dispo = {first}, "
                    f"dernière = {last}. Pour une photo unique sur la période, "
                    f"utiliser {last}."
                )

        elif kind == "multi" and values:
            resolved_values: list[str] = []
            for raw in values[:4]:
                if lookups >= MAX_LOOKUPS_PER_NODE:
                    break
                raw = str(raw).strip()
                if re.match(r"^\d{4}-\d{2}$", raw):
                    month_bounds = _month_bounds(raw)
                    if month_bounds is None:
                        continue
                    lookups += 1
                    first, last = bounds(expr, month_bounds[0], month_bounds[1])
                    chosen = first if prefer == "first" else last
                    if chosen:
                        resolved_values.append(f"{raw} -> {chosen}")
                elif re.match(r"^\d{4}-\d{2}-\d{2}$", raw):
                    lookups += 1
                    chosen = nearest_before(expr, raw) or nearest_after(expr, raw)
                    if chosen:
                        suffix = "" if chosen == raw else " (date demandée absente)"
                        resolved_values.append(f"{raw} -> {chosen}{suffix}")
            if resolved_values:
                lines.append(
                    f"{prefix}Dates de comparaison résolues : "
                    + " ; ".join(resolved_values)
                    + ". Utiliser EXACTEMENT les dates de droite dans un filtre IN."
                )
            else:
                lines.append(f"{prefix}Impossible de résoudre les dates de comparaison.")

    return lines


# =============================================================================
# ETAT LANGGRAPH
# =============================================================================


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

    question: str
    standalone_question: str
    conversation_context: str

    route: dict[str, Any]
    direct_answer: str

    portfolio_context: str
    filter_context: str
    date_context: str

    sql_plan: str
    sql: str
    sql_result: str

    judge: dict[str, Any]
    attempts: int

    context_notes: Annotated[list[str], list_add]


# =============================================================================
# GRAPHE LANGGRAPH
# =============================================================================


def build_graph(llm: Any):
    """Construit le graphe de l'agent text-to-SQL."""

    def _effective_question(state: AgentState) -> str:
        return state.get("standalone_question") or state.get("question", "")

    def initial_think_node(state: AgentState) -> dict[str, Any]:
        default_route = {
            "standalone_question": "",
            "direct_answer": False,
            "direct_answer_text": "",
            "needs_sql": True,
            "needs_portfolio_search": False,
            "needs_filter_extraction": False,
            "needs_date_extraction": False,
            "portfolio_terms": [],
            "filter_terms": [],
            "date_terms": [],
            "reason_short": "",
        }

        prompt = [
            SystemMessage(
                content=INITIAL_THINK_PROMPT.format(
                    schema=SCHEMA_DESCRIPTION,
                    today=date.today().isoformat(),
                )
            ),
        ] + state["messages"]

        route = _invoke_json(llm, prompt, default_route)

        # Normalisation défensive.
        for key in (
            "direct_answer",
            "needs_sql",
            "needs_portfolio_search",
            "needs_filter_extraction",
            "needs_date_extraction",
        ):
            route[key] = bool(route.get(key))

        for key in ("portfolio_terms", "filter_terms", "date_terms"):
            if not isinstance(route.get(key), list):
                route[key] = []

        raw_question = state.get("question", "")
        standalone = str(route.get("standalone_question", "") or "").strip()
        if not standalone:
            standalone = raw_question

        # --- streaming thinking : on expose la décision de routage ---
        branches = _selected_context_branches(route)
        if route.get("direct_answer"):
            decision = "réponse directe (pas de SQL nécessaire)"
        elif branches:
            decision = "contexte à préparer : " + ", ".join(branches)
        else:
            decision = "SQL direct sans contexte supplémentaire"
        reason = route.get("reason_short", "") or "—"
        thinking_text = f"{reason} | Décision : {decision}"
        if standalone and standalone != raw_question:
            thinking_text += f" | Question reformulée : {standalone}"
        _emit_thinking("Analyse de la question", thinking_text)

        return {
            "route": route,
            "standalone_question": standalone,
            "direct_answer": str(route.get("direct_answer_text", "") or ""),
            "context_notes": [f"Routage initial : {route.get('reason_short', '')}"],
        }

    def route_after_initial(state: AgentState) -> str:
        route = state.get("route", {}) or {}

        if route.get("direct_answer"):
            return "direct_response"

        branches = _selected_context_branches(route)
        if branches:
            return "context_start"

        return "plan"

    def direct_response_node(state: AgentState) -> dict[str, Any]:
        prompt = [SystemMessage(content=DIRECT_ANSWER_PROMPT)] + state["messages"]

        try:
            response = llm.invoke(prompt)
            text = str(response.content or "").strip()
        except Exception:
            text = ""

        if not text:
            text = state.get("direct_answer") or (
                "Je peux t'aider à transformer tes questions métier en requêtes SQL "
                "sur la table Impala décrite, puis à reformuler les résultats."
            )

        return {"messages": [AIMessage(content=text)]}

    def context_start_node(state: AgentState) -> dict[str, Any]:
        branches = _selected_context_branches(state.get("route", {}) or {})
        _emit_thinking("Préparation du contexte", "Branches : " + ", ".join(branches))
        return {"context_notes": [f"Branches contextuelles : {', '.join(branches)}"]}

    def route_context_branches(state: AgentState) -> list[str]:
        branches = _selected_context_branches(state.get("route", {}) or {})
        return branches

    def search_portfolio_node(state: AgentState) -> dict[str, Any]:
        route = state.get("route", {}) or {}
        terms = route.get("portfolio_terms", []) or []

        # --- règles métier portefeuille (ex. « portefeuille de taux ») ---
        fired_rules = _detect_business_rules(_effective_question(state), "portfolio")
        rule_aliases = _business_rule_aliases("portfolio")
        rule_blocks: list[str] = []
        for rule in fired_rules:
            _emit_thinking("Règle métier appliquée", f"{rule['name']} -> {rule['where']}")
            confirm_sql = f"""
SELECT DISTINCT {rule['column']}, id_portefeuille, type_ptf_cpt_french_l6
FROM {TABLE_NAME}
WHERE {rule['where']}
ORDER BY 1
LIMIT 30
""".strip()
            _emit_tool_call(f"Confirmation règle « {rule['name']} »", confirm_sql)
            confirm_result = _execute_sql_markdown(confirm_sql)
            _emit_tool_result(f"Confirmation règle « {rule['name']} »", confirm_result)
            rule_blocks.append(
                f"""
RÈGLE MÉTIER (à appliquer en priorité) : {rule['name']}
{rule['explanation']}
FILTRE À UTILISER TEL QUEL : {rule['where']}

Portefeuilles correspondants en base (confirmation) :
{confirm_result}
""".strip()
            )

        # Termes non couverts par une règle -> recherche libre classique.
        remaining_terms = [
            str(t) for t in terms
            if _normalize_country_key(str(t)) not in rule_aliases
        ]

        free_text_block = ""
        if remaining_terms:
            where_clause = _like_where(PORTFOLIO_COLUMNS, remaining_terms)
            sql = f"""
SELECT DISTINCT
    {", ".join(PORTFOLIO_COLUMNS)}
FROM {TABLE_NAME}
WHERE {where_clause}
LIMIT 20
""".strip()

            _emit_tool_call("Recherche portefeuille", sql)
            result = _execute_sql_markdown(sql)
            _emit_tool_result("Recherche portefeuille", result)

            sample_blocks: list[str] = []
            if _is_empty_result(result):
                _emit_thinking(
                    "Recherche portefeuille",
                    "Aucune correspondance : remontée d'échantillons de libellés portefeuille.",
                )
                for col in (
                    "classification_portefeuille_l7",
                    "type_ptf_cpt_french_l6",
                    "code_niveau_6",
                    "code_niveau_7",
                ):
                    sample = _distinct_sample(col, limit=30)
                    sample_blocks.append(f"Échantillon de valeurs pour {col} :\n{sample}")

            free_text_block = f"""
Termes portefeuille détectés : {remaining_terms}

SQL de recherche portefeuille :
{sql}

Résultat de recherche portefeuille :
{result}

{chr(10).join(sample_blocks)}
""".strip()

        # Cas où il n'y a ni règle ni terme exploitable.
        if not rule_blocks and not free_text_block:
            context = (
                "La question parle de portefeuille, mais aucun libellé ou identifiant "
                "de portefeuille précis n'a été détecté. "
                f"Colonnes portefeuille disponibles : {', '.join(PORTFOLIO_COLUMNS)}."
            )
            _emit_thinking(
                "Recherche portefeuille",
                "Aucun terme portefeuille précis détecté, recherche non exécutée.",
            )
            return {
                "portfolio_context": context,
                "context_notes": ["Contexte portefeuille sans recherche de valeur précise."],
            }

        parts = rule_blocks + ([free_text_block] if free_text_block else [])
        context = (chr(10) + chr(10)).join(parts)

        return {
            "portfolio_context": context,
            "context_notes": ["Recherche portefeuille exécutée."],
        }

    def get_filter_node(state: AgentState) -> dict[str, Any]:
        default_filters = {
            "filters": [],
            "notes": "",
        }

        prompt = [
            SystemMessage(
                content=GET_FILTER_PROMPT.format(
                    schema=SCHEMA_DESCRIPTION,
                    filterable_columns=", ".join(FILTERABLE_COLUMNS),
                    standalone_question=_effective_question(state),
                )
            )
        ] + state["messages"]

        parsed = _invoke_json(llm, prompt, default_filters)
        filters = parsed.get("filters", []) or []
        if not isinstance(filters, list):
            filters = []

        # --- streaming thinking : filtres extraits ---
        if filters:
            summary = ", ".join(
                f"{f.get('column', '?')} {f.get('operator', '=')} {f.get('value', '')}"
                for f in filters
                if isinstance(f, dict)
            )
        else:
            summary = "aucun filtre métier détecté"
        _emit_thinking("Extraction des filtres", summary)

        # --- règles métier déterministes (ex. « banques ») ---
        fired_rules = _detect_business_rules(_effective_question(state), "filter")
        rule_aliases = _business_rule_aliases("filter")
        rule_blocks: list[str] = []
        for rule in fired_rules:
            _emit_thinking(
                "Règle métier appliquée",
                f"{rule['name']} -> {rule['where']}",
            )
            rule_blocks.append(
                f"""
RÈGLE MÉTIER (à appliquer en priorité) : {rule['name']}
{rule['explanation']}
FILTRE À UTILISER TEL QUEL : {rule['where']}
""".strip()
            )

        lookup_blocks: list[str] = []
        lookups = 0

        for item in filters[:6]:
            if not isinstance(item, dict):
                continue
            if lookups >= MAX_LOOKUPS_PER_NODE:
                lookup_blocks.append("(Nombre max de vérifications de valeurs atteint.)")
                break

            column = str(item.get("column", "")).strip()
            value = item.get("value")

            if column not in FILTERABLE_COLUMNS:
                continue

            if value is None or str(value).strip() == "":
                continue

            value_str = str(value).strip()
            alternatives = item.get("alternatives", []) or []
            if not isinstance(alternatives, list):
                alternatives = []
            alternatives = [str(a).strip() for a in alternatives if str(a).strip()][:3]

            # Si ce filtre est déjà couvert par une règle métier déclenchée
            # (ex. « banque » -> secteur bancaire), on ne fait pas de recherche
            # texte libre : la règle fournit déjà le filtre exact.
            if rule_blocks and _normalize_country_key(value_str) in rule_aliases:
                continue

            normalized_value = _normalize_country_key(value_str)
            is_country_col = column in COUNTRY_COLUMNS

            # -------------------------------------------------------------
            # 1) GROUPE DE PAYS (Union européenne, zone euro...)
            #    -> filtre par codes ISO3, le plus fiable. On confirme juste
            #       que ces codes existent bien en base.
            # -------------------------------------------------------------
            if is_country_col and normalized_value in COUNTRY_GROUP_TERMS:
                lookups += 1
                confirm_sql = f"""
SELECT DISTINCT cod_pays_implant, libelle_pays_emetteur
FROM {TABLE_NAME}
WHERE cod_pays_implant IN {EU_IN_ISO3}
ORDER BY 1
LIMIT 40
""".strip()
                _emit_tool_call("Confirmation pays UE (codes ISO3)", confirm_sql)
                confirm_result = _execute_sql_markdown(confirm_sql)
                _emit_tool_result("Confirmation pays UE (codes ISO3)", confirm_result)

                lookup_blocks.append(
                    f"""
Filtre candidat (groupe de pays : Union européenne) :
- valeur demandée : {value_str}
- FILTRE À UTILISER (le plus fiable) : cod_pays_implant IN {EU_IN_ISO3}
- Équivalent par libellés français exacts :
  libelle_pays_emetteur IN {EU_IN_NAMES_FR}

Codes UE réellement présents en base (confirmation) :
{confirm_result}
""".strip()
                )
                continue

            # -------------------------------------------------------------
            # 2) PAYS UNIQUE résoluble -> code ISO3 + nom français exact.
            # -------------------------------------------------------------
            resolved_country = _resolve_country_term(value_str)
            if resolved_country is None:
                for alt in alternatives:
                    resolved_country = _resolve_country_term(alt)
                    if resolved_country is not None:
                        break

            if is_country_col and resolved_country is not None:
                fr_name, iso3 = resolved_country
                where_snippet = _country_where_snippet(fr_name, iso3)
                lookups += 1
                confirm_sql = f"""
SELECT cod_pays_implant, libelle_pays_emetteur, COUNT(*) AS nb_lignes
FROM {TABLE_NAME}
WHERE {where_snippet}
GROUP BY cod_pays_implant, libelle_pays_emetteur
ORDER BY nb_lignes DESC
LIMIT 10
""".strip()
                _emit_tool_call(f"Confirmation pays « {fr_name} » ({iso3})", confirm_sql)
                confirm_result = _execute_sql_markdown(confirm_sql)
                _emit_tool_result(f"Confirmation pays « {fr_name} » ({iso3})", confirm_result)

                lookup_blocks.append(
                    f"""
Filtre candidat (pays) :
- valeur demandée : {value_str}
- pays reconnu : {fr_name} (code ISO3 : {iso3})
- FILTRE À UTILISER : {where_snippet}
  (de préférence cod_pays_implant = '{iso3}')

Présence en base (confirmation) :
{confirm_result}
""".strip()
                )
                continue

            # -------------------------------------------------------------
            # 2bis) SECTEUR -> recherche EN PRIORITÉ dans la colonne principale
            #       (secteur_eco_bics_n3_libelle), fallback sur les autres
            #       colonnes secteur si rien n'est trouvé.
            # -------------------------------------------------------------
            if column in SECTOR_COLUMNS:
                search_terms = [value_str] + alternatives

                # 1) recherche dans la colonne principale.
                main_where = _like_where([SECTOR_MAIN_COLUMN], search_terms, max_terms=4)
                main_sql = f"""
SELECT DISTINCT {SECTOR_MAIN_COLUMN}
FROM {TABLE_NAME}
WHERE {main_where}
ORDER BY 1
LIMIT 20
""".strip()
                _emit_tool_call(f"Secteur (colonne principale) « {value_str} »", main_sql)
                main_result = _execute_sql_markdown(main_sql)
                _emit_tool_result(f"Secteur (colonne principale) « {value_str} »", main_result)
                lookups += 1

                fallback_block = ""
                if _is_empty_result(main_result) and lookups < MAX_LOOKUPS_PER_NODE:
                    # 2) fallback : chercher la valeur dans les colonnes secteur
                    #    secondaires pour aider à localiser le bon libellé.
                    other_cols = [c for c in SECTOR_COLUMNS if c != SECTOR_MAIN_COLUMN]
                    other_where = _like_where(other_cols, search_terms, max_terms=4)
                    select_cols = ", ".join(SECTOR_COLUMNS)
                    fb_sql = f"""
SELECT DISTINCT {select_cols}
FROM {TABLE_NAME}
WHERE {other_where}
LIMIT 20
""".strip()
                    _emit_tool_call(f"Secteur (colonnes secondaires) « {value_str} »", fb_sql)
                    fb_result = _execute_sql_markdown(fb_sql)
                    _emit_tool_result(f"Secteur (colonnes secondaires) « {value_str} »", fb_result)
                    lookups += 1
                    fallback_block = (
                        f"\n\nNon trouvé dans {SECTOR_MAIN_COLUMN}. Occurrences dans les "
                        f"autres colonnes secteur (pour identifier le bon libellé) :\n{fb_result}"
                    )

                lookup_blocks.append(
                    f"""
Filtre candidat (SECTEUR) :
- valeur demandée : {value_str}
- variantes testées : {search_terms}
- COLONNE À UTILISER EN PRIORITÉ : {SECTOR_MAIN_COLUMN}

Valeurs de {SECTOR_MAIN_COLUMN} correspondantes :
{main_result}{fallback_block}

Consigne : filtrer sur {SECTOR_MAIN_COLUMN} avec le libellé EXACT confirmé
ci-dessus (=/IN), sinon LOWER({SECTOR_MAIN_COLUMN}) LIKE '%...%' avec les
variantes. N'utiliser une colonne secteur secondaire que si la valeur ne figure
que là.
""".strip()
                )
                continue

            # -------------------------------------------------------------
            # 3) CAS GÉNÉRAL : recherche approximative avec variantes.
            # -------------------------------------------------------------
            search_terms = [value_str] + alternatives
            where_clause = _like_where([column], search_terms, max_terms=4)

            lookup_sql = f"""
SELECT DISTINCT
    {column}
FROM {TABLE_NAME}
WHERE {where_clause}
LIMIT 20
""".strip()

            _emit_tool_call(f"Vérif. valeurs « {column} »", lookup_sql)
            lookup_result = _execute_sql_markdown(lookup_sql)
            _emit_tool_result(f"Vérif. valeurs « {column} »", lookup_result)
            lookups += 1

            sample_block = ""
            if _is_empty_result(lookup_result) and lookups < MAX_LOOKUPS_PER_NODE:
                # Aucune correspondance même avec variantes/synonymes :
                # on montre le vocabulaire réel de la colonne.
                lookups += 1
                sample = _distinct_sample(column)
                sample_block = (
                    f"\n\nAucune correspondance directe. Échantillon des valeurs "
                    f"réelles de {column} (choisir le libellé le plus proche) :\n{sample}"
                )

            lookup_blocks.append(
                f"""
Filtre candidat :
- colonne : {column}
- opérateur : {item.get("operator", "=")}
- valeur demandée : {value_str}
- variantes testées : {search_terms}
- confiance : {item.get("confidence", "")}

Valeurs proches trouvées :
{lookup_result}{sample_block}
""".strip()
            )

        context = f"""
{(chr(10) + chr(10)).join(rule_blocks) + chr(10) + chr(10) if rule_blocks else ""}Filtres extraits :
{json.dumps(filters, ensure_ascii=False, indent=2)}

Notes :
{parsed.get("notes", "")}

Vérification de valeurs :
{chr(10).join(lookup_blocks) if lookup_blocks else "Aucune vérification de valeur exécutée."}

Rappel : dans le SQL final, appliquer d'abord les RÈGLES MÉTIER ci-dessus si
présentes, puis utiliser les libellés EXACTS confirmés (=/IN), sinon LIKE avec
les variantes.
""".strip()

        return {
            "filter_context": context,
            "context_notes": ["Extraction des filtres métier effectuée."],
        }

    def get_date_node(state: AgentState) -> dict[str, Any]:
        default_dates = {
            "current_date_override": "",
            "date_filters": [],
            "notes": "",
        }

        prompt = [
            SystemMessage(
                content=GET_DATE_PROMPT.format(
                    schema=SCHEMA_DESCRIPTION,
                    today=date.today().isoformat(),
                    date_columns=", ".join(DATE_COLUMNS),
                    standalone_question=_effective_question(state),
                )
            )
        ] + state["messages"]

        parsed = _invoke_json(llm, prompt, default_dates)
        date_filters = parsed.get("date_filters", []) or []
        if not isinstance(date_filters, list):
            date_filters = []

        # --- streaming thinking : contraintes de date ---
        if date_filters:
            summary = ", ".join(
                f"{d.get('column', '?')}:{d.get('kind', '?')}"
                for d in date_filters
                if isinstance(d, dict)
            )
        else:
            summary = "aucune contrainte de date détectée"
        _emit_thinking("Extraction des dates", summary)

        override = str(parsed.get("current_date_override", "") or "").strip()

        # --- résolution contre les dates réellement en base ---
        resolved_lines = _resolve_date_filters(date_filters)

        header_lines: list[str] = []
        if override:
            header_lines.append(
                f"Date du jour déclarée par l'utilisateur : {override} "
                "(les expressions relatives ont été calculées à partir d'elle)."
            )

        context = f"""
{chr(10).join(header_lines)}

Contraintes de date extraites :
{json.dumps(date_filters, ensure_ascii=False, indent=2)}

Notes :
{parsed.get("notes", "")}

DATES RÉSOLUES (réellement disponibles en base) :
{chr(10).join(resolved_lines) if resolved_lines else "Aucune résolution de date exécutée."}

Consigne impérative : le SQL final doit utiliser EXACTEMENT ces dates résolues
(au format 'YYYY-MM-DD'), jamais les dates brutes de la question. Si la date
utilisée diffère de la date demandée, la réponse finale doit le mentionner.
""".strip()

        return {
            "date_context": context,
            "context_notes": ["Extraction et résolution des contraintes de date effectuées."],
        }

    def plan_node(state: AgentState) -> dict[str, Any]:
        # Section streamée token par token (mode "messages") via process_stream.
        _emit_section("Planification SQL")

        prompt = [
            SystemMessage(
                content=PLAN_PROMPT.format(
                    schema=SCHEMA_DESCRIPTION,
                    conversation_context=state.get("conversation_context", ""),
                    question=_effective_question(state),
                    portfolio_context=state.get("portfolio_context", ""),
                    filter_context=state.get("filter_context", ""),
                    date_context=state.get("date_context", ""),
                )
            )
        ]

        response = llm.invoke(prompt)

        return {
            "sql_plan": response.content or "",
            "context_notes": ["Plan SQL produit."],
        }

    def generate_sql_node(state: AgentState) -> dict[str, Any]:
        attempt = state.get("attempts", 0) + 1
        judge = state.get("judge", {}) or {}

        if attempt > 1 and judge.get("fix_instruction"):
            _emit_thinking(
                f"Génération SQL — tentative {attempt}",
                "Correction : " + str(judge.get("fix_instruction", ""))[:300],
            )
        else:
            _emit_thinking(f"Génération SQL — tentative {attempt}", "Construction de la requête.")

        prompt = [
            SystemMessage(
                content=GENERATE_SQL_PROMPT.format(
                    schema=SCHEMA_DESCRIPTION,
                    conversation_context=state.get("conversation_context", ""),
                    question=_effective_question(state),
                    sql_plan=state.get("sql_plan", ""),
                    portfolio_context=state.get("portfolio_context", ""),
                    filter_context=state.get("filter_context", ""),
                    date_context=state.get("date_context", ""),
                    attempt=attempt,
                    previous_sql=state.get("sql", ""),
                    previous_result=state.get("sql_result", ""),
                    fix_instruction=judge.get("fix_instruction", ""),
                    table_name=TABLE_NAME,
                    date_expr_hint=_date_expr("date_valorisation"),
                )
            )
        ]

        response = llm.invoke(prompt)
        sql = _extract_sql(response.content or "")

        return {
            "sql": sql,
            "attempts": attempt,
            "context_notes": [f"SQL généré, tentative {attempt}."],
        }

    def execute_sql_node(state: AgentState) -> dict[str, Any]:
        sql = state.get("sql", "")

        if not sql:
            result = "Erreur : aucune requête SQL générée."
            _emit_tool_result("Exécution SQL", result)
        else:
            _emit_tool_call("Exécution SQL", sql)
            result = _execute_sql_markdown(sql)
            _emit_tool_result("Exécution SQL", result)

        return {
            "sql_result": result,
            "context_notes": ["SQL exécuté."],
        }

    def judge_result_node(state: AgentState) -> dict[str, Any]:
        sql_result = state.get("sql_result", "")

        if sql_result.startswith("Erreur"):
            judge = {
                "is_sufficient": False,
                "reason": "Erreur SQL détectée.",
                "fix_instruction": sql_result[:2000],
            }
            _emit_thinking("Contrôle qualité", "Insuffisant : erreur SQL détectée.")
            return {
                "judge": judge,
                "context_notes": ["Résultat SQL jugé insuffisant à cause d'une erreur."],
            }

        default_judge = {
            "is_sufficient": True,
            "reason": "",
            "fix_instruction": "",
        }

        prompt = [
            SystemMessage(
                content=JUDGE_PROMPT.format(
                    question=_effective_question(state),
                    date_context=state.get("date_context", ""),
                    filter_context=state.get("filter_context", ""),
                    sql_plan=state.get("sql_plan", ""),
                    sql=state.get("sql", ""),
                    sql_result=state.get("sql_result", ""),
                    attempt=state.get("attempts", 0),
                    max_attempts=MAX_SQL_ATTEMPTS,
                )
            )
        ]

        judge = _invoke_json(llm, prompt, default_judge)
        judge["is_sufficient"] = bool(judge.get("is_sufficient"))

        if judge["is_sufficient"]:
            _emit_thinking("Contrôle qualité", "Résultat suffisant pour répondre.")
        else:
            _emit_thinking(
                "Contrôle qualité",
                "Insuffisant : " + str(judge.get("reason", "") or judge.get("fix_instruction", ""))[:300],
            )

        return {
            "judge": judge,
            "context_notes": [f"Contrôle qualité : {judge.get('reason', '')}"],
        }

    def route_after_judge(state: AgentState) -> str:
        judge = state.get("judge", {}) or {}

        if judge.get("is_sufficient"):
            return "formulate_answer"

        if state.get("attempts", 0) >= MAX_SQL_ATTEMPTS:
            return "formulate_answer"

        return "generate_sql"

    def formulate_answer_node(state: AgentState) -> dict[str, Any]:
        # Section streamée token par token (mode "messages") via process_stream.
        _emit_section("Réponse")

        prompt = [
            SystemMessage(
                content=FORMULATE_PROMPT.format(
                    question=_effective_question(state),
                    date_context=state.get("date_context", ""),
                    sql_plan=state.get("sql_plan", ""),
                    sql=state.get("sql", ""),
                    sql_result=state.get("sql_result", ""),
                    judge=json.dumps(state.get("judge", {}), ensure_ascii=False, indent=2),
                )
            )
        ]

        response = llm.invoke(prompt)
        text = response.content or "Je n'ai pas pu produire de réponse exploitable."

        return {"messages": [AIMessage(content=text)]}

    builder = StateGraph(AgentState)

    builder.add_node("initial_think", initial_think_node)
    builder.add_node("direct_response", direct_response_node)
    builder.add_node("context_start", context_start_node)

    builder.add_node("search_portfolio", search_portfolio_node)
    builder.add_node("get_filter", get_filter_node)
    builder.add_node("get_date", get_date_node)

    builder.add_node("plan", plan_node)
    builder.add_node("generate_sql", generate_sql_node)
    builder.add_node("execute_sql", execute_sql_node)
    builder.add_node("judge_result", judge_result_node)
    builder.add_node("formulate_answer", formulate_answer_node)

    builder.add_edge(START, "initial_think")

    builder.add_conditional_edges(
        "initial_think",
        route_after_initial,
        {
            "direct_response": "direct_response",
            "context_start": "context_start",
            "plan": "plan",
        },
    )

    builder.add_edge("direct_response", END)

    # Fan-out parallèle puis jonction sur plan.
    # `then="plan"` fait office de barrière : le noeud plan attend la fin des
    # branches sélectionnées par route_context_branches.
    builder.add_conditional_edges(
        "context_start",
        route_context_branches,
        {
            "search_portfolio": "search_portfolio",
            "get_filter": "get_filter",
            "get_date": "get_date",
        },
        then="plan",
    )

    builder.add_edge("plan", "generate_sql")
    builder.add_edge("generate_sql", "execute_sql")
    builder.add_edge("execute_sql", "judge_result")

    builder.add_conditional_edges(
        "judge_result",
        route_after_judge,
        {
            "generate_sql": "generate_sql",
            "formulate_answer": "formulate_answer",
        },
    )

    builder.add_edge("formulate_answer", END)

    return builder.compile()


# =============================================================================
# CODE AGENT DATAIKU
# =============================================================================


class MyLLM(BaseLLM):
    # Noeuds dont les tokens LLM sont streamés en direct (mode "messages").
    # direct_response est désormais rédigé par le LLM : on le streame aussi.
    # Les autres noeuds ne produisent que du JSON : on ne streame pas leurs tokens.
    STREAM_TOKEN_NODES = {"plan", "formulate_answer", "direct_response"}

    def __init__(self):
        client = dataiku.api_client()
        project = client.get_default_project()

        dku_llm = project.get_llm(LLM_ID)
        base_llm = dku_llm.as_langchain_chat_model()

        # Température basse recommandée pour du SQL reproductible.
        # On n'utilise pas bind_tools : les actions sont des noeuds explicites.
        self.llm = base_llm.bind(temperature=TEMPERATURE)

        self.graph = build_graph(self.llm)

    @staticmethod
    def _to_langchain_messages(query: dict[str, Any]) -> list[AnyMessage]:
        """Convertit les messages Dataiku LLM Mesh vers LangChain.

        On garde seulement user/assistant pour éviter qu'un message externe
        injecte un SystemMessage non maîtrisé dans le graphe.
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
        messages = self._to_langchain_messages(query)
        question = _last_user_question(messages)

        return {
            "messages": messages,
            "question": question,
            "standalone_question": "",
            "conversation_context": _render_recent_conversation(messages),
            "route": {},
            "direct_answer": "",
            "portfolio_context": "",
            "filter_context": "",
            "date_context": "",
            "sql_plan": "",
            "sql": "",
            "sql_result": "",
            "judge": {},
            "attempts": 0,
            "context_notes": [],
        }

    @staticmethod
    def _final_text_from_state(final_state: dict[str, Any]) -> str:
        messages = final_state.get("messages", []) or []

        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return str(msg.content)

        return (
            "Je n'ai pas pu produire de réponse finale exploitable. "
            "Vérifie les logs de l'agent, les droits SQL et la configuration LLM."
        )

    # -------------------------------------------------------------------------
    # Helpers streaming
    # -------------------------------------------------------------------------
    @staticmethod
    def _coerce_token(content: Any) -> str:
        """Normalise le contenu d'un message chunk en texte.

        Certains modèles renvoient une liste de blocs (content blocks) plutôt
        qu'une chaîne ; on concatène uniquement les morceaux texte.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(str(block.get("text", "")))
                elif isinstance(block, str):
                    parts.append(block)
            return "".join(parts)
        return str(content or "")

    @staticmethod
    def _render_custom_event(event: Any) -> Iterator[dict[str, Any]]:
        """Transforme un évènement custom de noeud en chunks Dataiku affichables."""
        if not isinstance(event, dict):
            return

        etype = event.get("type")

        if etype == "section":
            title = str(event.get("title", "") or "")
            yield {"chunk": {"text": f"\n\n---\n\n### {title}\n\n"}}

        elif etype == "thinking":
            title = str(event.get("title", "Réflexion") or "Réflexion")
            text = str(event.get("text", "") or "")
            yield {"chunk": {"text": f"\n\n> 🧠 **{title}** — {text}"}}

        elif etype == "tool_call":
            name = str(event.get("name", "Outil") or "Outil")
            sql = str(event.get("sql", "") or "")
            yield {"chunk": {"text": f"\n\n> 🔧 **Outil : {name}**\n\n```sql\n{sql}\n```\n"}}

        elif etype == "tool_result":
            name = str(event.get("name", "Outil") or "Outil")
            result = str(event.get("result", "") or "")
            if len(result) > TOOL_RESULT_PREVIEW_CHARS:
                result = result[:TOOL_RESULT_PREVIEW_CHARS] + "\n…(aperçu tronqué)"
            yield {"chunk": {"text": f"\n\n_Résultat ({name}) :_\n\n{result}\n"}}

    # -------------------------------------------------------------------------
    # Mode non-streaming
    # -------------------------------------------------------------------------
    def process(self, query, settings, trace):
        state = self._initial_state(query)

        try:
            final_state = self.graph.invoke(state)
        except Exception as exc:
            return {
                "text": (
                    "Une erreur est survenue pendant le traitement de l'agent. "
                    f"Détail technique : {exc}"
                )
            }

        return {"text": self._final_text_from_state(final_state)}

    # -------------------------------------------------------------------------
    # Mode streaming (tools + thinking + réponse en direct)
    # -------------------------------------------------------------------------
    def process_stream(self, query, settings, trace):
        state = self._initial_state(query)
        final_text = ""
        answer_started = False

        try:
            for mode, data in self.graph.stream(
                state, stream_mode=["updates", "messages", "custom"]
            ):
                # --- Évènements custom : tools + thinking + titres de section ---
                if mode == "custom":
                    for chunk in self._render_custom_event(data):
                        yield chunk

                # --- Tokens LLM : plan + réponse finale + réponse directe ---
                elif mode == "messages":
                    message_chunk, metadata = data
                    node = (metadata or {}).get("langgraph_node")

                    if node not in self.STREAM_TOKEN_NODES:
                        # initial_think / get_filter / get_date / judge -> JSON,
                        # déjà résumés via les évènements custom : on ignore.
                        continue

                    token = self._coerce_token(getattr(message_chunk, "content", ""))
                    if not token:
                        continue

                    # Les noeuds qui produisent la réponse utilisateur finale.
                    if node in ("formulate_answer", "direct_response"):
                        answer_started = True
                        final_text += token

                    yield {"chunk": {"text": token}}

                # --- Updates : filet de sécurité si le modèle ne streame pas ---
                elif mode == "updates":
                    for node, update in data.items():
                        if not isinstance(update, dict):
                            continue

                        if node in ("direct_response", "formulate_answer"):
                            # Si aucun token n'a été streamé pour la réponse
                            # (modèle sans support streaming), on émet le bloc final.
                            msgs = update.get("messages", []) or []
                            if msgs and isinstance(msgs[-1], AIMessage):
                                txt = str(msgs[-1].content or "")
                                if not answer_started and txt:
                                    final_text = txt
                                    yield {"chunk": {"text": txt}}

            if not final_text:
                yield {
                    "chunk": {
                        "text": (
                            "\n\nJe n'ai pas pu produire de réponse finale exploitable. "
                            "Vérifie les logs de l'agent, les droits SQL et la configuration LLM."
                        )
                    }
                }

        except Exception as exc:
            yield {
                "chunk": {
                    "text": (
                        "\n\nUne erreur est survenue pendant le traitement de l'agent. "
                        f"Détail technique : {exc}"
                    )
                }
            }
