# ============================================================================
# ANALYSE DE REQUÊTES BUSINESS OBJECTS -> MODÈLE DE DONNÉES  (version notebook)
# ============================================================================
# Les requêtes sont lues depuis un fichier Excel contenant une colonne "SQL"
# (une requête par ligne). Copiez chaque bloc "CELLULE n" dans une cellule
# de votre notebook Jupyter, puis exécutez-les dans l'ordre.
#
# v2 : - nettoyage des messages d'erreur avant export Excel (openpyxl
#        refusait les caractères de contrôle ANSI insérés par sqlglot) ;
#      - découpage de secours pour les cellules contenant plusieurs
#        requêtes concaténées sans ';' (SQL BO multi-flux : "...) SELECT ...").
# ============================================================================

# %% ========================= CELLULE 1 =====================================
# Installation des dépendances (à exécuter une seule fois)
%pip install sqlglot pandas openpyxl

# %% ========================= CELLULE 2 =====================================
# Paramètres — adaptez ces valeurs

FICHIER_EXCEL = "requetes.xlsx"   # chemin de votre fichier Excel
COLONNE_SQL   = "SQL"             # nom de la colonne contenant les requêtes
FEUILLE       = 0                 # nom ou index de l'onglet (0 = premier)

DIALECTE        = "oracle"        # dialecte essayé en premier (repli auto ensuite)
IGNORER_SCHEMA  = False           # True pour fusionner SCHEMA.TABLE et TABLE
RESPECTER_CASSE = False           # False = tout en MAJUSCULES (recommandé)
DOSSIER_SORTIE  = "modele_donnees"

# %% ========================= CELLULE 3 =====================================
# Fonctions d'analyse (rien à modifier)

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import sqlglot
from sqlglot import exp

try:
    display  # défini dans Jupyter
except NameError:
    display = print

# --- Nettoyage spécifique Business Objects / Oracle -------------------------

# @Prompt('...','A',...), @Variable('...'), @Select(...), etc.
RE_FONCTION_BO = re.compile(r"@\w+\s*\((?:[^()']|'[^']*'|\([^()]*\))*\)", re.S)
# Marqueur de jointure externe Oracle :  COL (+)
RE_ORACLE_PLUS = re.compile(r"\(\s*\+\s*\)")
# Colonnes marquées (+) : ALIAS.COL (+)
RE_COL_PLUS = re.compile(
    r"([A-Za-z_][\w$#]*)\s*\.\s*([A-Za-z_][\w$#]*)\s*\(\s*\+\s*\)"
)
# Séquences de style ANSI (couleurs/soulignés) et caractères de contrôle,
# interdits dans les cellules Excel par openpyxl
RE_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def nettoyer_sql(sql):
    """Neutralise les fonctions @xxx(...) de BO et la syntaxe Oracle (+)."""
    sql = RE_ORACLE_PLUS.sub(" ", sql)
    sql = RE_FONCTION_BO.sub("'@BO'", sql)
    return sql


def nettoyer_message(err, longueur_max=300):
    """Rend un message d'erreur inoffensif pour l'affichage et Excel."""
    s = RE_ANSI.sub("", str(err))
    s = RE_CTRL.sub(" ", s).replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", s).strip()[:longueur_max]


def decouper_requetes(texte):
    """Découpe une cellule en requêtes (';' ou ligne de ---- / ====)."""
    requetes = []
    for bloc in re.split(r"(?m)^\s*[-=*]{4,}\s*$", texte):
        for q in re.split(r";\s*(?:\r?\n|$)", bloc):
            q = q.strip().rstrip(";").strip()
            if len(q) > 15 and re.search(r"\bselect\b", q, re.I):
                requetes.append(q)
    return requetes


# --- Analyse SQL ------------------------------------------------------------

DIALECTES_ESSAI = ["oracle", "tsql", "postgres", "mysql", None]


def parser_sql(sql, dialecte=None):
    """Analyse la requête en essayant plusieurs dialectes SQL."""
    essais = []
    if dialecte:
        essais.append(dialecte)
    essais += [d for d in DIALECTES_ESSAI if d != dialecte]

    derniere = None
    for d in essais:
        try:
            arbres = [a for a in sqlglot.parse(sql, read=d) if a is not None]
            if arbres:
                return arbres
        except Exception as err:
            derniere = err
    raise derniere if derniere else ValueError("SQL vide ou inanalysable")


class Modele:
    """Agrégats du modèle de données reconstitué."""

    def __init__(self):
        self.tables = defaultdict(set)                 # table -> {req_ids}
        self.colonnes = defaultdict(set)               # (table, col) -> {req_ids}
        self.jointures = defaultdict(
            lambda: {"reqs": set(), "types": set()}
        )                                              # (ta, ca, tb, cb) -> infos
        self.non_attribuees = defaultdict(set)         # col -> {req_ids}
        self.erreurs = []                              # [(req_id, message)]


def ajouter_jointure(mdl, req_id, ta, ca, tb, cb, type_j):
    if (tb, cb) < (ta, ca):
        ta, ca, tb, cb = tb, cb, ta, ca
    cle = (ta, ca, tb, cb)
    mdl.jointures[cle]["reqs"].add(req_id)
    mdl.jointures[cle]["types"].add(type_j)


def analyser_arbre(arbre, req_id, mdl, ignorer_schema, respecter_casse,
                   cols_plus_brutes):
    """Extrait tables, colonnes et jointures d'un arbre syntaxique."""

    def N(s):
        return s if respecter_casse else s.upper()

    cols_plus = {(N(a), N(c)) for a, c in cols_plus_brutes}

    # Noms "virtuels" (CTE, alias de sous-requêtes) : pas des tables physiques
    ctes = {N(c.alias) for c in arbre.find_all(exp.CTE) if c.alias}
    derivees = {N(s.alias) for s in arbre.find_all(exp.Subquery) if s.alias}
    virtuelles = ctes | derivees

    # 1) Tables physiques + carte alias -> nom complet
    alias_vers_table = {}
    tables_stmt = set()
    for t in arbre.find_all(exp.Table):
        nom = N(t.name)
        if not nom or nom in ctes:
            continue
        schema = N(t.db) if t.db else ""
        complet = nom if (ignorer_schema or not schema) else f"{schema}.{nom}"
        tables_stmt.add(complet)
        mdl.tables[complet].add(req_id)
        alias_vers_table[N(t.alias_or_name)] = complet
        alias_vers_table.setdefault(nom, complet)

    def resoudre(qualif):
        q = N(qualif)
        if q in virtuelles:
            return "__VIRTUELLE__"
        return alias_vers_table.get(q)

    # 2) Colonnes
    for c in arbre.find_all(exp.Column):
        nom_col = c.name
        if not nom_col or nom_col == "*":
            continue
        nom_col = N(nom_col)
        qualif = c.table or ""
        if qualif:
            cible = resoudre(qualif)
            if cible == "__VIRTUELLE__":
                continue
            if cible:
                mdl.colonnes[(cible, nom_col)].add(req_id)
            else:
                mdl.non_attribuees[nom_col].add(req_id)
        elif len(tables_stmt) == 1:
            mdl.colonnes[(next(iter(tables_stmt)), nom_col)].add(req_id)
        else:
            mdl.non_attribuees[nom_col].add(req_id)

    # 3) Égalités colonne = colonne -> jointures
    def collecter_eq(expr, libelle, tester_plus):
        for eq in expr.find_all(exp.EQ):
            g, d = eq.left, eq.right
            if not (isinstance(g, exp.Column) and isinstance(d, exp.Column)):
                continue
            if not (g.table and d.table):
                continue
            if N(g.table) == N(d.table):
                continue  # même occurrence de table : simple filtre
            tg, td = resoudre(g.table), resoudre(d.table)
            if not tg or not td or "__VIRTUELLE__" in (tg, td):
                continue
            lib = libelle
            if tester_plus and ((N(g.table), N(g.name)) in cols_plus
                                or (N(d.table), N(d.name)) in cols_plus):
                lib = "WHERE externe (+)"
            ajouter_jointure(mdl, req_id, tg, N(g.name), td, N(d.name), lib)

    # 3a) Jointures ANSI : JOIN ... ON ...
    for j in arbre.find_all(exp.Join):
        cond = j.args.get("on")
        if cond is None:
            continue
        side = str(j.side or "").strip()
        kind = str(j.kind or "").strip()
        libelle = " ".join(x for x in (side, kind) if x) or "INNER"
        collecter_eq(cond, libelle, tester_plus=False)

    # 3b) Jointures "ancien style" dans le WHERE (typique du SQL BO)
    for w in arbre.find_all(exp.Where):
        collecter_eq(w.this, "WHERE", tester_plus=True)


# --- Diagramme Mermaid ------------------------------------------------------

def nom_mermaid(nom):
    return re.sub(r"\W", "_", nom)


def generer_mermaid(mdl, max_cols=40):
    lignes = [
        "%% Modèle de données reconstitué depuis les requêtes BO",
        "%% Collez ce contenu tel quel sur https://mermaid.live",
        "erDiagram",
    ]
    cols_par_table = defaultdict(list)
    for (t, c), reqs in mdl.colonnes.items():
        cols_par_table[t].append((len(reqs), c))

    for t in sorted(mdl.tables):
        lignes.append(f"    {nom_mermaid(t)} {{")
        cols = sorted(cols_par_table.get(t, []), key=lambda x: (-x[0], x[1]))
        for _, c in cols[:max_cols]:
            lignes.append(f"        string {nom_mermaid(c)}")
        reste = len(cols) - max_cols
        if reste > 0:
            lignes.append(f"        string _plus_{reste}_autres_colonnes")
        lignes.append("    }")

    for (ta, ca, tb, cb), _infos in sorted(mdl.jointures.items()):
        lignes.append(
            f'    {nom_mermaid(ta)} }}o--o{{ {nom_mermaid(tb)} : "{ca} = {cb}"'
        )
    return "\n".join(lignes) + "\n"


print("Fonctions chargées.")

# %% ========================= CELLULE 4 =====================================
# Lecture de l'Excel et analyse des requêtes

# Frontière "...) SELECT ..." : requêtes BO multi-flux concaténées sans ';'
RE_SELECT_COLLE = re.compile(r"(?<=\))\s*(?=SELECT\b)", re.I)


def analyser_ou_secourir(brute, req_id, mdl):
    """Analyse une requête ; en cas d'échec, tente de découper les
    requêtes concaténées sans ';'. Renvoie True si au moins une partie
    a été analysée avec succès."""
    cols_plus = RE_COL_PLUS.findall(brute)
    sql = nettoyer_sql(brute)
    try:
        arbres = parser_sql(sql, DIALECTE)
    except Exception as err:
        fragments = RE_SELECT_COLLE.split(sql)
        if len(fragments) > 1:
            nb_ok_frag = 0
            for k, frag in enumerate(fragments, start=1):
                sous_id = f"{req_id}({k})"
                try:
                    arbres_f = parser_sql(frag, DIALECTE)
                except Exception as err_f:
                    mdl.erreurs.append((sous_id, nettoyer_message(err_f)))
                    continue
                nb_ok_frag += 1
                for a in arbres_f:
                    analyser_arbre(a, sous_id, mdl, IGNORER_SCHEMA,
                                   RESPECTER_CASSE, cols_plus)
            return nb_ok_frag > 0
        mdl.erreurs.append((req_id, nettoyer_message(err)))
        return False
    for a in arbres:
        analyser_arbre(a, req_id, mdl, IGNORER_SCHEMA,
                       RESPECTER_CASSE, cols_plus)
    return True


df_src = pd.read_excel(FICHIER_EXCEL, sheet_name=FEUILLE)
assert COLONNE_SQL in df_src.columns, (
    f"Colonne '{COLONNE_SQL}' absente. Colonnes trouvées : {list(df_src.columns)}"
)

mdl = Modele()
nb_req = nb_ok = 0

for idx, cellule in df_src[COLONNE_SQL].items():
    if not isinstance(cellule, str) or not cellule.strip():
        continue
    ligne_excel = idx + 2  # +1 pour l'en-tête, +1 car l'index commence à 0
    sous_requetes = decouper_requetes(cellule)
    if not sous_requetes and "select" in cellule.lower():
        sous_requetes = [cellule.strip()]
    for j, brute in enumerate(sous_requetes, start=1):
        req_id = (f"L{ligne_excel}" if len(sous_requetes) == 1
                  else f"L{ligne_excel}.{j}")
        nb_req += 1
        if analyser_ou_secourir(brute, req_id, mdl):
            nb_ok += 1

print("Analyse terminée")
print(f"  Requêtes trouvées      : {nb_req}")
print(f"  Analysées avec succès  : {nb_ok}  (échecs : {len(mdl.erreurs)})")
print(f"  Tables détectées       : {len(mdl.tables)}")
print(f"  Couples table/colonne  : {len(mdl.colonnes)}")
print(f"  Jointures distinctes   : {len(mdl.jointures)}")
if mdl.erreurs:
    print("\nRequêtes (ou fragments) non analysés :")
    for rid, msg in mdl.erreurs:
        print(f"  {rid} : {msg}")

# %% ========================= CELLULE 5 =====================================
# Résultats en DataFrames + export vers un fichier Excel multi-onglets

Path(DOSSIER_SORTIE).mkdir(parents=True, exist_ok=True)


def nettoyer_df_pour_excel(df):
    """Supprime tout caractère refusé par openpyxl dans les cellules texte."""
    fn = df.map if hasattr(df, "map") else df.applymap
    return fn(lambda v: nettoyer_message(v, 3000) if isinstance(v, str) else v)


df_tables = pd.DataFrame(
    [[t, len(r), sum(1 for (tt, _c) in mdl.colonnes if tt == t)]
     for t, r in mdl.tables.items()],
    columns=["TABLE", "NB_REQUETES", "NB_COLONNES_VUES"],
).sort_values(["NB_REQUETES", "TABLE"], ascending=[False, True],
              ignore_index=True)

df_colonnes = pd.DataFrame(
    [[t, c, len(r)] for (t, c), r in mdl.colonnes.items()],
    columns=["TABLE", "COLONNE", "NB_REQUETES"],
).sort_values(["TABLE", "NB_REQUETES", "COLONNE"],
              ascending=[True, False, True], ignore_index=True)

df_jointures = pd.DataFrame(
    [[ta, ca, tb, cb, " | ".join(sorted(i["types"])), len(i["reqs"]),
      sorted(i["reqs"])[0]]
     for (ta, ca, tb, cb), i in mdl.jointures.items()],
    columns=["TABLE_A", "COLONNE_A", "TABLE_B", "COLONNE_B",
             "TYPES_JOINTURE", "NB_REQUETES", "EXEMPLE_LIGNE"],
).sort_values("NB_REQUETES", ascending=False, ignore_index=True)

df_erreurs = pd.DataFrame(
    [["ECHEC", rid, msg] for rid, msg in mdl.erreurs]
    + [["AVERTISSEMENT", ", ".join(sorted(r)),
        f"colonne '{c}' sans table identifiable"]
       for c, r in sorted(mdl.non_attribuees.items())],
    columns=["TYPE", "LIGNES_EXCEL", "MESSAGE"],
)

chemin_xlsx = Path(DOSSIER_SORTIE) / "modele_donnees.xlsx"
with pd.ExcelWriter(chemin_xlsx, engine="openpyxl") as w:
    nettoyer_df_pour_excel(df_tables).to_excel(w, sheet_name="Tables", index=False)
    nettoyer_df_pour_excel(df_colonnes).to_excel(w, sheet_name="Colonnes", index=False)
    nettoyer_df_pour_excel(df_jointures).to_excel(w, sheet_name="Jointures", index=False)
    nettoyer_df_pour_excel(df_erreurs).to_excel(w, sheet_name="Erreurs", index=False)
print(f"Modèle exporté : {chemin_xlsx.resolve()}\n")

print("=== JOINTURES (le cœur du modèle) ===")
display(df_jointures.head(40))
print("=== TABLES les plus utilisées ===")
display(df_tables.head(30))

# %% ========================= CELLULE 6 =====================================
# Diagramme entité-relation Mermaid

texte_mermaid = generer_mermaid(mdl)
chemin_mmd = Path(DOSSIER_SORTIE) / "modele.mmd"
chemin_mmd.write_text(texte_mermaid, encoding="utf-8")
print(f"Diagramme écrit : {chemin_mmd.resolve()}")
print("Collez le contenu ci-dessous sur https://mermaid.live :\n")
print(texte_mermaid)
