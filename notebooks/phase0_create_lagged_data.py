#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PHASE 0 — CRÉATION DES DONNÉES LAGGÉES
Stage ABIL 2025 - Jirince BIABA KUYA

OBJECTIF :
Créer deux bases laggées pour imposer une structure temporelle cohérente :
    - Treatment/confounders en t-1
    - Outcome en t

FICHIERS CRÉÉS :
1. data_transformed_unscaled_lagged.csv
   -> Pour causal discovery (Granger / PCMCI / DAG)
   -> Les noms laggés sont conservés explicitement

2. data_prepared_for_dml_lagged.csv
   -> Pour DML
   -> Les noms laggés sont renommés vers les noms usuels
      (mais représentent bien les valeurs en t-1)

STRUCTURE TEMPORELLE :
- Capital_Formation_lag1 = Capital_Formation(t-1)
- Inflation_lag1 = Inflation(t-1)
- Exchange_Rate_lag1 = Exchange_Rate(t-1)
- Government_Debt_lag1 = Government_Debt(t-1)
- Trade_Balance_lag1 = Trade_Balance(t-1)
- Reserves_lag1 = Reserves(t-1)
- GDP_Growth = GDP_Growth(t)

CONSÉQUENCE :
Cette structure réduit fortement le risque de causalité inverse contemporaine
entre Capital_Formation et GDP_Growth, et rend l’interprétation causale plus cohérente.
"""

import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

print("=" * 100)
print(" " * 28 + "PHASE 0 — CRÉATION DES DONNÉES LAGGÉES")
print("=" * 100)
print()

# ======================================================================================
# CONFIGURATION
# ======================================================================================

DATA_PROCESSED = Path("../data/processed")

INPUT_UNSCALED = DATA_PROCESSED / "data_transformed_unscaled.csv"
INPUT_SCALED = DATA_PROCESSED / "data_prepared_for_dml.csv"

OUTPUT_UNSCALED = DATA_PROCESSED / "data_transformed_unscaled_lagged.csv"
OUTPUT_SCALED = DATA_PROCESSED / "data_prepared_for_dml_lagged.csv"

REPORT_FILE = Path("./phase0_lagged_data_report.txt")

print("Configuration :")
print(f"  Input unscaled : {INPUT_UNSCALED}")
print(f"  Input scaled   : {INPUT_SCALED}")
print(f"  Output unscaled: {OUTPUT_UNSCALED}")
print(f"  Output scaled  : {OUTPUT_SCALED}")
print(f"  Rapport        : {REPORT_FILE}")
print()

# ======================================================================================
# VARIABLES
# ======================================================================================

VARS_TO_LAG = [
    "Capital_Formation",
    "Inflation",
    "Exchange_Rate",
    "Government_Debt",
    "Trade_Balance",
    "Reserves",
]

OUTCOME_VAR = "GDP_Growth"
ID_COLS = ["Country", "Year"]

print("Plan de transformation :")
print(f"  Variables en t-1 : {VARS_TO_LAG}")
print(f"  Variable en t    : {OUTCOME_VAR}")
print()

# ======================================================================================
# HELPERS
# ======================================================================================

def check_required_columns(df, dataset_name):
    required = set(ID_COLS + [OUTCOME_VAR])
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(f"[{dataset_name}] Colonnes obligatoires manquantes: {missing_required}")

    missing_lag_vars = [v for v in VARS_TO_LAG if v not in df.columns]
    if missing_lag_vars:
        print(f"⚠️ [{dataset_name}] Variables à lagger manquantes (ignorées): {missing_lag_vars}")

    present_lag_vars = [v for v in VARS_TO_LAG if v in df.columns]
    return present_lag_vars


def create_lagged_base(df_input: pd.DataFrame, dataset_name: str, keep_explicit_lag_names: bool):
    """
    Crée une base laggée par pays.

    Args:
        df_input: DataFrame source
        dataset_name: nom pour affichage
        keep_explicit_lag_names:
            - True  -> garde Capital_Formation_lag1, Inflation_lag1, ...
            - False -> renomme Capital_Formation_lag1 -> Capital_Formation, etc.

    Returns:
        DataFrame transformé
    """
    print("=" * 100)
    print(f"CRÉATION DES LAGS — {dataset_name}")
    print("=" * 100)

    df = df_input.copy()
    present_lag_vars = check_required_columns(df, dataset_name)

    n_obs_before = len(df)
    n_countries_before = df["Country"].nunique()

    print(f"Dataset chargé : {n_obs_before} observations")
    print(f"  Pays    : {n_countries_before}")
    print(f"  Période : {df['Year'].min()}–{df['Year'].max()}")
    print()

    # Tri pour sécurité
    df = df.sort_values(["Country", "Year"]).reset_index(drop=True)

    # Création des colonnes laggées par pays
    print("Création des lags par pays :")
    for var in present_lag_vars:
        lag_col = f"{var}_lag1"
        df[lag_col] = df.groupby("Country")[var].shift(1)
        print(f"  ✅ {lag_col} créé")
    print()

    # On supprime les versions contemporaines du traitement/confounders
    cols_to_drop = [v for v in present_lag_vars if v in df.columns]
    if cols_to_drop:
        print("Suppression des colonnes contemporaines du traitement/confounders :")
        for c in cols_to_drop:
            print(f"  - {c}")
        df = df.drop(columns=cols_to_drop)
        print()

    # Si version DML -> renommage lag1 vers noms usuels
    if not keep_explicit_lag_names:
        rename_map = {f"{v}_lag1": v for v in present_lag_vars if f"{v}_lag1" in df.columns}
        print("Renommage des variables laggées (version DML) :")
        for old, new in rename_map.items():
            print(f"  {old} -> {new}")
        df = df.rename(columns=rename_map)
        print()
    else:
        print("Version causal discovery : les noms laggés sont conservés explicitement.")
        print()

    # Suppression des lignes avec NaN (première année par pays)
    before_dropna = len(df)
    countries_after_build = df["Country"].nunique()

    df = df.dropna().reset_index(drop=True)

    after_dropna = len(df)
    lost_obs = before_dropna - after_dropna

    print("Nettoyage des NaN :")
    print(f"  Avant  : {before_dropna}")
    print(f"  Après  : {after_dropna}")
    print(f"  Perdues: {lost_obs} (= {countries_after_build} pays × 1 année)")
    print()

    print("Vérification finale :")
    print(f"  Colonnes : {list(df.columns)}")
    print(f"  Période  : {df['Year'].min()}–{df['Year'].max()}")
    print(f"  Shape    : {df.shape}")
    print()

    return df


# ======================================================================================
# EXECUTION
# ======================================================================================

# ------------------------------------------------------------------
# 1) Base unscaled laggée pour causal discovery
# ------------------------------------------------------------------
print("=" * 100)
print("ÉTAPE 1 — GÉNÉRATION DE data_transformed_unscaled_lagged.csv")
print("=" * 100)

df_unscaled = pd.read_csv(INPUT_UNSCALED)
unscaled_before = len(df_unscaled)

df_unscaled_lagged = create_lagged_base(
    df_input=df_unscaled,
    dataset_name="UNSCALED / CAUSAL DISCOVERY",
    keep_explicit_lag_names=True
)
df_unscaled_lagged.to_csv(OUTPUT_UNSCALED, index=False)

print(f"✅ Sauvegardé : {OUTPUT_UNSCALED}")
print()

# ------------------------------------------------------------------
# 2) Base scaled laggée pour DML
# ------------------------------------------------------------------
print("=" * 100)
print("ÉTAPE 2 — GÉNÉRATION DE data_prepared_for_dml_lagged.csv")
print("=" * 100)

df_scaled = pd.read_csv(INPUT_SCALED)
scaled_before = len(df_scaled)

df_scaled_lagged = create_lagged_base(
    df_input=df_scaled,
    dataset_name="SCALED / DML",
    keep_explicit_lag_names=False
)
df_scaled_lagged.to_csv(OUTPUT_SCALED, index=False)

print(f"✅ Sauvegardé : {OUTPUT_SCALED}")
print()

# ======================================================================================
# RAPPORT
# ======================================================================================

lost_unscaled = unscaled_before - len(df_unscaled_lagged)
lost_scaled = scaled_before - len(df_scaled_lagged)

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("=" * 100 + "\n")
    f.write("PHASE 0 — CRÉATION DES DONNÉES LAGGÉES — RAPPORT\n")
    f.write("=" * 100 + "\n\n")

    f.write("OBJECTIF\n")
    f.write("-------\n")
    f.write("Introduire une structure temporelle cohérente entre traitement, confounders et outcome.\n")
    f.write("Le traitement et les confounders sont considérés en t-1, tandis que l’outcome reste en t.\n\n")

    f.write("JUSTIFICATION\n")
    f.write("-------------\n")
    f.write("Cette transformation permet d’éviter la simultanéité contemporaine entre Capital_Formation et GDP_Growth,\n")
    f.write("et rend l’interprétation causale plus cohérente avec la théorie macroéconomique.\n\n")

    f.write("FICHIERS CRÉÉS\n")
    f.write("--------------\n")
    f.write(f"1. {OUTPUT_UNSCALED.name}\n")
    f.write("   Usage : causal discovery (Granger / PCMCI / DAG)\n")
    f.write("   Particularité : les noms laggés sont explicites (_lag1)\n")
    f.write(f"   Observations : {len(df_unscaled_lagged)}\n")
    f.write(f"   Période      : {df_unscaled_lagged['Year'].min()}–{df_unscaled_lagged['Year'].max()}\n")
    f.write(f"   Perte        : {lost_unscaled} observations\n\n")

    f.write(f"2. {OUTPUT_SCALED.name}\n")
    f.write("   Usage : Double Machine Learning (DML)\n")
    f.write("   Particularité : les noms laggés sont renommés avec les noms usuels\n")
    f.write("   (mais correspondent bien à des valeurs en t-1)\n")
    f.write(f"   Observations : {len(df_scaled_lagged)}\n")
    f.write(f"   Période      : {df_scaled_lagged['Year'].min()}–{df_scaled_lagged['Year'].max()}\n")
    f.write(f"   Perte        : {lost_scaled} observations\n\n")

    f.write("STRUCTURE TEMPORELLE RETENUE\n")
    f.write("----------------------------\n")
    f.write("Treatment / confounders en t-1 :\n")
    for v in VARS_TO_LAG:
        f.write(f"  - {v}(t-1)\n")
    f.write(f"\nOutcome en t :\n")
    f.write(f"  - {OUTCOME_VAR}(t)\n\n")

    f.write("MODIFICATIONS À FAIRE ENSUITE\n")
    f.write("-----------------------------\n")
    f.write("Phase 3 (causal discovery)\n")
    f.write("  - phase3A_01_granger.py : utiliser data_transformed_unscaled_lagged.csv\n")
    f.write("  - phase3D_00_prepare_pcmci_data.py : utiliser data_transformed_unscaled_lagged.csv\n")
    f.write("  - revoir ensuite les DAGs à la lumière de la théorie économique\n\n")

    f.write("Phase 4 (DML)\n")
    f.write("  - utiliser data_prepared_for_dml_lagged.csv\n")
    f.write("  - interprétation : effet de Capital_Formation(t-1) sur GDP_Growth(t)\n\n")

    f.write("REMARQUE MÉTHODOLOGIQUE\n")
    f.write("-----------------------\n")
    f.write("Cette structure réduit fortement le risque de causalité inverse contemporaine,\n")
    f.write("mais les DAGs obtenus devront toujours être validés par la théorie économique.\n\n")

    f.write("RÉSUMÉ OBSERVATIONS\n")
    f.write("-------------------\n")
    f.write(f"Base initiale unscaled : {unscaled_before} observations\n")
    f.write(f"Base finale unscaled   : {len(df_unscaled_lagged)} observations\n")
    f.write(f"Base initiale scaled   : {scaled_before} observations\n")
    f.write(f"Base finale scaled     : {len(df_scaled_lagged)} observations\n\n")

    f.write("=" * 100 + "\n")

print(f"✅ Rapport sauvegardé : {REPORT_FILE}")
print()

# ======================================================================================
# RÉSUMÉ FINAL
# ======================================================================================

print("=" * 100)
print(" " * 38 + "RÉSUMÉ FINAL")
print("=" * 100)
print()

print("FICHIERS CRÉÉS :")
print(f"  1. {OUTPUT_UNSCALED} ({len(df_unscaled_lagged)} observations)")
print(f"  2. {OUTPUT_SCALED} ({len(df_scaled_lagged)} observations)")
print(f"  3. {REPORT_FILE}")
print()

print("PÉRIODE :")
print(f"  Avant : 2000–2024 ({scaled_before} obs)")
print(f"  Après : {df_scaled_lagged['Year'].min()}–{df_scaled_lagged['Year'].max()} ({len(df_scaled_lagged)} obs)")
print(f"  Perte : {lost_scaled} observations (1 première année par pays)")
print()

print("STRUCTURE CAUSALE RETENUE :")
print("  Capital_Formation(t-1) -> GDP_Growth(t)")
print("  Confounders(t-1)       -> GDP_Growth(t)")
print()

print("PROCHAINES ÉTAPES :")
print("  1. Modifier Granger et PCMCI pour utiliser la base laggée")
print("  2. Réexécuter la causal discovery")
print("  3. Vérifier les DAGs obtenus")
print("  4. Modifier les scripts DML pour utiliser la base laggée")
print()

print("=" * 100)
print("✅ PHASE 0 TERMINÉE AVEC SUCCÈS")
print("=" * 100)