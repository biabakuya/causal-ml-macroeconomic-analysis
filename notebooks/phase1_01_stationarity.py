#!/usr/bin/env python3
"""
PHASE 1 : PREPARATION DONNEES (VERSION DEFINITIVE + DOC + INTERMEDIAIRES)

Ce script :
1) Charge dataset harmonisé (wide)
2) Applique transformations macro (log / log1p / level)
3) Nettoie (dropna après log)
4) Winsorisation (par pays recommandé) avant standardisation
5) Standardisation Z-score (robuste aux std=0)
6) Sauvegarde :
   - data_transformed_unscaled.csv (transformé + winsorisé, NON standardisé)
   - data_prepared_for_dml.csv (transformé + winsorisé + standardisé)
   - docs/data_dictionary.csv + docs/data_documentation_phase1.md
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

print("=" * 110)
print("PHASE 1 : TRANSFORMATIONS + DROPNA + WINSORISATION + STANDARDISATION + DOCUMENTATION")
print("=" * 110)


DATA_PROCESSED = Path("../data/processed")
DOCS_PATH = Path("../docs")
DOCS_PATH.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_PROCESSED / "dataset_harmonised_final.csv"

OUT_UNSCALED = DATA_PROCESSED / "data_transformed_unscaled.csv"
OUT_SCALED = DATA_PROCESSED / "data_prepared_for_dml.csv"

OUT_DICT = DOCS_PATH / "data_dictionary.csv"
OUT_DOC = DOCS_PATH / "data_documentation_phase1.md"

# Winsorisation settings
WINSOR_Q_LOW = 0.01
WINSOR_Q_HIGH = 0.99
WINSOR_MODE = "by_country"  # "global" or "by_country"

# Standardisation settings
STD_DDOF = 0  # 0 = population std, 1 = sample std

# =============================================================================
# LOAD
# =============================================================================
df = pd.read_csv(INPUT_FILE)

required_cols = {"Country", "Year"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Dataset doit contenir {required_cols}. Colonnes trouvées: {list(df.columns)}")

vars_to_transform = [c for c in df.columns if c not in ["Country", "Year"]]

print(f"✅ Input : {INPUT_FILE}")
print(f"✅ Shape : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"✅ Pays  : {df['Country'].nunique()} | Période : {df['Year'].min()}–{df['Year'].max()}")
print(f"✅ Vars  : {vars_to_transform}")

# =============================================================================
# TRANSFORMATION PLAN + METADATA (pour documentation)
# =============================================================================
TRANSFORM_PLAN = {
    "GDP_Growth": {
        "transform": "level",
        "unit": "% (annual)",
        "description": "Real GDP growth rate (annual %)",
        "notes": "Conservé en niveau (puis standardisé)."
    },
    "Capital_Formation": {
        "transform": "log",
        "unit": "% of GDP",
        "description": "Gross capital formation (% of GDP)",
        "notes": "log(x), x>0; 0 remplacé par NaN puis dropna."
    },
    "Inflation": {
        "transform": "log1p",
        "unit": "% (annual)",
        "description": "Inflation rate (annual %)",
        "notes": "log(1+x) pour gérer 0; suppose x>=0."
    },
    "Government_Debt": {
        "transform": "log",
        "unit": "% of GDP",
        "description": "General government gross debt (% of GDP)",
        "notes": "log(x), x>0; 0 remplacé par NaN puis dropna."
    },
    "Trade_Balance": {
        "transform": "level",
        "unit": "% of GDP (or comparable)",
        "description": "Trade balance indicator (as provided by source)",
        "notes": "Conservé en niveau (valeurs négatives possibles)."
    },
    "Exchange_Rate": {
        "transform": "log",
        "unit": "local currency per USD (or comparable)",
        "description": "Exchange rate indicator (source IMF/IFS)",
        "notes": "log(x), x>0; 0 remplacé par NaN puis dropna."
    },
    "Reserves": {
        "transform": "log",
        "unit": "USD (or comparable)",
        "description": "International reserves (source IMF/IFS)",
        "notes": "log(x), x>0; 0 remplacé par NaN puis dropna."
    }
}

# Vérifier que toutes les variables existent (sinon on documente quand même mais on skip)
missing = [v for v in TRANSFORM_PLAN.keys() if v not in df.columns]
if missing:
    print(f"⚠️ Variables attendues absentes du dataset: {missing}")

print("\nTRANSFORMATIONS APPLIQUEES\n")

df_t = df.copy()

def safe_log(s: pd.Series) -> pd.Series:
    return np.log(s.replace(0, np.nan))

def safe_log1p(s: pd.Series) -> pd.Series:
    # log(1+x) exige x > -1. Ici inflation devrait être >=0.
    return np.log1p(s)

# Appliquer transformations uniquement si la colonne existe
if "Inflation" in df_t.columns:
    df_t["Inflation"] = safe_log1p(df_t["Inflation"])
    print("Inflation → log(1+x)")

if "Capital_Formation" in df_t.columns:
    df_t["Capital_Formation"] = safe_log(df_t["Capital_Formation"])
    print("Capital_Formation → log")

if "Government_Debt" in df_t.columns:
    df_t["Government_Debt"] = safe_log(df_t["Government_Debt"])
    print("Government_Debt → log")

if "Exchange_Rate" in df_t.columns:
    df_t["Exchange_Rate"] = safe_log(df_t["Exchange_Rate"])
    print("Exchange_Rate → log")

if "Reserves" in df_t.columns:
    df_t["Reserves"] = safe_log(df_t["Reserves"])
    print("Reserves → log")

# Variables en level
for v in ["GDP_Growth", "Trade_Balance"]:
    if v in df_t.columns:
        print(f"{v} → level")

# =============================================================================
# DROPNA (après log)
# =============================================================================
before = len(df_t)
df_t = df_t.dropna().reset_index(drop=True)
after = len(df_t)
print(f"\nDROPNA après log : {before - after} ligne(s) supprimée(s) | restant = {after}")

# =============================================================================
# WINSORISATION
# =============================================================================
print(f"\nWINSORISATION {int(WINSOR_Q_LOW*100)}% (clip {WINSOR_Q_LOW:.2f}–{WINSOR_Q_HIGH:.2f}) | Mode = {WINSOR_MODE}\n")

def winsorize_series(s: pd.Series, q_low: float, q_high: float) -> pd.Series:
    low = s.quantile(q_low)
    high = s.quantile(q_high)
    return s.clip(low, high)

num_cols = [c for c in vars_to_transform if c in df_t.columns]

if WINSOR_MODE == "global":
    for col in num_cols:
        df_t[col] = winsorize_series(df_t[col], WINSOR_Q_LOW, WINSOR_Q_HIGH)
else:
    # winsorisation par pays (souvent plus stable en panel)
    for col in num_cols:
        df_t[col] = df_t.groupby("Country")[col].transform(lambda s: winsorize_series(s, WINSOR_Q_LOW, WINSOR_Q_HIGH))

print("Winsorisation appliquée.")

# =============================================================================
# SAVE INTERMEDIAIRE (NON STANDARDISE)
# =============================================================================
df_t.to_csv(OUT_UNSCALED, index=False)
print(f"\n✅ Dataset transformé (non standardisé) : {OUT_UNSCALED}")

# =============================================================================
# STANDARDISATION Z-SCORE (robuste)
# =============================================================================
print("\nSTANDARDISATION Z-SCORE\n")

df_s = df_t.copy()

for col in num_cols:
    mu = df_s[col].mean()
    sigma = df_s[col].std(ddof=STD_DDOF)

    if sigma == 0 or np.isnan(sigma):
        # Si variance nulle (rare), on met 0 partout (aucune info)
        df_s[col] = 0.0
        print(f"⚠️ {col} : std=0 -> remplacé par 0")
    else:
        df_s[col] = (df_s[col] - mu) / sigma

print("Standardisation terminée.")

df_s.to_csv(OUT_SCALED, index=False)
print(f"\n✅ Dataset prêt pour DML : {OUT_SCALED}")

# =============================================================================
# DATA DICTIONARY + DOCUMENTATION
# =============================================================================
dict_rows = []
for col in ["Country", "Year"] + num_cols:
    if col in ["Country", "Year"]:
        dict_rows.append({
            "variable": col,
            "type": "categorical" if col == "Country" else "time",
            "unit": "-" if col == "Country" else "year",
            "description": "Country name" if col == "Country" else "Year",
            "transform": "none",
            "winsorization": "none",
            "standardization": "none"
        })
        continue

    meta = TRANSFORM_PLAN.get(col, {})
    dict_rows.append({
        "variable": col,
        "type": "numeric",
        "unit": meta.get("unit", "unknown"),
        "description": meta.get("description", "not documented"),
        "transform": meta.get("transform", "unknown"),
        "winsorization": f"{WINSOR_MODE} ({WINSOR_Q_LOW:.2f}-{WINSOR_Q_HIGH:.2f})",
        "standardization": "z-score"
    })

df_dict = pd.DataFrame(dict_rows)
df_dict.to_csv(OUT_DICT, index=False)
print(f"\n✅ Data dictionary : {OUT_DICT}")

doc_md = f"""# Data Documentation — Phase 1 (Preparation)

**Stage ABIL 2025 — Jirince BIABA KUYA**

## Input / Output
- Input: `{INPUT_FILE}`
- Output (transformed, unscaled): `{OUT_UNSCALED}`
- Output (transformed + scaled): `{OUT_SCALED}`

## Panel
- Countries: {df_s['Country'].nunique()}
- Period: {df_s['Year'].min()}–{df_s['Year'].max()}
- Rows: {len(df_s)}

## Transformations (macro-standard)
- GDP_Growth: level
- Inflation: log(1 + x)
- Capital_Formation: log(x)
- Government_Debt: log(x)
- Exchange_Rate: log(x)
- Reserves: log(x)
- Trade_Balance: level

## Cleaning
- Drop rows with NaN after logs: {before - after} row(s) removed

## Winsorization
- Mode: {WINSOR_MODE}
- Quantiles: {WINSOR_Q_LOW:.2f} – {WINSOR_Q_HIGH:.2f}
- Applied **before** standardization

## Standardization
- Z-score on numeric variables (μ=0, σ=1) with ddof={STD_DDOF}

## Notes
- All plots/analyses in Phase B use the **standardized** dataset (`data_prepared_for_dml.csv`).
- For “units on axes”, prefer explicit labels like **Z-score (standardized)**, and keep original units documented in `data_dictionary.csv`.
"""
OUT_DOC.write_text(doc_md, encoding="utf-8")
print(f"✅ Documentation : {OUT_DOC}")

print("\n" + "=" * 110)
print("✅ PHASE 1 TERMINÉE (préparation + fichiers + documentation)")
print("=" * 110)