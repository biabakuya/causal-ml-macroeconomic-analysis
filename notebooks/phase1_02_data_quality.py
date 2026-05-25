#!/usr/bin/env python3
"""
PHASE 1B : CONTROLE QUALITE DATASET (post-transformation & standardisation)

Ce script vérifie :
1) Structure : doublons, années manquantes par pays, types
2) NaN / inf : par variable + lignes concernées
3) Valeurs extrêmes : max abs, z-scores, top outliers
4) Stationnarité (ADF/KPSS) sur le dataset préparé (après transformations)
5) Sauvegarde d'un rapport texte

Entrée : ../data/processed/data_prepared_for_dml.csv
Sorties :
- ./phase1_data_quality_report.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings("ignore")



DATA_PROCESSED = Path("../data/processed")
INPUT_FILE = DATA_PROCESSED / "data_prepared_for_dml.csv"
REPORT_FILE = Path("./phase1_data_quality_report.txt")
ALPHA = 0.05

ID_COLS = ["Country", "Year"]



# HELPERS

def safe_adf(series: pd.Series):
    s = series.dropna()
    if len(s) < 8:
        return np.nan
    try:
        return adfuller(s, autolag="AIC")[1]
    except Exception:
        return np.nan


def safe_kpss(series: pd.Series):
    s = series.dropna()
    if len(s) < 8:
        return np.nan
    try:
        return kpss(s, regression="c", nlags="auto")[1]
    except Exception:
        return np.nan


def write_line(f, txt=""):
    f.write(txt + "\n")
    print(txt)

# LOAD

df = pd.read_csv(INPUT_FILE)

vars_num = [c for c in df.columns if c not in ID_COLS]

# REPORT
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    write_line(f, "=" * 100)
    write_line(f, "PHASE 1B : CONTROLE QUALITE DATASET (post-preparation)")
    write_line(f, "=" * 100)
    write_line(f)
    write_line(f, f"Input : {INPUT_FILE}")
    write_line(f, f"Obs   : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    write_line(f, f"Pays  : {df['Country'].nunique()} | Période : {df['Year'].min()}–{df['Year'].max()}")
    write_line(f)

    # 1) STRUCTURE
   
    write_line(f, "=" * 100)
    write_line(f, "1) STRUCTURE : doublons, années manquantes, types")
    write_line(f, "=" * 100)

    # Duplicates Country-Year
    dup = df.duplicated(subset=ID_COLS).sum()
    write_line(f, f"- Doublons (Country, Year) : {dup}")
    if dup > 0:
        dups = df[df.duplicated(subset=ID_COLS, keep=False)].sort_values(ID_COLS)
        write_line(f, "  -> Exemples de doublons :")
        write_line(f, dups.head(10).to_string(index=False))

    # Missing years per country
    years_all = set(range(int(df["Year"].min()), int(df["Year"].max()) + 1))
    write_line(f, "\n- Années manquantes par pays :")
    missing_any = False
    for c in sorted(df["Country"].unique()):
        years_c = set(df.loc[df["Country"] == c, "Year"].astype(int).tolist())
        missing = sorted(list(years_all - years_c))
        if missing:
            missing_any = True
            write_line(f, f"  {c:<18} -> {len(missing)} manquantes : {missing[:10]}{'...' if len(missing)>10 else ''}")
        else:
            write_line(f, f"  {c:<18} -> OK (aucune)")

    if not missing_any:
        write_line(f, "  -> Aucun trou temporel détecté ")

    # Dtypes
    write_line(f, "\n- Types de colonnes :")
    write_line(f, df.dtypes.to_string())

       # 2) NaN / inf
   
    write_line(f, "\n" + "=" * 100)
    write_line(f, "2) NaN / INF : par variable + lignes concernées")
    write_line(f, "=" * 100)

    nan_counts = df[vars_num].isna().sum().sort_values(ascending=False)
    write_line(f, "- NaN par variable (top) :")
    write_line(f, nan_counts.head(20).to_string())

    # Inf check
    inf_mask = np.isinf(df[vars_num].to_numpy())
    n_inf = int(inf_mask.sum())
    write_line(f, f"\n- INF total : {n_inf}")
    if n_inf > 0:
        # show where
        where = np.argwhere(inf_mask)
        rows = sorted(set(int(r) for r, _ in where))
        write_line(f, f"  -> Lignes avec INF (exemples) : {rows[:10]}{'...' if len(rows)>10 else ''}")

    # Rows with any NaN/Inf
    any_bad = df[vars_num].isna().any(axis=1) | np.isinf(df[vars_num]).any(axis=1)
    n_bad_rows = int(any_bad.sum())
    write_line(f, f"\n- Lignes avec NaN ou INF : {n_bad_rows}/{len(df)}")
    if n_bad_rows > 0:
        write_line(f, "  -> Exemples (Country, Year + variables problématiques) :")
        sample = df.loc[any_bad, ID_COLS + vars_num].head(10)
        write_line(f, sample.to_string(index=False))

    # 3) Valeurs extrêmes

    write_line(f, "\n" + "=" * 100)
    write_line(f, "3) VALEURS EXTREMES : max abs, outliers z-score")
    write_line(f, "=" * 100)

    # Max abs per variable
    max_abs = df[vars_num].abs().max().sort_values(ascending=False)
    write_line(f, "- Max |x| par variable (top) :")
    write_line(f, max_abs.head(20).to_string())

    # Z-score outliers (since already standardized, z == value)
    # If dataset is standardized, abs(value) > 3.5 is a strong outlier
    thresh = 3.5
    write_line(f, f"\n- Outliers forts (|z| > {thresh}) :")
    outlier_counts = {}
    for v in vars_num:
        count = int((df[v].abs() > thresh).sum())
        outlier_counts[v] = count
    outlier_counts = pd.Series(outlier_counts).sort_values(ascending=False)
    write_line(f, outlier_counts.to_string())

    # Show top 5 extreme rows for each variable (optional concise)
    write_line(f, f"\n- Top 5 valeurs extrêmes par variable (|z| max) :")
    for v in vars_num:
        top = df.loc[:, ID_COLS + [v]].copy()
        top["abs"] = top[v].abs()
        top = top.sort_values("abs", ascending=False).head(5).drop(columns=["abs"])
        write_line(f, f"\n  {v}:")
        write_line(f, top.to_string(index=False))

    
    # 4) Stationnarité (post-transform)

    write_line(f, "\n" + "=" * 100)
    write_line(f, "4) STATIONNARITE (post-transform) : ADF & KPSS par pays")
    write_line(f, "=" * 100)
    write_line(f, f"Alpha = {ALPHA}")

    for country in sorted(df["Country"].unique()):
        write_line(f, f"\n----- {country} -----")
        subset = df[df["Country"] == country].sort_values("Year")

        for v in vars_num:
            adf_p = safe_adf(subset[v])
            kpss_p = safe_kpss(subset[v])

            if np.isnan(adf_p) or np.isnan(kpss_p):
                status = "SKIP"
            else:
                adf_ok = adf_p < ALPHA      # reject unit root => stationary
                kpss_ok = kpss_p > ALPHA    # fail to reject stationarity => stationary
                if adf_ok and kpss_ok:
                    status = "STATIONARY"
                elif (not adf_ok) and (not kpss_ok):
                    status = "NON-STATIONARY"
                else:
                    status = "INCONCLUSIVE"

            write_line(f, f"{v:<25} | ADF p={adf_p if not np.isnan(adf_p) else np.nan:>7.4f} "
                           f"| KPSS p={kpss_p if not np.isnan(kpss_p) else np.nan:>7.4f} | {status}")

  
    # 5) Conclusion rapide
  
    write_line(f, "\n" + "=" * 100)
    write_line(f, "5) CONCLUSION")
    write_line(f, "=" * 100)

    write_line(f, f"- Doublons (Country,Year) : {dup}")
    write_line(f, f"- Lignes NaN/INF          : {n_bad_rows}")
    write_line(f, f"- Max |z| global          : {max_abs.iloc[0]:.4f}")
    write_line(f, f"- Rapport enregistré      : {REPORT_FILE}")

print(f"\n Rapport généré : {REPORT_FILE}")