#!/usr/bin/env python3
"""
PHASE B : ANALYSE DESCRIPTIVE (VERSION RAPPORT STAGE - CORRIGÉE)


    Ce script travaille sur data_prepared_for_dml.csv (déjà transformé + winsorisé + standardisé). 
    nous cherchons de comprendre la distribution et le comportement des variables avant d'appliquer les méthodes Causales
     
nous avons fait:  

- Summary statistics global du dataset
- Summary statistics par pays (descriptive_by_country.csv)
- Matrice de corrélation (heatmap)
- Scatter Capital Formation vs GDP Growth + tendance globale

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# CONFIGURATION


DATA_PATH = Path("../data/processed/data_prepared_for_dml.csv")

REPORTS_PATH = Path("../reports")
FIG_PATH = REPORTS_PATH / "figures"
FIG_PATH.mkdir(parents=True, exist_ok=True)

# Outputs demandés
OUT_DESC_BY_COUNTRY = REPORTS_PATH / "descriptive_by_country.csv"
OUT_SUMMARY_GLOBAL = REPORTS_PATH / "summary_statistics_global.csv"

sns.set(style="whitegrid")

print("=" * 100)
print("PHASE B : ANALYSE DESCRIPTIVE (CORRIGÉE: UNITÉS + SUMMARY STATS)")
print("=" * 100)


# LOAD DATA


df = pd.read_csv(DATA_PATH)

required_cols = {"Country", "Year"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Le dataset doit contenir {required_cols}. Colonnes trouvées: {list(df.columns)}")

vars_numeric = [c for c in df.columns if c not in ["Country", "Year"]]

print(f" Dataset chargé : {df.shape[0]} observations × {df.shape[1]} colonnes")
print(f" Pays : {df['Country'].nunique()} | Période : {df['Year'].min()}–{df['Year'].max()}")
print(f" Variables numériques : {vars_numeric}")


# METADATA UNITÉS / NOTES

# Ici, les données sont standardisées (Z-score).
# On met quand même des libellés utiles sur les figures.

UNITS = {
    "GDP_Growth": "Z-score (GDP growth, %)",
    "Capital_Formation": "Z-score (Capital formation, % of GDP)",
    "Inflation": "Z-score (Inflation, transformed)",
    "Government_Debt": "Z-score (Gov. debt, % of GDP)",
    "Trade_Balance": "Z-score (Trade balance, % of GDP)",
    "Exchange_Rate": "Z-score (Exchange rate, transformed)",
    "Reserves": "Z-score (Reserves, transformed)"
}

TRANSFORM_NOTES = [
    "Dataset utilisé: data_prepared_for_dml.csv",
    "Pré-traitements (Phase 1): transformations log / log1p selon variable, nettoyage NaN, winsorisation 1% (1–99%), standardisation Z-score.",
    "Les graphiques affichent des valeurs standardisées (Z-score). Les unités d'origine (%, USD, etc.) sont indiquées à titre de référence."
]

def add_footnote(fig, note_lines, fontsize=9):
    """Ajoute un bloc note en bas de figure."""
    note = " | ".join(note_lines)
    fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=fontsize)


# B1 — SUMMARY STATISTICS (GLOBAL)


# Summary global (toutes observations confondues)
summary_global = df[vars_numeric].describe().T
# Ajouter skew/kurtosis pour lecture qualité
summary_global["skew"] = df[vars_numeric].skew(numeric_only=True)
summary_global["kurtosis"] = df[vars_numeric].kurtosis(numeric_only=True)

summary_global.to_csv(OUT_SUMMARY_GLOBAL)
print(f" Summary statistics global sauvegardé : {OUT_SUMMARY_GLOBAL}")


# B1bis — DESCRIPTIVE BY COUNTRY


# Table descriptive par pays (mean/std/min/max)
desc_by_country = (
    df.groupby("Country")[vars_numeric]
      .agg(["mean", "std", "min", "max"])
)

desc_by_country.to_csv(OUT_DESC_BY_COUNTRY)
print(f" Descriptive by country sauvegardé : {OUT_DESC_BY_COUNTRY}")


# B2 — CORRELATION MATRIX


corr = df[vars_numeric].corr()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Matrix (standardised variables)")
plt.tight_layout()
add_footnote(fig, TRANSFORM_NOTES)
fig.savefig(FIG_PATH / "correlation_matrix.png", dpi=200)
plt.close(fig)

print(" Heatmap corrélation sauvegardée : correlation_matrix.png")


# B3 — SCATTER CAPITAL vs GROWTH


fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111)

sns.scatterplot(
    data=df,
    x="Capital_Formation",
    y="GDP_Growth",
    hue="Country",
    s=80,
    ax=ax
)

# Tendance globale (simple OLS sur données standardisées)
sns.regplot(
    data=df,
    x="Capital_Formation",
    y="GDP_Growth",
    scatter=False,
    ax=ax
)

ax.set_title("Capital Formation vs GDP Growth (Z-score)")
ax.set_xlabel(UNITS.get("Capital_Formation", "Capital_Formation"))
ax.set_ylabel(UNITS.get("GDP_Growth", "GDP_Growth"))

plt.tight_layout()
add_footnote(fig, TRANSFORM_NOTES)
fig.savefig(FIG_PATH / "scatter_capital_growth.png", dpi=200)
plt.close(fig)

print(" Scatter plot sauvegardé : scatter_capital_growth.png")


# B4 — EVOLUTION TEMPORELLE (2000–2024)


# On inclut aussi Trade_Balance pour cohérence, si présent
time_vars = [
    "GDP_Growth",
    "Capital_Formation",
    "Inflation",
    "Government_Debt",
    "Trade_Balance",
    "Exchange_Rate",
    "Reserves"
]

time_vars = [v for v in time_vars if v in df.columns]

for var in time_vars:
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)

    sns.lineplot(
        data=df.sort_values(["Country", "Year"]),
        x="Year",
        y=var,
        hue="Country",
        ax=ax
    )

    ax.set_title(f"{var} Evolution ({df['Year'].min()}–{df['Year'].max()})")
    ax.set_xlabel("Year")
    ax.set_ylabel(UNITS.get(var, var))

    plt.tight_layout()
    add_footnote(fig, TRANSFORM_NOTES)
    fig.savefig(FIG_PATH / f"time_{var}.png", dpi=200)
    plt.close(fig)

print(" Graphiques temporels sauvegardés : time_*.png")

print("\n PHASE B TERMINÉE (unités + summary stats + exports).")