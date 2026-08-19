#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PHASE 2 (bis) — FIGURES DESCRIPTIVES EN UNITES REELLES

Corrige l'incoherence axe/texte des Figures 1 et 2 :
les figures d'origine utilisaient data_prepared_for_dml.csv (log + z-score),
d'ou des axes en z-score etiquetes "% of GDP". On retrace ici a partir des
donnees BRUTES harmonisees (dataset_harmonised_final.csv), en unites reelles.

Sorties :
    reports/figures/fig1_gdp_growth_raw.png
    reports/figures/fig2_capital_formation_raw.png

+ impression des min/max par pays pour VERIFIER le texte de l'article
  (ex. "Morocco/Angola 30-45%", "DRC 5-15%").
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATA_PATH = Path("../data/processed/dataset_harmonised_final.csv")
FIG_PATH = Path("../reports/figures")
FIG_PATH.mkdir(parents=True, exist_ok=True)

sns.set(style="whitegrid")

df = pd.read_csv(DATA_PATH)
print(f"Chargé : {df.shape[0]} obs × {df.shape[1]} col")
print(f"Colonnes : {list(df.columns)}")
print(f"Pays : {sorted(df['Country'].unique())}")
print(f"Période : {df['Year'].min()}–{df['Year'].max()}")
print()


for var in ["Capital_Formation", "GDP_Growth"]:
    if var in df.columns:
        print(f"=== {var} : min / max par pays (unités brutes) ===")
        g = df.groupby("Country")[var].agg(["min", "max", "mean"]).round(1)
        print(g.to_string())
        print()

# ------------------------------------------------------------------
# FIGURE 1 — GDP GROWTH (brut, %)
# ------------------------------------------------------------------
if "GDP_Growth" in df.columns:
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.lineplot(data=df, x="Year", y="GDP_Growth", hue="Country", ax=ax, marker="o")
    ax.set_title("Annual GDP Growth by Country (2000–2024)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual GDP growth (%)")
    ax.legend(title="Country", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_PATH / "fig1_gdp_growth_raw.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Sauvé : fig1_gdp_growth_raw.png")

# ------------------------------------------------------------------
# FIGURE 2 — CAPITAL FORMATION (brut, % du PIB)
# ------------------------------------------------------------------
if "Capital_Formation" in df.columns:
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.lineplot(data=df, x="Year", y="Capital_Formation", hue="Country", ax=ax, marker="o")
    ax.set_title("Gross Capital Formation by Country (2000–2024)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Gross capital formation (% of GDP)")
    ax.legend(title="Country", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_PATH / "fig2_capital_formation_raw.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Sauvé : fig2_capital_formation_raw.png")

print()
print("FAIT. Vérifie les min/max ci-dessus contre le texte de l'article")
print("(Descriptive Analysis) avant de finaliser les pourcentages cités.")
