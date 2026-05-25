#!/usr/bin/env python3
"""
PHASE 3D — PREPARATION DATASET PCMCI
Stage ABIL 2025 - Jirince BIABA KUYA
"""

import pandas as pd
import argparse
from pathlib import Path

print("=" * 100)
print("PHASE 3D — PREPARATION DATASET PCMCI")
print("=" * 100)

parser = argparse.ArgumentParser()
parser.add_argument("--country", type=str, required=True)
args = parser.parse_args()
country = args.country

# IMPORTANT :
# PCMCI doit utiliser les variables originales NON laggées.
# L'algorithme gère lui-même les lags temporels.
INPUT_PATH = Path("../data/processed/data_transformed_unscaled.csv")

OUTPUT_DIR = Path("../data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

safe_country = (
    country.replace(",", "")
           .replace(".", "")
           .replace(" ", "_")
)

OUTPUT_FILE = OUTPUT_DIR / f"pcmci_{safe_country}.csv"

# ===============================
# LOAD DATA
# ===============================

df = pd.read_csv(INPUT_PATH)

df_country = df[df["Country"] == country].copy()
df_country = df_country.sort_values("Year")

# Supprimer colonne Country
df_country = df_country.drop(columns=["Country"])

# Reset index
df_country = df_country.reset_index(drop=True)

df_country.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Dataset PCMCI créé : {OUTPUT_FILE}")
print(f"Observations : {len(df_country)}")
print(f"Colonnes : {list(df_country.columns)}")
print("=" * 100)