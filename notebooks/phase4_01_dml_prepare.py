#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PHASE 4 — DML — PREPARE (VERSION LAGGED)

Objectif :
Construire la matrice DML finale :
    Y = GDP_Growth(t)
    T = Capital_Formation(t-1)
    X = confounders(t-1) + effets fixes pays

Input :
    ../data/processed/data_prepared_for_dml_lagged.csv

Output :
    ../results/phase4_dml/dml_design_matrix.csv
"""

import pandas as pd
import numpy as np

from phase4_00_dml_config import (
    DATA_PATH, RES_DIR, Y_COL, T_COL, BASE_X_COLS, COUNTRY_COL, YEAR_COL
)

print("=" * 100)
print("PHASE 4 — DML — PREPARE (LAGGED T, Y, X)")
print("=" * 100)

# ======================================================================================
# LOAD
# ======================================================================================

df = pd.read_csv(DATA_PATH)

print(f"📥 Input dataset : {DATA_PATH}")
print(f"📊 Shape initial : {df.shape}")
print()

# ======================================================================================
# CHECK REQUIRED COLUMNS
# ======================================================================================

needed = [Y_COL, T_COL, COUNTRY_COL] + BASE_X_COLS
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Colonnes manquantes: {missing}")

# ======================================================================================
# DROP YEAR (pas utilisé directement dans DML)
# ======================================================================================

if YEAR_COL in df.columns:
    df = df.drop(columns=[YEAR_COL])

# ======================================================================================
# CLEAN NA
# ======================================================================================

use_cols = [Y_COL, T_COL, COUNTRY_COL] + BASE_X_COLS
before = len(df)
df = df.dropna(subset=use_cols).reset_index(drop=True)
after = len(df)

if after < before:
    print(f"⚠️ NA supprimés : {before - after} lignes (restant : {after})")
    print()

# ======================================================================================
# DEFINE Y, T, X
# ======================================================================================

# Outcome : GDP_Growth(t)
Y = df[Y_COL].astype(float).to_numpy()

# Treatment : Capital_Formation(t-1)
T = df[T_COL].astype(float).to_numpy().reshape(-1, 1)

# Confounders : X(t-1)
X_macro = df[BASE_X_COLS].astype(float).reset_index(drop=True)

# Effets fixes pays
X_country = pd.get_dummies(
    df[COUNTRY_COL].astype(str),
    prefix="C",
    drop_first=True
).astype(int).reset_index(drop=True)

# X final
X = pd.concat([X_macro, X_country], axis=1)

# ======================================================================================
# SANITY CHECKS
# ======================================================================================

bad_cols = [c for c in X.columns if str(c).strip().lower() in {"false", "true"}]
if bad_cols:
    raise ValueError(
        f"Colonnes invalides détectées dans X: {bad_cols}. Vérifie le get_dummies/concat."
    )

if len(Y) != len(T) or len(Y) != len(X):
    raise ValueError(
        f"Incohérence dimensions: len(Y)={len(Y)}, len(T)={len(T)}, len(X)={len(X)}"
    )

# ======================================================================================
# EXPORT DESIGN MATRIX
# ======================================================================================

out = pd.DataFrame({"Y": Y, "T": T.flatten()})
out = pd.concat([out, X], axis=1)

out_path = RES_DIR / "dml_design_matrix.csv"
out.to_csv(out_path, index=False)

# ======================================================================================
# REPORTING
# ======================================================================================

print("✅ MATRICE DML CONSTRUITE")
print(f"✅ Observations        : {len(df)}")
print(f"✅ Outcome Y           : {Y_COL} = GDP_Growth(t)")
print(f"✅ Treatment T         : {T_COL} = Capital_Formation(t-1)")
print(f"✅ X macro (t-1)       : {BASE_X_COLS} ({X_macro.shape[1]} colonnes)")
print(f"✅ Country dummies     : {X_country.shape[1]} colonnes -> {list(X_country.columns)}")
print(f"✅ X total             : {X.shape[1]} colonnes")
print(f"✅ Saved               : {out_path}")
print()

print("Structure causale retenue :")
print("  Capital_Formation(t-1) -> GDP_Growth(t)")
print("  Confounders(t-1)       -> GDP_Growth(t)")
print()
print("✅ PREPARE OK")