#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PHASE 4 — CausalForestDML — HTE Global

"""

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

from econml.dml import CausalForestDML

from phase4_00_dml_config import (
    DATA_PATH, RES_DIR, RANDOM_STATE, N_SPLITS,
    Y_COL, T_COL, BASE_X_COLS, COUNTRY_COL, YEAR_COL
)

np.random.seed(int(RANDOM_STATE))

print("=" * 100)
print("PHASE 4 — CausalForestDML — HTE GLOBAL")
print("=" * 100)

# ------------------------------------------------
# Load
# ------------------------------------------------
df = pd.read_csv(DATA_PATH)

if YEAR_COL in df.columns:
    df = df.drop(columns=[YEAR_COL])

# ------------------------------------------------
# Define groups
# ------------------------------------------------
developed = {"France", "Allemagne"}
developing = {"Maroc", "Congo, Dem. Rep.", "Angola", "Ghana", "Nigeria"}

df["Category"] = df[COUNTRY_COL].astype(str).apply(
    lambda c: "Developed" if c in developed else "Developing"
)

# ------------------------------------------------
# Build matrices
# ------------------------------------------------
Y = df[Y_COL].astype(float).values.ravel()
T = df[T_COL].astype(float).values.ravel()

X_macro = df[BASE_X_COLS].astype(float)
X_country = pd.get_dummies(df[COUNTRY_COL].astype(str), prefix="C", drop_first=True)

X = pd.concat([X_macro, X_country], axis=1).astype(np.float64).values

print("Observations:", len(df))
print("X dimension:", X.shape[1])

# ------------------------------------------------
# Models
# ------------------------------------------------
model_y = RandomForestRegressor(
    n_estimators=500,
    max_depth=8,
    min_samples_leaf=2,
    random_state=RANDOM_STATE
)

model_t = RandomForestRegressor(
    n_estimators=500,
    max_depth=8,
    min_samples_leaf=2,
    random_state=RANDOM_STATE
)

cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

cf = CausalForestDML(
    model_y=model_y,
    model_t=model_t,
    discrete_treatment=False,
    cv=cv,
    n_estimators=800,
    min_samples_leaf=5,
    random_state=RANDOM_STATE
)

# ------------------------------------------------
# Fit
# ------------------------------------------------
cf.fit(Y, T, X=X)

# ------------------------------------------------
# Individual treatment effects
# ------------------------------------------------
tau = cf.effect(X)

df["HTE"] = tau

# ------------------------------------------------
# Export individual effects
# ------------------------------------------------
out_indiv = RES_DIR / "dml_hte_individual_effects.csv"
df.to_csv(out_indiv, index=False)

print("✅ Individual HTE saved:", out_indiv)

# ------------------------------------------------
# HTE by group
# ------------------------------------------------
group_results = (
    df.groupby("Category")["HTE"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

group_results.rename(columns={
    "mean": "ATE_CF",
    "std": "HTE_std",
    "count": "n"
}, inplace=True)

out_group = RES_DIR / "dml_hte_by_group_cf.csv"
group_results.to_csv(out_group, index=False)

print("\nHTE by group:")
print(group_results)

print("\n✅ Group HTE saved:", out_group)

print("=" * 100)
print("✅ PHASE 4 — CausalForestDML — DONE")
print("=" * 100)