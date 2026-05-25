#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

# XGBoost optionnel
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from phase4_00_dml_config import RES_DIR, RANDOM_STATE, N_SPLITS

print("="*100)
print("PHASE 4 — DML — NUISANCE MODELS EVALUATION (R² + MSE)")
print("="*100)

# ------------------------------------------------------------------
# Load design matrix
# ------------------------------------------------------------------
design_path = RES_DIR / "dml_design_matrix.csv"
df = pd.read_csv(design_path)

Y = df["Y"].astype(float).values.ravel()
T = df["T"].astype(float).values.ravel()
X = df.drop(columns=["Y", "T"]).astype(float).values

print(f"Observations: {len(Y)}")
print(f"X dimension: {X.shape[1]} variables")

# ------------------------------------------------------------------
# CV setup
# ------------------------------------------------------------------
cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# scorers
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
r2_scorer = make_scorer(r2_score)

# ------------------------------------------------------------------
# Candidate models
# ------------------------------------------------------------------
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(random_state=RANDOM_STATE),
    "Lasso": Lasso(random_state=RANDOM_STATE, max_iter=5000),
    "RandomForest": RandomForestRegressor(
        n_estimators=400, max_depth=6, min_samples_leaf=2, random_state=RANDOM_STATE
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=400, max_depth=3, random_state=RANDOM_STATE
    ),
}

if HAS_XGB:
    models["XGBoost"] = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        random_state=RANDOM_STATE, verbosity=0
    )

# ------------------------------------------------------------------
# Evaluation loop
# ------------------------------------------------------------------
rows = []

for name, model in models.items():
    print(f"\n🔎 Evaluating: {name}")

    # ---- Model Y (E[Y|X])
    r2_y = cross_val_score(model, X, Y, cv=cv, scoring=r2_scorer)
    mse_y = cross_val_score(model, X, Y, cv=cv, scoring=mse_scorer)

    # ---- Model T (E[T|X])
    r2_t = cross_val_score(model, X, T, cv=cv, scoring=r2_scorer)
    mse_t = cross_val_score(model, X, T, cv=cv, scoring=mse_scorer)

    rows.append({
        "Model": name,
        "R2_Y_mean": np.mean(r2_y),
        "R2_Y_std": np.std(r2_y),
        "MSE_Y_mean": -np.mean(mse_y),  # negatif car scorer inversé
        "R2_T_mean": np.mean(r2_t),
        "R2_T_std": np.std(r2_t),
        "MSE_T_mean": -np.mean(mse_t)
    })

    print(f"   Y → R2={np.mean(r2_y):.4f} | MSE={-np.mean(mse_y):.4f}")
    print(f"   T → R2={np.mean(r2_t):.4f} | MSE={-np.mean(mse_t):.4f}")

# ------------------------------------------------------------------
# Save results
# ------------------------------------------------------------------
results = pd.DataFrame(rows)
results = results.sort_values("R2_Y_mean", ascending=False)

out_path = RES_DIR / "dml_nuisance_model_comparison.csv"
results.to_csv(out_path, index=False)

print("\n" + "="*100)
print("✅ Nuisance evaluation completed")
print(f"✅ Saved: {out_path}")
print("="*100)