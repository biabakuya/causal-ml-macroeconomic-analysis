#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PHASE 4 — DML — FIT ATE (VERSION LAGGED)

Estimation de l'effet causal moyen (ATE) :

    Capital_Formation(t-1)  →  GDP_Growth(t)

Confounders (t-1) :
    Inflation
    Government_Debt
    Trade_Balance
    Exchange_Rate
    Reserves
    + Country fixed effects
"""

import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

from econml.dml import LinearDML
from scipy.stats import norm

from phase4_00_dml_config import RES_DIR, RANDOM_STATE, N_SPLITS

warnings.filterwarnings("ignore")

print("=" * 100)
print("PHASE 4 — DML — FIT ATE (LAGGED)")
print("=" * 100)

# =========================================================
# Load design matrix
# =========================================================

design_path = RES_DIR / "dml_design_matrix.csv"

if not design_path.exists():
    raise FileNotFoundError(f"Design matrix introuvable: {design_path}")

df = pd.read_csv(design_path)

# =========================================================
# Extraction Y, T, X
# =========================================================

Y = df["Y"].astype(float).values.ravel()
T = df["T"].astype(float).values.ravel()

X = df.drop(columns=["Y", "T"]).astype(float).values

n = len(Y)

print("Observations :", n)
print("X features   :", X.shape[1])
print()

# =========================================================
# ML models (nuisance)
# =========================================================

model_y = RandomForestRegressor(
    n_estimators=400,
    max_depth=6,
    min_samples_leaf=2,
    random_state=RANDOM_STATE
)

model_t = RandomForestRegressor(
    n_estimators=400,
    max_depth=6,
    min_samples_leaf=2,
    random_state=RANDOM_STATE
)

cv = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

# =========================================================
# Linear DML
# =========================================================

dml = LinearDML(
    model_y=model_y,
    model_t=model_t,
    discrete_treatment=False,
    cv=cv,
    random_state=RANDOM_STATE
)

dml.fit(Y, T, X=X, inference="auto")

# =========================================================
# ATE estimation
# =========================================================

ate = float(dml.ate(X=X))

lb, ub = dml.ate_interval(X=X)
lb = float(np.asarray(lb).ravel()[0])
ub = float(np.asarray(ub).ravel()[0])

# =========================================================
# Std error + p-value
# =========================================================

inf = dml.ate_inference(X=X)

stderr = float(np.asarray(inf.stderr_mean).ravel()[0])

z = ate / stderr if stderr > 0 else np.nan
pval = float(2 * (1 - norm.cdf(abs(z)))) if np.isfinite(z) else np.nan

# =========================================================
# Save results
# =========================================================

res = pd.DataFrame([{
    "ATE": ate,
    "ATE_LB_95": lb,
    "ATE_UB_95": ub,
    "StdErr": stderr,
    "p_value": pval,
    "n_obs": int(n),
    "n_features_X": int(X.shape[1]),
    "cv_splits": int(N_SPLITS),
    "treatment": "Capital_Formation(t-1)",
    "outcome": "GDP_Growth(t)"
}])

out_path = RES_DIR / "dml_ate_results.csv"

res.to_csv(out_path, index=False)

print("ATE        :", ate)
print("IC95%      :", (lb, ub))
print("StdErr     :", stderr)
print("p-value    :", pval)
print("Saved      :", out_path)