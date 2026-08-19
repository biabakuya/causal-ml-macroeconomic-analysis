#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PHASE 4 — DML — ROBUSTNESS CHECKS (Faiblesse 6)

Deux analyses de robustesse sur l'ATE global :

  (1) YEAR FIXED EFFECTS :
      Ré-estimation du LinearDML en ajoutant des indicatrices année,
      afin d'absorber les chocs communs (2008-2009, 2020) qui affectent
      tous les pays simultanément.

  (2) LEAVE-ONE-COUNTRY-OUT (LOCO) :
      Ré-estimation du LinearDML en retirant chaque pays à tour de rôle,
      afin de vérifier qu'aucun pays ne pilote seul le résultat global.

Spécification DML strictement identique à phase4_02_dml_fit_ate.py :
  RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf=2)
  KFold(n_splits=5, shuffle=True, random_state=42)
  LinearDML(discrete_treatment=False, inference="auto")

Input :
    ../data/processed/data_prepared_for_dml_lagged.csv
    (la matrice est reconstruite ici en GARDANT Year, contrairement
     à phase4_01 qui la droppe)

Output :
    ../results/phase4_dml/dml_robustness_year_fe.csv
    ../results/phase4_dml/dml_robustness_loco.csv
"""

import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

from econml.dml import LinearDML
from scipy.stats import norm

from phase4_00_dml_config import (
    DATA_PATH, RES_DIR, Y_COL, T_COL, BASE_X_COLS,
    COUNTRY_COL, YEAR_COL, RANDOM_STATE, N_SPLITS
)

warnings.filterwarnings("ignore")

print("=" * 100)
print("PHASE 4 — DML — ROBUSTNESS CHECKS (year FE + leave-one-country-out)")
print("=" * 100)

# ======================================================================================
# LOAD (en gardant Year)
# ======================================================================================

df = pd.read_csv(DATA_PATH)

needed = [Y_COL, T_COL, COUNTRY_COL, YEAR_COL] + BASE_X_COLS
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Colonnes manquantes: {missing}")

df = df.dropna(subset=needed).reset_index(drop=True)
print(f"Observations : {len(df)}")
print()


# ======================================================================================
# HELPER : construit X et estime le LinearDML (spec identique à phase4_02)
# ======================================================================================

def fit_dml(data, with_year_fe=False):
    """Retourne (ate, lb, ub, stderr, pval, n, k) sur le sous-échantillon fourni."""
    Y = data[Y_COL].astype(float).values.ravel()
    T = data[T_COL].astype(float).values.ravel()

    X_macro = data[BASE_X_COLS].astype(float).reset_index(drop=True)

    X_country = pd.get_dummies(
        data[COUNTRY_COL].astype(str), prefix="C", drop_first=True
    ).astype(int).reset_index(drop=True)

    parts = [X_macro, X_country]

    if with_year_fe:
        X_year = pd.get_dummies(
            data[YEAR_COL].astype(int).astype(str), prefix="Yr", drop_first=True
        ).astype(int).reset_index(drop=True)
        parts.append(X_year)

    X = pd.concat(parts, axis=1).astype(float).values

    model_y = RandomForestRegressor(
        n_estimators=400, max_depth=6, min_samples_leaf=2,
        random_state=RANDOM_STATE
    )
    model_t = RandomForestRegressor(
        n_estimators=400, max_depth=6, min_samples_leaf=2,
        random_state=RANDOM_STATE
    )
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    dml = LinearDML(
        model_y=model_y, model_t=model_t,
        discrete_treatment=False, cv=cv, random_state=RANDOM_STATE
    )
    dml.fit(Y, T, X=X, inference="auto")

    ate = float(dml.ate(X=X))
    lb, ub = dml.ate_interval(X=X)
    lb = float(np.asarray(lb).ravel()[0])
    ub = float(np.asarray(ub).ravel()[0])

    inf = dml.ate_inference(X=X)
    stderr = float(np.asarray(inf.stderr_mean).ravel()[0])
    z = ate / stderr if stderr > 0 else np.nan
    pval = float(2 * (1 - norm.cdf(abs(z)))) if np.isfinite(z) else np.nan

    return ate, lb, ub, stderr, pval, len(Y), X.shape[1]


# ======================================================================================
# (0) BASELINE : sans effets année (doit répliquer phase4_02)
# ======================================================================================

print("-" * 100)
print("(0) BASELINE — country FE only (réplication de phase4_02)")
print("-" * 100)

ate0, lb0, ub0, se0, p0, n0, k0 = fit_dml(df, with_year_fe=False)
print(f"ATE = {ate0:+.4f} | IC95% = [{lb0:+.4f} ; {ub0:+.4f}] | SE = {se0:.4f} | p = {p0:.4f} | n = {n0} | k = {k0}")
print()

# ======================================================================================
# (1) YEAR FIXED EFFECTS
# ======================================================================================

print("-" * 100)
print("(1) YEAR FIXED EFFECTS — country FE + year FE")
print("-" * 100)

ate1, lb1, ub1, se1, p1, n1, k1 = fit_dml(df, with_year_fe=True)
print(f"ATE = {ate1:+.4f} | IC95% = [{lb1:+.4f} ; {ub1:+.4f}] | SE = {se1:.4f} | p = {p1:.4f} | n = {n1} | k = {k1}")
print()

res_year = pd.DataFrame([
    {"specification": "country_FE_only (baseline)", "ATE": ate0, "LB95": lb0, "UB95": ub0,
     "StdErr": se0, "p_value": p0, "n_obs": n0, "n_features_X": k0},
    {"specification": "country_FE + year_FE", "ATE": ate1, "LB95": lb1, "UB95": ub1,
     "StdErr": se1, "p_value": p1, "n_obs": n1, "n_features_X": k1},
])
res_year.to_csv(RES_DIR / "dml_robustness_year_fe.csv", index=False)
print(f"Saved : {RES_DIR / 'dml_robustness_year_fe.csv'}")
print()

# ======================================================================================
# (2) LEAVE-ONE-COUNTRY-OUT
# ======================================================================================

print("-" * 100)
print("(2) LEAVE-ONE-COUNTRY-OUT — country FE only, un pays retiré à la fois")
print("-" * 100)

rows = []
for country in sorted(df[COUNTRY_COL].astype(str).unique()):
    sub = df[df[COUNTRY_COL].astype(str) != country].reset_index(drop=True)
    ate_c, lb_c, ub_c, se_c, p_c, n_c, k_c = fit_dml(sub, with_year_fe=False)
    rows.append({
        "excluded_country": country, "ATE": ate_c, "LB95": lb_c, "UB95": ub_c,
        "StdErr": se_c, "p_value": p_c, "n_obs": n_c
    })
    print(f"sans {country:<10} : ATE = {ate_c:+.4f} | IC95% = [{lb_c:+.4f} ; {ub_c:+.4f}] | p = {p_c:.4f} | n = {n_c}")

res_loco = pd.DataFrame(rows)
res_loco.to_csv(RES_DIR / "dml_robustness_loco.csv", index=False)
print()
print(f"Saved : {RES_DIR / 'dml_robustness_loco.csv'}")
print()

# ======================================================================================
# SYNTHESE
# ======================================================================================

print("=" * 100)
print("SYNTHESE")
print("=" * 100)
all_null = all(r["p_value"] > 0.05 for r in rows) and p1 > 0.05
print(f"Baseline               : ATE = {ate0:+.4f} (p = {p0:.3f})")
print(f"Avec effets année      : ATE = {ate1:+.4f} (p = {p1:.3f})")
print(f"LOCO — plage des ATE   : [{min(r['ATE'] for r in rows):+.4f} ; {max(r['ATE'] for r in rows):+.4f}]")
print(f"Tous non significatifs : {all_null}")
print()
print("ROBUSTNESS CHECKS OK")
