#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PHASE 4 — DML — ATE PAR GROUPE (Developed vs Developing)


- Traitement (T) continu : Capital_Formation
- Outcome (Y) : GDP_Growth
- Confounders (X) : Inflation, Government_Debt, Trade_Balance, Exchange_Rate, Reserves + dummies Country
- Estimator : LinearDML (econml)
- Inference : BootstrapInference (sans random_state car dépend des versions econml)

Sorties:
  ../results/phase4_dml/dml_ate_by_group.csv
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

from econml.dml import LinearDML
from econml.inference import BootstrapInference

from phase4_00_dml_config import (
    DATA_PATH, RES_DIR, RANDOM_STATE, N_SPLITS,
    Y_COL, T_COL, BASE_X_COLS, COUNTRY_COL, YEAR_COL
)

# Repro globale (utile même si BootstrapInference n'a pas random_state)
np.random.seed(int(RANDOM_STATE))

print("=" * 100)
print("PHASE 4 — DML — ATE PAR GROUPE (Developed vs Developing)")
print("=" * 100)


# -----------------------------
# Helpers robustes
# -----------------------------
def norm_cdf(x: float) -> float:
    """CDF normale standard sans scipy."""
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def safe_call(obj):
    """Si obj est callable -> appelle, sinon retourne."""
    return obj() if callable(obj) else obj


def to_float_scalar(x):
    """Convertit un scalaire/array/list en float scalaire."""
    if x is None:
        return np.nan
    if np.isscalar(x):
        try:
            return float(x)
        except Exception:
            return np.nan
    try:
        arr = np.array(x).reshape(-1)
        if arr.size == 0:
            return np.nan
        return float(arr[0])
    except Exception:
        return np.nan


def build_X(df_sub: pd.DataFrame, x_macro_cols, country_col) -> np.ndarray:
    """Construit X = X_macro + dummies country, en float64 garanti."""
    X_macro = df_sub[x_macro_cols].astype(np.float64)
    X_country = pd.get_dummies(df_sub[country_col].astype(str), prefix="C", drop_first=True)
    X = pd.concat([X_macro, X_country], axis=1)
    return X.astype(np.float64).values


def fit_dml_ate(df_sub: pd.DataFrame, label: str):
    df_sub = df_sub.copy()

    # Drop Year si présent
    if YEAR_COL in df_sub.columns:
        df_sub = df_sub.drop(columns=[YEAR_COL])

    # Y, T en 1D (évite DataConversionWarning)
    Y = df_sub[Y_COL].astype(np.float64).values.ravel()
    T = df_sub[T_COL].astype(np.float64).values.ravel()
    X = build_X(df_sub, BASE_X_COLS, COUNTRY_COL)

    # Modèles de nuisance
    model_y = RandomForestRegressor(
        n_estimators=400, max_depth=6, min_samples_leaf=2, random_state=RANDOM_STATE
    )
    model_t = RandomForestRegressor(
        n_estimators=400, max_depth=6, min_samples_leaf=2, random_state=RANDOM_STATE
    )

    # CV adapté au n
    splits = min(int(N_SPLITS), max(2, len(df_sub) // 10))
    cv = KFold(n_splits=splits, shuffle=True, random_state=RANDOM_STATE)

    # Bootstrap inference (API compatible versions)
    inf = BootstrapInference(n_bootstrap_samples=200, n_jobs=1)

    dml = LinearDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=False,
        cv=cv,
        random_state=RANDOM_STATE,
    )

    # Fit
    dml.fit(Y, T, X=X, inference=inf)

    # ATE
    ate = to_float_scalar(dml.ate(X=X))

    # IC95%
    lb, ub = (np.nan, np.nan)
    try:
        lb_, ub_ = dml.ate_interval(X=X)
        lb = to_float_scalar(lb_)
        ub = to_float_scalar(ub_)
    except Exception as e:
        print(f"⚠️ {label}: ate_interval failed -> {e}")

    # StdErr + p-value
    stderr, pval = (np.nan, np.nan)
    try:
        ate_inf = dml.ate_inference(X=X)

        # 1) Si summary_frame existe (le plus propre)
        if hasattr(ate_inf, "summary_frame"):
            sf = ate_inf.summary_frame()

            if "stderr" in sf.columns:
                stderr = to_float_scalar(sf["stderr"].values[0])
            if "pvalue" in sf.columns:
                pval = to_float_scalar(sf["pvalue"].values[0])

            # Certains summary_frame contiennent aussi l'IC
            if np.isnan(lb) and "ci_lower" in sf.columns:
                lb = to_float_scalar(sf["ci_lower"].values[0])
            if np.isnan(ub) and "ci_upper" in sf.columns:
                ub = to_float_scalar(sf["ci_upper"].values[0])

        else:
            # 2) Fallback versions : stderr_mean / pvalue peuvent être methods ou arrays
            if hasattr(ate_inf, "stderr_mean"):
                se_raw = safe_call(ate_inf.stderr_mean)
                stderr = to_float_scalar(se_raw)

            if hasattr(ate_inf, "pvalue"):
                pv_raw = safe_call(ate_inf.pvalue)
                pval = to_float_scalar(pv_raw)

    except Exception as e:
        print(f"⚠️ {label}: ate_inference failed -> {e}")

    # Fallback final : approx StdErr + p-value depuis l'IC95
    # IC95 = ate ± 1.96*se => se ≈ (ub-lb)/(2*1.96)
    if (np.isnan(stderr) or np.isnan(pval)) and (not np.isnan(lb)) and (not np.isnan(ub)):
        try:
            se_approx = float((ub - lb) / (2 * 1.96))
            if np.isnan(stderr):
                stderr = se_approx
            if np.isnan(pval) and se_approx > 0:
                z = abs(float(ate) / se_approx)
                pval = 2 * (1 - norm_cdf(z))
        except Exception:
            pass

    return {
        "Group": label,
        "n": int(len(df_sub)),
        "splits": int(splits),
        "ATE": float(ate) if not np.isnan(ate) else np.nan,
        "ATE_LB_95": float(lb) if not np.isnan(lb) else np.nan,
        "ATE_UB_95": float(ub) if not np.isnan(ub) else np.nan,
        "StdErr": float(stderr) if not np.isnan(stderr) else np.nan,
        "p_value": float(pval) if not np.isnan(pval) else np.nan,
        "X_dim": int(X.shape[1]),
    }


# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)

needed = [Y_COL, T_COL, COUNTRY_COL] + BASE_X_COLS
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Colonnes manquantes dans DATA_PATH: {missing}")

countries = sorted(df[COUNTRY_COL].astype(str).unique().tolist())
print("✅ Countries in dataset:", countries)

# -----------------------------
# -----------------------------
# Define groups
# -----------------------------
developed = {"France", "Germany"}
developing = {"Morocco", "DRC", "Angola", "Ghana", "Nigeria"}

df["Category"] = df[COUNTRY_COL].astype(str).apply(
    lambda c: "Developed" if c in developed else ("Developing" if c in developing else "Other")
)

print("✅ Category counts:\n", df["Category"].value_counts(dropna=False))

df_dev = df[df["Category"] == "Developed"].copy()
df_devg = df[df["Category"] == "Developing"].copy()

rows = []

if len(df_dev) >= 20:
    rows.append(fit_dml_ate(df_dev, "Developed"))
    print(f"✅ Developed ATE computed (n={len(df_dev)})")
else:
    print(f"⚠️ Developed group too small (n={len(df_dev)}) -> skip")

if len(df_devg) >= 20:
    rows.append(fit_dml_ate(df_devg, "Developing"))
    print(f"✅ Developing ATE computed (n={len(df_devg)})")
else:
    print(f"⚠️ Developing group too small (n={len(df_devg)}) -> skip")

# -----------------------------
# Export
# -----------------------------
out_path = RES_DIR / "dml_ate_by_group.csv"
if rows:
    out = pd.DataFrame(rows)
    if "p_value" in out.columns:
        out = out.sort_values(["p_value", "Group"], ascending=[True, True])
    out.to_csv(out_path, index=False)
    print(f"\n✅ Saved: {out_path}")
    print(out.to_string(index=False))
else:
    print("\n⚠️ Aucun résultat exporté.")

print("=" * 100)
print("✅ PHASE 4 — DML — ATE PAR GROUPE — DONE")
print("=" * 100)