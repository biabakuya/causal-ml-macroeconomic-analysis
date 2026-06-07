#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PHASE 4 — DML — HTE / ATE PAR PAYS (VERSION LAGGED)

Objectif :
  Estimer, pour chaque pays, l'effet causal moyen (ATE par sous-échantillon pays) de :

      Capital_Formation(t-1)  --->  GDP_Growth(t)

  en contrôlant les confondeurs macroéconomiques en t-1.

Entrée :
  - ../results/phase4_dml/dml_design_matrix.csv
    (généré par phase4_01_dml_prepare.py à partir de la base laggée)

Sortie :
  - ../results/phase4_dml/dml_hte_by_country.csv
"""

import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from econml.dml import LinearDML

from phase4_00_dml_config import RES_DIR, RANDOM_STATE, N_SPLITS

warnings.filterwarnings("ignore")

print("=" * 100)
print("PHASE 4 — DML — HTE (ATE PAR PAYS, VERSION LAGGED)")
print("=" * 100)

design_path = RES_DIR / "dml_design_matrix.csv"
if not design_path.exists():
    raise FileNotFoundError(f"Design matrix introuvable: {design_path}")

df = pd.read_csv(design_path)


def safe_float(obj):
    if obj is None:
        return np.nan

    if callable(obj):
        try:
            val = obj()
            return float(np.ravel(val)[0])
        except Exception:
            return np.nan

    try:
        return float(np.ravel(obj)[0])
    except Exception:
        return np.nan


if "Y" not in df.columns or "T" not in df.columns:
    raise ValueError("Le fichier dml_design_matrix.csv doit contenir les colonnes 'Y' et 'T'.")

Y_all = df["Y"].astype(float).values.ravel()
T_all = df["T"].astype(float).values.ravel()

country_cols = [c for c in df.columns if c.startswith("C_")]
if not country_cols:
    raise ValueError("Aucune dummy pays trouvée (colonnes 'C_*'). Vérifie phase4_01_dml_prepare.py.")

country_mat = df[country_cols].values

# Avec pd.get_dummies(..., drop_first=True), le pays de référence est le premier pays
# par ordre alphabétique. Ici : Angola.
base_country = "Angola"

countries = []
for i in range(country_mat.shape[0]):
    if np.all(country_mat[i] == 0):
        countries.append(base_country)
    else:
        j = int(np.argmax(country_mat[i]))
        countries.append(country_cols[j].replace("C_", ""))

df["_Country"] = countries

X_cols_all = [c for c in df.columns if c not in ["Y", "T", "_Country"]]
X_macro_cols = [c for c in X_cols_all if not c.startswith("C_")]

if len(X_macro_cols) == 0:
    raise ValueError("X_macro_cols vide. Vérifie dml_design_matrix.csv.")

print(f"✅ Colonnes X macro utilisées (t-1) : {X_macro_cols}")
print(f"✅ Pays détectés : {sorted(df['_Country'].unique())}")
print()

rows = []

for c in sorted(df["_Country"].unique()):
    idx = (df["_Country"] == c).values
    n = int(idx.sum())

    min_n = 18
    if n < min_n:
        print(f"⚠️ {c}: n={n} trop petit (<{min_n}) -> skip")
        continue

    Y = Y_all[idx]
    T = T_all[idx]
    X = df.loc[idx, X_macro_cols].astype(float).values

    n_splits = min(N_SPLITS, max(2, n // 6))
    if n_splits < 2:
        print(f"⚠️ {c}: n={n} trop petit pour CV -> skip")
        continue

    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    model_y = RandomForestRegressor(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE
    )

    model_t = RandomForestRegressor(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE
    )

    try:
        dml = LinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=False,
            cv=cv,
            random_state=RANDOM_STATE,
        )

        dml.fit(Y, T, X=X, inference="auto")

        ate = safe_float(dml.ate(X=X))

        lb, ub = dml.ate_interval(X=X)
        lb = safe_float(lb)
        ub = safe_float(ub)

        pval = np.nan
        stderr = np.nan

        try:
            inf = dml.ate_inference(X=X)

            pval = safe_float(getattr(inf, "pvalue", None))
            if np.isnan(pval):
                pval = safe_float(getattr(inf, "pvalue_", None))

            stderr = safe_float(getattr(inf, "stderr", None))
            if np.isnan(stderr):
                stderr = safe_float(getattr(inf, "stderr_mean", None))
            if np.isnan(stderr):
                stderr = safe_float(getattr(inf, "stderr_", None))

        except Exception:
            pass

        rows.append({
            "Country": c,
            "n_obs": n,
            "n_splits": n_splits,
            "ATE": ate,
            "ATE_LB_95": lb,
            "ATE_UB_95": ub,
            "StdErr": stderr,
            "p_value": pval,
            "treatment": "Capital_Formation(t-1)",
            "outcome": "GDP_Growth(t)"
        })

        print(
            f"✅ {c}: "
            f"ATE={ate:.4f} | "
            f"p={pval if not np.isnan(pval) else 'NA'} | "
            f"n={n} | "
            f"splits={n_splits}"
        )

    except Exception as e:
        print(f"❌ {c}: erreur DML -> {e}")


out = pd.DataFrame(rows)
out_path = RES_DIR / "dml_hte_by_country.csv"

if out.empty:
    out.to_csv(out_path, index=False)
    print("\n⚠️ Aucun résultat HTE exporté.")
else:
    out["p_value_sort"] = out["p_value"].fillna(1.0)
    out = out.sort_values(["p_value_sort", "Country"], ascending=[True, True]).drop(columns=["p_value_sort"])
    out.to_csv(out_path, index=False)

    print(f"\n✅ Saved: {out_path}")
    print(out.to_string(index=False))

print()
print("Structure causale estimée par pays :")
print("  Capital_Formation(t-1) -> GDP_Growth(t)")
print("✅ HTE BY COUNTRY DONE")