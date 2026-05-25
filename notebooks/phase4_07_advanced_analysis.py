#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PHASE 4 — ADVANCED ANALYSIS 


Inputs (from RES_DIR):
- dml_hte_individual_effects.csv
- dml_hte_by_group_cf.csv
- dml_ate_by_group.csv

Outputs (saved in RES_DIR):
1) dml_hte_distribution.png + dml_hte_by_group_summary.csv
2) dml_hte_group_diff_test.csv
3) dml_compare_linear_vs_causalforest_group.csv
4) dml_hte_top_extremes.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind

from phase4_00_dml_config import RES_DIR

print("=" * 100)
print("PHASE 4 — ADVANCED ANALYSIS")
print("=" * 100)

# -----------------------------
# Paths
# -----------------------------
p_ind = RES_DIR / "dml_hte_individual_effects.csv"
p_cf  = RES_DIR / "dml_hte_by_group_cf.csv"
p_lin = RES_DIR / "dml_ate_by_group.csv"

# -----------------------------
# Load safely
# -----------------------------
ind = pd.read_csv(p_ind)
cf  = pd.read_csv(p_cf)
lin = pd.read_csv(p_lin)

# Harmoniser le nom de colonne pour merge
# cf: Category -> Group
if "Category" in cf.columns and "Group" not in cf.columns:
    cf = cf.rename(columns={"Category": "Group"})

# Vérifs minimales
required_ind = {"Category", "HTE", "Country"}
missing_ind = required_ind - set(ind.columns)
if missing_ind:
    raise ValueError(f"Colonnes manquantes dans {p_ind.name}: {sorted(list(missing_ind))}")

required_cf = {"Group", "ATE_CF", "HTE_std", "n"}
missing_cf = required_cf - set(cf.columns)
if missing_cf:
    raise ValueError(f"Colonnes manquantes dans {p_cf.name}: {sorted(list(missing_cf))}")

required_lin = {"Group", "ATE", "StdErr", "p_value", "n"}
missing_lin = required_lin - set(lin.columns)
if missing_lin:
    raise ValueError(f"Colonnes manquantes dans {p_lin.name}: {sorted(list(missing_lin))}")

# Forcer types
ind["HTE"] = pd.to_numeric(ind["HTE"], errors="coerce")
ind = ind.dropna(subset=["HTE", "Category"])

# ======================================================================================
# (1) Distribution HTE + stats par groupe
# ======================================================================================
fig_path = RES_DIR / "dml_hte_distribution.png"

plt.figure()
# histogram global
plt.hist(ind["HTE"].values, bins=25, alpha=0.7)
plt.title("Distribution globale des effets individuels (HTE)")
plt.xlabel("HTE (effet estimé)")
plt.ylabel("Fréquence")
plt.tight_layout()
plt.savefig(fig_path, dpi=200)
plt.close()

# Résumé par groupe
summary = (
    ind.groupby("Category")["HTE"]
    .agg(["count", "mean", "std", "min", "max"])
    .reset_index()
    .rename(columns={"Category": "Group"})
)
summary_path = RES_DIR / "dml_hte_by_group_summary.csv"
summary.to_csv(summary_path, index=False)

print("✅ HTE distribution saved:", fig_path)
print("✅ HTE summary saved:", summary_path)

# ======================================================================================
# (2) Test différence Developed vs Developing (t-test)
# ======================================================================================
dev = ind.loc[ind["Category"] == "Developed", "HTE"].values
deving = ind.loc[ind["Category"] == "Developing", "HTE"].values

test_path = RES_DIR / "dml_hte_group_diff_test.csv"

if len(dev) >= 5 and len(deving) >= 5:
    # Welch t-test (variances possiblement différentes)
    t_stat, p_val = ttest_ind(dev, deving, equal_var=False, nan_policy="omit")
    out_test = pd.DataFrame([{
        "GroupA": "Developed",
        "GroupB": "Developing",
        "n_A": int(len(dev)),
        "n_B": int(len(deving)),
        "mean_A": float(np.mean(dev)),
        "mean_B": float(np.mean(deving)),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
    }])
    out_test.to_csv(test_path, index=False)

    print("\nGroup difference test:")
    print("T-stat:", t_stat)
    print("P-value:", p_val)
    print("✅ Group diff test saved:", test_path)
else:
    print("⚠️ Pas assez d'observations pour t-test (>=5 par groupe).")

# ======================================================================================
# (3) Comparaison ATE (LinearDML) vs ATE_CF (CausalForestDML) par groupe
# ======================================================================================
# lin: Group, ATE, StdErr, p_value, n...
# cf : Group, ATE_CF, HTE_std, n...
comparison = lin.merge(cf, on="Group", how="left", suffixes=("_LinearDML", "_CausalForest"))

comp_path = RES_DIR / "dml_compare_linear_vs_causalforest_group.csv"
comparison.to_csv(comp_path, index=False)

print("\n✅ Comparison table saved:", comp_path)
print(comparison.to_string(index=False))

# ======================================================================================
# (4) Top extrêmes (individus/pays) — les 10 plus grands |HTE|
# ======================================================================================
top_path = RES_DIR / "dml_hte_top_extremes.csv"

tmp = ind.copy()
tmp["abs_HTE"] = tmp["HTE"].abs()
top = tmp.sort_values("abs_HTE", ascending=False).head(10)[["Country", "Category", "HTE", "abs_HTE"]]
top.to_csv(top_path, index=False)

print("\n✅ Top extremes saved:", top_path)
print(top.to_string(index=False))

print("=" * 100)
print("✅ PHASE 4 — ADVANCED ANALYSIS — DONE")
print("=" * 100)