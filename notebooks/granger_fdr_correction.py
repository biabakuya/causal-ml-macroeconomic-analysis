#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 3A-bis — GRANGER : correction de multiplicite (Benjamini-Hochberg FDR)

Objectif :
Appliquer la correction FDR de Benjamini-Hochberg aux p-values Granger,
sur deux familles de tests :
  (1) les 7 tests Capital_Formation -> GDP_Growth (hypothese centrale)
  (2) la famille complete (42 paires x 7 pays = 294 tests)

Arborescence attendue (une matrice 7x7 par pays) :
    results/phase3A_granger/<Country>/granger_pmin_matrix.csv

Output :
    results/phase3A_granger/granger_fdr_cf_to_gdp.csv    (table q-values CF->GDP)
    results/phase3A_granger/granger_fdr_full_family.csv  (294 tests + q-values)

"""

import pandas as pd
import numpy as np
from pathlib import Path


GRANGER_DIR = Path(__file__).resolve().parent.parent / "results" / "phase3A_granger"


COUNTRY_DIRS = {
    "Germany": "Germany",
    "Angola":  "Angola",
    "Ghana":   "Ghana",
    "Morocco": "Morocco",
    "France":  "France",
    "DRC":     "DRC",
    "Nigeria": "Nigeria",
}
MATRIX_FILE = "granger_pmin_matrix.csv"

TREATMENT = "Capital_Formation_lag1"
OUTCOME   = "GDP_Growth"
ALPHA     = 0.05


def benjamini_hochberg(pvals, alpha=0.05):
    """Retourne (q-values, rejet booleen) pour un vecteur de p-values."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]

    q = np.empty(m)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        prev = min(prev, ranked[i] * m / (i + 1))
        q[i] = prev
    qfull = np.empty(m)
    qfull[order] = q

    kmax = 0
    for i in range(m):
        if ranked[i] <= (i + 1) / m * alpha:
            kmax = i + 1
    rej = np.zeros(m, dtype=bool)
    if kmax > 0:
        rej = p <= ranked[kmax - 1]
    return qfull, rej


def load_matrices():
    mats = {}
    for country, folder in COUNTRY_DIRS.items():
        path = GRANGER_DIR / folder / MATRIX_FILE
        if not path.exists():
            raise FileNotFoundError(f"Introuvable : {path}")
        mats[country] = pd.read_csv(path, index_col=0)
    return mats


def main():
    print(f"Lecture depuis : {GRANGER_DIR}")
    mats = load_matrices()

    # ---- Famille 1 : les 7 tests CF -> GDP ----
    cf = {c: float(mats[c].loc[TREATMENT, OUTCOME]) for c in COUNTRY_DIRS}
    fam1 = pd.DataFrame({"country": list(cf.keys()), "p": list(cf.values())})
    fam1 = fam1.sort_values("p").reset_index(drop=True)
    q1, rej1 = benjamini_hochberg(fam1["p"].values, ALPHA)
    fam1["q_BH"] = q1
    fam1["significant_FDR"] = rej1
    fam1.to_csv(GRANGER_DIR / "granger_fdr_cf_to_gdp.csv", index=False)

    print("\n=== Family 1: CF -> GDP (7 tests) ===")
    print(fam1.to_string(index=False))

    # ---- Famille 2 : les 294 tests ----
    rows = []
    for c in COUNTRY_DIRS:
        M = mats[c]
        for cause in M.index:
            for effect in M.columns:
                if cause == effect:
                    continue
                val = M.loc[cause, effect]
                if pd.notna(val):
                    rows.append((c, cause, effect, float(val)))
    fam2 = pd.DataFrame(rows, columns=["country", "cause", "effect", "p"])
    q2, rej2 = benjamini_hochberg(fam2["p"].values, ALPHA)
    fam2["q_BH"] = q2
    fam2["significant_FDR"] = rej2
    fam2.to_csv(GRANGER_DIR / "granger_fdr_full_family.csv", index=False)

    print("\n=== Family 2: full family (294 tests) ===")
    print(f"Nominally significant (p<{ALPHA}) : {(fam2['p'] < ALPHA).sum()}")
    print(f"Surviving BH-FDR (q<{ALPHA})       : {fam2['significant_FDR'].sum()}")
    n_cf = fam2[(fam2.cause == TREATMENT) & (fam2.effect == OUTCOME) & fam2.significant_FDR].shape[0]
    print(f"CF->GDP links surviving FDR         : {n_cf}")


if __name__ == "__main__":
    main()
