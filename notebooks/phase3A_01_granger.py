#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PHASE 3A — CAUSAL DISCOVERY (GRANGER) 


 ce script  fait: 

1  Exécute Granger (lags 1..MAX_LAG) pour un pays à la fois
2 Produit matrices p-values (min sur lags) + best lag + liste d'arêtes significatives
3 Génère une heatmap (p-values) et un résumé texte

 pendant la preparation, nous avons préparés deux fichiers: data_prepared_for_dml.csv et data_transformed_unscaled.csv

ici  nous allons utiliser data_transformed_unscaled.csv qui contient les variables transformées (log, diff) 
mais non standardisées.


INPUT  :
  ../data/processed/data_transformed_unscaled_lagged.csv 

"""

import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests



# Helpers

def safe_slug(name: str) -> str:
    return (
        name.replace(",", "")
            .replace(".", "")
            .replace(" ", "_")
            .replace("__", "_")
            .strip("_")
    )

def granger_min_pvalue(y: np.ndarray, x: np.ndarray, max_lag: int):
    """
    Test Granger: x -> y
    Retourne (p_min, lag_best). Utilise ssr_ftest.
    """
    data = np.column_stack([y, x])  # [y, x]
    res = grangercausalitytests(data, maxlag=max_lag, verbose=False)

    best_lag, best_p = None, None
    for lag, out in res.items():
        p = out[0]["ssr_ftest"][1]
        if best_p is None or p < best_p:
            best_p = p
            best_lag = lag

    return float(best_p), int(best_lag)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", required=True, help='Nom exact du pays (ex: "France")')
    parser.add_argument("--max_lag", type=int, default=3, help="Lag max (défaut=3)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Seuil alpha (défaut=0.05)")
    args = parser.parse_args()

    # ------------------------------
    # Paths (auto-create)
    # ------------------------------
    DATA_PROCESSED = Path("../data/processed")

    # recommandé (transformé, non standardisé)
    INPUT_PRIMARY = DATA_PROCESSED / "data_transformed_unscaled_lagged.csv"
    INPUT_FALLBACK = DATA_PROCESSED / "data_transformed_unscaled.csv"

    if INPUT_PRIMARY.exists():
        INPUT_FILE = INPUT_PRIMARY
    elif INPUT_FALLBACK.exists():
        INPUT_FILE = INPUT_FALLBACK
    else:
        raise FileNotFoundError(
            "Aucun fichier input trouvé. Attendu: "
            f"{INPUT_PRIMARY} ou {INPUT_FALLBACK}"
        )

    RESULTS_DIR = Path("../results") / "phase3A_granger" / safe_slug(args.country)
    REPORTS_DIR = Path("../reports")
    FIG_DIR = REPORTS_DIR / "figures" / "phase3A_granger" / safe_slug(args.country)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # Load
    # ------------------------------
    df = pd.read_csv(INPUT_FILE)

    required = {"Country", "Year"}
    if not required.issubset(df.columns):
        raise ValueError(f"Le dataset doit contenir {required}. Colonnes: {list(df.columns)}")

    if args.country not in set(df["Country"].unique()):
        raise ValueError(f'Pays "{args.country}" introuvable. Pays disponibles: {sorted(df["Country"].unique())}')

    vars_num = [c for c in df.columns if c not in ["Country", "Year"]]

    sub = df[df["Country"] == args.country].sort_values("Year").reset_index(drop=True)

    # Guard minimal size
    T = len(sub)
    if T < args.max_lag + 5:
        raise ValueError(
            f"Pas assez d'observations pour max_lag={args.max_lag}. "
            f"T={T}. Réduis max_lag ou vérifie la série."
        )

    # ------------------------------
    # Compute matrices
    # ------------------------------
    pmat = pd.DataFrame(index=vars_num, columns=vars_num, dtype=float)
    lagmat = pd.DataFrame(index=vars_num, columns=vars_num, dtype=float)

    edges = []

    for x in vars_num:          # cause
        for y in vars_num:      # effect
            if x == y:
                pmat.loc[x, y] = np.nan
                lagmat.loc[x, y] = np.nan
                continue

            try:
                y_arr = sub[y].astype(float).to_numpy()
                x_arr = sub[x].astype(float).to_numpy()
                p_best, lag_best = granger_min_pvalue(y_arr, x_arr, args.max_lag)
            except Exception:
                p_best, lag_best = np.nan, np.nan

            pmat.loc[x, y] = p_best
            lagmat.loc[x, y] = lag_best

            if np.isfinite(p_best) and p_best < args.alpha:
                edges.append({
                    "Country": args.country,
                    "Cause": x,
                    "Effect": y,
                    "p_value_min": p_best,
                    "best_lag": int(lag_best)
                })

    edges_df = pd.DataFrame(edges).sort_values(["p_value_min", "best_lag"])

    # ------------------------------
    # Save outputs
    # ------------------------------
    pmat_path = RESULTS_DIR / "granger_pmin_matrix.csv"
    lag_path = RESULTS_DIR / "granger_bestlag_matrix.csv"
    edges_path = RESULTS_DIR / "granger_edges_significant.csv"
    summary_path = RESULTS_DIR / "granger_summary.txt"

    pmat.to_csv(pmat_path)
    lagmat.to_csv(lag_path)
    edges_df.to_csv(edges_path, index=False)

    # Heatmap figure (p-values)
    plt.figure(figsize=(10, 8))
    sns.heatmap(pmat, annot=True, fmt=".3f", cmap="coolwarm_r", cbar=True)
    plt.title(f"Granger p-values (min over lags 1–{args.max_lag}) — {args.country}")
    plt.tight_layout()
    fig_path = FIG_DIR / "granger_pmin_heatmap.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    # Summary txt
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("PHASE 3A — GRANGER (par pays)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Country   : {args.country}\n")
        f.write(f"Input     : {INPUT_FILE}\n")
        f.write(f"Obs (T)   : {T}\n")
        f.write(f"Vars (N)  : {len(vars_num)}\n")
        f.write(f"Alpha     : {args.alpha}\n")
        f.write(f"Max lag   : {args.max_lag}\n")
        f.write("\n")
        f.write(f"Edges significatives (p < alpha) : {len(edges_df)}\n")
        if len(edges_df) > 0:
            f.write("\nTop 20 edges (p croissant):\n")
            f.write(edges_df.head(20).to_string(index=False))
            f.write("\n")

    # Console recap
    print("=" * 100)
    print("PHASE 3A — GRANGER (SÉQUENTIEL) — TERMINÉ")
    print("=" * 100)
    print(f" Country : {args.country}")
    print(f" Input   : {INPUT_FILE}")
    print(f" Output  : {RESULTS_DIR}")
    print(f"   - {pmat_path.name}")
    print(f"   - {lag_path.name}")
    print(f"   - {edges_path.name}")
    print(f"   - {summary_path.name}")
    print(f" Figure  : {fig_path}")
    print(f" Edges significatives : {len(edges_df)}")


if __name__ == "__main__":
    main()