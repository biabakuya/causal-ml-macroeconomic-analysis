#!/usr/bin/env python3
"""
PHASE 3C — COINTEGRATION TEST (JOHANSEN)

avant de commencer d'estimer le VECM, nous avons creer ce ficher pour vérifier si les variables sont cointegrées.

Lit les données transformées, applique le test de cointégration de Johansen pour chaque pays, 
et sauvegarde les résultats dans results/phase3C_cointegration/<Country>/
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.vector_ar.vecm import coint_johansen


# CONFIG


INPUT_FILE = Path("../data/processed/data_transformed_unscaled_lagged.csv")
RESULTS_DIR = Path("../results/phase3C_cointegration")

VARS = [
    "GDP_Growth",
    "Capital_Formation_lag1",
    "Inflation_lag1",
    "Government_Debt_lag1",
    "Trade_Balance_lag1",
    "Exchange_Rate_lag1",
    "Reserves_lag1"
]


# MAIN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", type=str, required=True)
    args = parser.parse_args()

    country = args.country
    country_slug = country.replace(",", "").replace(" ", "_")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = RESULTS_DIR / country_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    df_country = df[df["Country"] == country].sort_values("Year")

    data = df_country[VARS].dropna()

    print("="*90)
    print(f"PHASE 3C — JOHANSEN TEST — {country}")
    print("="*90)

    # Johansen test
    johansen_test = coint_johansen(data, det_order=0, k_ar_diff=1)

    trace_stat = johansen_test.lr1
    crit_values = johansen_test.cvt  # critical values (90%,95%,99%)

    # Determine rank at 95%
    rank = 0
    for i in range(len(trace_stat)):
        if trace_stat[i] > crit_values[i, 1]:
            rank += 1

    # Save results
    results_df = pd.DataFrame({
        "Trace Statistic": trace_stat,
        "Critical Value 95%": crit_values[:, 1]
    })

    results_df.to_csv(out_dir / "cointegration_results.csv", index=False)

    with open(out_dir / "cointegration_summary.txt", "w") as f:
        f.write(f"Country: {country}\n")
        f.write(f"Cointegration Rank (95%): {rank}\n")

    print(f"Trace statistics: {trace_stat}")
    print(f"Cointegration rank (95%): {rank}")
    print(f"Saved to: {out_dir}")
    print("="*90)


if __name__ == "__main__":
    main()