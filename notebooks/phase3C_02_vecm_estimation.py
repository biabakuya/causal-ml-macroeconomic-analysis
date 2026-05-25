#!/usr/bin/env python3
"""
PHASE 3C — VECM ESTIMATION 



Entrées :
- data/processed/data_transformed_unscaled.csv

Sorties :
- results/phase3C_vecm/<Country>/vecm_summary.txt
- results/phase3C_vecm/<Country>/vecm_alpha.csv
- results/phase3C_vecm/<Country>/vecm_beta.csv
- results/phase3C_vecm/<Country>/vecm_resid_stats.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from statsmodels.tsa.vector_ar.vecm import VECM


# -----------------------------
# Utils
# -----------------------------
def safe_name(country: str) -> str:
    return (
        country.replace(",", "")
        .replace(".", "")
        .replace(" ", "_")
        .replace("__", "_")
        .strip("_")
    )


def load_selected_lag_from_var(country_dir: Path) -> int | None:
    """
    Lit results/phase3B_var/<Country>/var_lag_selection.csv
    Format réel observé:
      lag,aic,bic,hqic,fpe,status
    """
    f = country_dir / "var_lag_selection.csv"
    if not f.exists():
        return None

    lag_df = pd.read_csv(f)
    # garder les lignes OK si la colonne existe
    if "status" in lag_df.columns:
        lag_df_ok = lag_df[lag_df["status"].astype(str).str.upper() == "OK"].copy()
        if len(lag_df_ok) > 0:
            lag_df = lag_df_ok

    # si pas de bic, fallback aic, sinon min sur bic
    if "bic" in lag_df.columns:
        best_row = lag_df.loc[lag_df["bic"].astype(float).idxmin()]
    elif "aic" in lag_df.columns:
        best_row = lag_df.loc[lag_df["aic"].astype(float).idxmin()]
    else:
        # dernier recours: plus grand lag disponible
        best_row = lag_df.iloc[-1]

    return int(best_row["lag"])


def load_rank_from_cointegration(country_dir: Path) -> int | None:
    """
    Lit results/phase3C_cointegration/<Country>/johansen_rank.txt si présent.
    Sinon None.
    """
    f = country_dir / "johansen_rank.txt"
    if not f.exists():
        return None
    try:
        txt = f.read_text(encoding="utf-8").strip()
        # attendu : "rank=5" ou "5"
        if "rank" in txt.lower():
            # extrait le dernier entier
            import re

            nums = re.findall(r"\d+", txt)
            return int(nums[-1]) if nums else None
        return int(txt)
    except Exception:
        return None


def resid_stats(resid: np.ndarray, cols: list[str]) -> pd.DataFrame:
    """
    Stats simples sur résidus (utile pour contrôle qualité).
    resid: (T, k)
    """
    out = []
    for j, name in enumerate(cols):
        r = resid[:, j]
        r = r[~np.isnan(r)]
        if len(r) == 0:
            out.append({"variable": name, "count": 0})
            continue
        out.append(
            {
                "variable": name,
                "count": len(r),
                "mean": float(np.mean(r)),
                "std": float(np.std(r, ddof=1)) if len(r) > 1 else np.nan,
                "min": float(np.min(r)),
                "max": float(np.max(r)),
            }
        )
    return pd.DataFrame(out)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", required=True, help='e.g. "France"')
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Cointegration rank. Si non fourni, tente de lire johansen_rank.txt, sinon fallback=1.",
    )
    parser.add_argument(
        "--deterministic",
        type=str,
        default="ci",
        help="Terme déterministe VECM (statsmodels): 'ci' recommandé (constante dans la relation de cointégration).",
    )
    args = parser.parse_args()

    DATA_PROCESSED = Path("../data/processed")
    INPUT_FILE = DATA_PROCESSED / "data_transformed_unscaled_lagged.csv"

    RESULTS_ROOT = Path("../results")
    VAR_ROOT = RESULTS_ROOT / "phase3B_var"
    COINT_ROOT = RESULTS_ROOT / "phase3C_cointegration"
    VECM_ROOT = RESULTS_ROOT / "phase3C_vecm"

    country = args.country
    cdir = safe_name(country)

    out_dir = VECM_ROOT / cdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("PHASE 3C — VECM ESTIMATION")
    print("=" * 100)
    print(f"Country : {country}")

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(INPUT_FILE)
    df = df[df["Country"] == country].sort_values("Year").reset_index(drop=True)

    vars_ = [c for c in df.columns if c not in ["Country", "Year"]]
    X = df[vars_].dropna()
    print(f"Observations : {len(X)}")

    if len(X) < 12:
        raise ValueError(
            f"Trop peu d'observations après dropna pour VECM: {len(X)}. "
            "VECM est fragile, vise plutôt >= 20."
        )

    # -----------------------------
    # Lag selection (from VAR output)
    # -----------------------------
    var_country_dir = VAR_ROOT / cdir
    selected_lag = load_selected_lag_from_var(var_country_dir)

    if selected_lag is None:
        # fallback prudent
        selected_lag = 2

    # VECM utilise k_ar_diff = p - 1 (si VAR(p))
    k_ar_diff = max(int(selected_lag) - 1, 1)

    # -----------------------------
    # Rank
    # -----------------------------
    rank = args.rank
    if rank is None:
        rank_from_file = load_rank_from_cointegration(COINT_ROOT / cdir)
        rank = rank_from_file if rank_from_file is not None else 1

    # garde-fou: rank doit être entre 1 et k-1
    k = len(vars_)
    rank = int(rank)
    if rank < 1:
        rank = 1
    if rank >= k:
        rank = k - 1

    print(f"✅ Vars    : {vars_}")
    print(f"✅ Selected lag (from VAR BIC) : {selected_lag}  -> k_ar_diff={k_ar_diff}")
    print(f"✅ Cointegration rank used     : {rank}")
    print(f"✅ Deterministic               : {args.deterministic}")

    # -----------------------------
    # Fit VECM
    # -----------------------------
    model = VECM(
        X,
        k_ar_diff=k_ar_diff,
        coint_rank=rank,
        deterministic=args.deterministic,
    )
    res = model.fit()

    # -----------------------------
    # Save outputs
    # -----------------------------
    summary_txt = str(res.summary())
    (out_dir / "vecm_summary.txt").write_text(summary_txt, encoding="utf-8")

    # alpha (loading) et beta (cointegration vectors)
    alpha = pd.DataFrame(res.alpha, index=vars_, columns=[f"coint_{i+1}" for i in range(rank)])
    beta = pd.DataFrame(res.beta, index=vars_, columns=[f"coint_{i+1}" for i in range(rank)])

    alpha.to_csv(out_dir / "vecm_alpha.csv")
    beta.to_csv(out_dir / "vecm_beta.csv")

    # resid stats
    resid = res.resid  # (T, k)
    rs = resid_stats(resid, vars_)
    rs.to_csv(out_dir / "vecm_resid_stats.csv", index=False)

    print("\n" + "=" * 100)
    print("✅ PHASE 3C — VECM ESTIMATION — TERMINÉ")
    print("=" * 100)
    print(f"✅ Output  : {out_dir}")
    print("   - vecm_summary.txt")
    print("   - vecm_alpha.csv")
    print("   - vecm_beta.csv")
    print("   - vecm_resid_stats.csv")


if __name__ == "__main__":
    main()