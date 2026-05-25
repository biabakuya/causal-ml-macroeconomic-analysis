#!/usr/bin/env python3
"""
PHASE 3B — VAR


le but ici est :
- Estimer un VAR par pays (séquentiel)
- Sélection du lag robuste (AIC/BIC/HQIC/FPE) par boucle d'estimation
- Export des fichiers: lag selection, summary, paramètres, stabilité, IRF, FEVD

Input:

comme avec Granger nous avons utilisés:  data_transformed_unscaled_lagged.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR



# Utils


def sanitize_country(name: str) -> str:
    return (
        name.replace(",", "")
            .replace(".", "")
            .replace(" ", "_")
            .replace("/", "_")
            .replace("__", "_")
    )

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_max_lag(T: int, k: int, requested: int) -> int:
    """
    Heuristique conservative pour éviter over-parameterization:
    - T observations, k variables
    - maxlag ~ (T-5)//k (au moins 1)
    """
    heuristic = max(1, (T - 5) // max(1, k))
    return max(1, min(requested, heuristic))

def compute_lag_criteria(model: VAR, max_lag: int):
    """
    Boucle sur p=1..max_lag, fit VAR(p), collect criteria.
    Retourne DataFrame avec aic,bic,hqic,fpe + status.
    """
    rows = []
    for p in range(1, max_lag + 1):
        try:
            res = model.fit(p)
            rows.append({
                "lag": p,
                "aic": res.aic,
                "bic": res.bic,
                "hqic": res.hqic,
                "fpe": res.fpe,
                "status": "OK"
            })
        except Exception as e:
            rows.append({
                "lag": p,
                "aic": np.nan,
                "bic": np.nan,
                "hqic": np.nan,
                "fpe": np.nan,
                "status": f"FAIL: {str(e)[:120]}"
            })
    return pd.DataFrame(rows)

def choose_lag(criteria_df: pd.DataFrame, criterion: str = "bic") -> int:
    """
    Choisit le lag qui minimise le critère (sur les lignes status OK).
    """
    ok = criteria_df[criteria_df["status"] == "OK"].copy()
    if ok.empty:
        return 1
    # minimiser le critère
    ok = ok.dropna(subset=[criterion])
    if ok.empty:
        return 1
    best_row = ok.loc[ok[criterion].idxmin()]
    return int(best_row["lag"])

def write_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")

def plot_irf(irf, out_png: Path, title: str):
    fig = irf.plot(orth=False)
    fig.set_size_inches(14, 10)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_fevd(fevd, out_png: Path, title: str):
    fig = fevd.plot()
    fig.set_size_inches(14, 10)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def stability_report(res) -> str:
    # Eigenvalues of companion matrix should be < 1 in modulus for stability
    eigvals = res.roots  # roots of VAR polynomial; stability if all abs(roots) > 1? (statsmodels convention)
    # statsmodels VARResults.is_stable(): checks companion matrix eigenvalues < 1 (stable)
    stable = res.is_stable(verbose=False)

    lines = []
    lines.append("STABILITY CHECK\n")
    lines.append(f"- is_stable() : {stable}\n")
    lines.append("- Roots (VAR characteristic roots):\n")
    for i, r in enumerate(eigvals, 1):
        lines.append(f"  {i:02d}: root={r:.6f} | abs={abs(r):.6f}\n")
    lines.append("\nNOTE: statsmodels stability uses companion matrix eigenvalues.\n")
    return "".join(lines)



# Main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", type=str, required=True, help='Country name exactly as in CSV, e.g. "France"')
    parser.add_argument("--input", type=str, default="../data/processed/data_transformed_unscaled_lagged.csv")
    parser.add_argument("--outdir", type=str, default="../results/phase3B_var")
    parser.add_argument("--figdir", type=str, default="../reports/figures/phase3B_var")
    parser.add_argument("--max_lag", type=int, default=4)
    parser.add_argument("--criterion", type=str, default="bic", choices=["aic", "bic", "hqic", "fpe"])
    parser.add_argument("--irf_horizon", type=int, default=10)
    parser.add_argument("--fevd_horizon", type=int, default=10)
    args = parser.parse_args()

    print("=" * 100)
    print("PHASE 3B — VAR (SÉQUENTIEL) — TERMINÉ")
    print("=" * 100)

    country = args.country
    input_path = Path(args.input)
    out_root = Path(args.outdir)
    fig_root = Path(args.figdir)

    out_country = out_root / sanitize_country(country)
    fig_country = fig_root / sanitize_country(country)
    ensure_dir(out_country)
    ensure_dir(fig_country)

    df = pd.read_csv(input_path)

    vars_list = ["GDP_Growth", "Capital_Formation_lag1", "Inflation_lag1",
             "Government_Debt_lag1", "Trade_Balance_lag1", "Exchange_Rate_lag1", "Reserves_lag1"]

    sub = df[df["Country"] == country].sort_values("Year").copy()

    # Drop NaN rows (should be none in transformed_unscaled)
    before = len(sub)
    sub = sub.dropna(subset=vars_list).reset_index(drop=True)
    after = len(sub)

    T = len(sub)
    k = len(vars_list)

    print(f" Country : {country}")
    print(f" Input   : {input_path}")
    print(f" Output  : {out_country}")
    print(f" N obs   : {T} (dropped {before - after} NaN rows)")
    print(f" Vars    : {vars_list}")
    print()

    if T < 10:
        raise RuntimeError(f"Pas assez d'observations pour VAR: T={T}")

    # Build VAR model
    y = sub[vars_list].astype(float)
    model = VAR(y)

    # Safe max_lag
    max_lag = safe_max_lag(T=T, k=k, requested=args.max_lag)
    if max_lag != args.max_lag:
        print(f" max_lag ajusté (sécurité) : demandé={args.max_lag} -> utilisé={max_lag} (T={T}, k={k})")

    # Lag selection robust
    criteria_df = compute_lag_criteria(model, max_lag=max_lag)
    criteria_path = out_country / "var_lag_selection.csv"
    criteria_df.to_csv(criteria_path, index=False)

    chosen = choose_lag(criteria_df, criterion=args.criterion)

    # Fallback: if chosen lag row not OK, choose smallest OK
    ok_lags = criteria_df[criteria_df["status"] == "OK"]["lag"].tolist()
    if chosen not in ok_lags and ok_lags:
        chosen = int(min(ok_lags))

    print(f" Lag selection saved : {criteria_path}")
    print(f" Selected lag ({args.criterion.upper()}) : {chosen}")
    print()

    # Fit final VAR
    res = model.fit(chosen)

    # Save summary
    summary_txt = str(res.summary())
    write_text(out_country / "var_summary.txt", summary_txt)

    # Save params
    # res.params is DataFrame with coefficients
    res.params.to_csv(out_country / "var_params.csv")

    # Stability report
    stab_txt = stability_report(res)
    write_text(out_country / "var_stability.txt", stab_txt)

    # IRF
    try:
        irf = res.irf(args.irf_horizon)
        plot_irf(irf, fig_country / "var_irf.png", f"IRF — {country} (lag={chosen})")
        irf_df = irf.irfs.reshape((args.irf_horizon + 1, k, k))
        # Save IRF array in long format
        rows = []
        for h in range(args.irf_horizon + 1):
            for i, resp in enumerate(vars_list):
                for j, shock in enumerate(vars_list):
                    rows.append({"horizon": h, "response": resp, "shock": shock, "irf": float(irf_df[h, i, j])})
        pd.DataFrame(rows).to_csv(out_country / "var_irf_long.csv", index=False)
        print(f" IRF saved : {fig_country / 'var_irf.png'} + var_irf_long.csv")
    except Exception as e:
        print(f" IRF skipped (error): {e}")

    # FEVD
    try:
        fevd = res.fevd(args.fevd_horizon)
        plot_fevd(fevd, fig_country / "var_fevd.png", f"FEVD — {country} (lag={chosen})")
        # Save FEVD in long format
        # fevd.decomp shape: (k, steps, k) -> variable, horizon, shock
        dec = fevd.decomp
        rows = []
        steps = dec.shape[1]
        for i, var in enumerate(vars_list):
            for h in range(steps):
                for j, shock in enumerate(vars_list):
                    rows.append({
                        "variable": var,
                        "horizon": h,
                        "shock": shock,
                        "fevd": float(dec[i, h, j])
                    })
        pd.DataFrame(rows).to_csv(out_country / "var_fevd_long.csv", index=False)
        print(f" FEVD saved : {fig_country / 'var_fevd.png'} + var_fevd_long.csv")
    except Exception as e:
        print(f" FEVD skipped (error): {e}")

    # Save a small run report
    run_report = []
    run_report.append("PHASE 3B — VAR REPORT\n")
    run_report.append(f"Country: {country}\n")
    run_report.append(f"Input: {input_path}\n")
    run_report.append(f"N obs: {T}\n")
    run_report.append(f"Variables: {vars_list}\n")
    run_report.append(f"Max lag used: {max_lag}\n")
    run_report.append(f"Criterion: {args.criterion}\n")
    run_report.append(f"Chosen lag: {chosen}\n")
    run_report.append(f"Stable: {res.is_stable(verbose=False)}\n")
    write_text(out_country / "var_run_report.txt", "".join(run_report))

    print("\n" + "=" * 100)
    print(" PHASE 3B — VAR (SÉQUENTIEL) — TERMINÉ")
    print("=" * 100)
    print(f" Outputs: {out_country}")
    print(f" Figures: {fig_country}")


if __name__ == "__main__":
    main()