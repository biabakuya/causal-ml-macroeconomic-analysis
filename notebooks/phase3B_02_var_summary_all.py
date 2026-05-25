#!/usr/bin/env python3
"""
PHASE 3B (POST) — RESUME VAR MULTI-PAYS

Lit les sorties VAR déjà générées dans results/phase3B_var/<Country>/
et produit un tableau récapitulatif.

"""

import pandas as pd
from pathlib import Path

ROOT = Path("..")
RESULTS_DIR = ROOT / "results" / "phase3B_var"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = REPORTS_DIR / "phase3B_var_summary_all.csv"

def safe_read_csv(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None

def extract_selected_lag(lag_df: pd.DataFrame):
    """
    var_lag_selection.csv contient normalement les colonnes: lag,aic,bic,hqic,fpe + selected_bic
    Selon ta version, on gère plusieurs formats.
    """
    if lag_df is None or lag_df.empty:
        return None

    # cas 1: une colonne explicit "selected_bic"
    if "selected_bic" in lag_df.columns:
        try:
            val = lag_df["selected_bic"].dropna().iloc[0]
            return int(val)
        except Exception:
            pass

    # cas 2: une ligne "selected_bic" dans une colonne "metric"
    if {"metric", "value"}.issubset(lag_df.columns):
        sel = lag_df.loc[lag_df["metric"].astype(str).str.lower().str.contains("selected"), "value"]
        if len(sel) > 0:
            try:
                return int(sel.iloc[0])
            except Exception:
                return None

    # cas 3: on prend le lag qui minimise BIC si bic existe
    if "bic" in lag_df.columns and "lag" in lag_df.columns:
        sub = lag_df.dropna(subset=["bic", "lag"]).copy()
        if len(sub) > 0:
            best = sub.loc[sub["bic"].idxmin(), "lag"]
            try:
                return int(best)
            except Exception:
                return None

    return None

def summarize_fevd(fevd_long: pd.DataFrame):
    """
    fevd_long format attendu: horizon, target, source, share
    On résume la part moyenne (horizons 1..H) de GDP_Growth expliquée par les autres variables.
    """
    if fevd_long is None or fevd_long.empty:
        return {}

    needed = {"horizon", "target", "source", "share"}
    if not needed.issubset(set(fevd_long.columns)):
        return {}

    df = fevd_long.copy()
    df["target"] = df["target"].astype(str)
    df["source"] = df["source"].astype(str)

    g = df[df["target"] == "GDP_Growth"]
    if g.empty:
        return {}

    # moyenne sur horizons >=1 (on exclut h=0 si présent)
    g = g[g["horizon"] >= 1].copy()
    if g.empty:
        return {}

    mean_share = g.groupby("source")["share"].mean().sort_values(ascending=False)

    # top 3 sources hors GDP_Growth lui-même
    mean_share_no_self = mean_share.drop(index=["GDP_Growth"], errors="ignore")
    top3 = mean_share_no_self.head(3)

    out = {}
    for i, (src, val) in enumerate(top3.items(), start=1):
        out[f"fevd_gdp_top{i}_source"] = src
        out[f"fevd_gdp_top{i}_mean_share"] = float(val)

    # part "self" moyenne (inertie de GDP)
    if "GDP_Growth" in mean_share.index:
        out["fevd_gdp_self_mean_share"] = float(mean_share.loc["GDP_Growth"])

    return out

def main():
    rows = []
    if not RESULTS_DIR.exists():
        raise SystemExit(f"Folder not found: {RESULTS_DIR}")

    # chaque sous-dossier = un pays
    for country_dir in sorted([p for p in RESULTS_DIR.iterdir() if p.is_dir()]):
        country = country_dir.name

        lag_path = country_dir / "var_lag_selection.csv"
        fevd_path = ROOT / "reports" / "figures" / "phase3B_var" / country / "var_fevd_long.csv"
        # fallback si fevd_long est aussi enregistré côté results
        if not fevd_path.exists():
            fevd_path = country_dir / "var_fevd_long.csv"

        lag_df = safe_read_csv(lag_path)
        fevd_df = safe_read_csv(fevd_path)

        selected_lag = extract_selected_lag(lag_df)
        fevd_summary = summarize_fevd(fevd_df)

        row = {"Country": country, "selected_lag_bic": selected_lag}
        row.update(fevd_summary)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("Country")
    out.to_csv(OUT_FILE, index=False, encoding="utf-8")

    print("=" * 90)
    print("PHASE 3B (POST) — RESUME VAR MULTI-PAYS")
    print("=" * 90)
    print(f"✅ Saved: {OUT_FILE}")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()