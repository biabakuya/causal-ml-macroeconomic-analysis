#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PHASE 3F — TABLEAU COMPARATIF (Synthèse multi-pays)

VERSION CORRIGÉE POUR DONNÉES LAGGÉES :
- Utilise Capital_Formation_lag1 (t-1) → GDP_Growth (t)
- Respecte la structure temporelle des DAGs

But:
  Générer un tableau comparatif lisible pour le rapport
  (centralité, lien Capital_Formation_lag1 -> GDP_Growth, support, IRF, structure)

Inputs attendus (par pays slug):
  ../results/phase3E_dag/<slug>/dag_edges_support.csv   (ou dag_edges_final.csv)
  ../results/phase3E_dag/<slug>/dag_nodes_metrics.csv
  ../results/phase3B_var/<slug>/var_irf_long.csv
  ../results/phase3D_pcmci/<slug>/pcmci_links.csv

Outputs:
  ../reports/phase3F_comparative_table.csv
  ../reports/phase3F_comparative_table.md

Usage:
  python phase3F_01_comparative_table.py
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd


# -------------------------
# Utils
# -------------------------
def slugify(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\w\s\-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s)
    s = s.replace("__", "_")
    return s


def safe_read_csv(path: Path) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def classify_structure(n_edges: int, n_nodes: int) -> str:
    # Heuristique simple: densité du graphe
    if n_nodes <= 1:
        return "Insuffisant"
    dens = n_edges / (n_nodes * (n_nodes - 1))
    if dens >= 0.35:
        return "Très intégrée"
    if dens >= 0.20:
        return "Intégrée"
    if dens >= 0.10:
        return "Intermédiaire"
    return "Fragmentée"


def bucket_centrality(deg: float) -> str:
    # deg = degree (in+out)
    if deg >= 6:
        return "Très élevée"
    if deg >= 4:
        return "Élevée"
    if deg >= 2:
        return "Moyenne"
    return "Faible"


def summarize_irf_effect(irf_df: pd.DataFrame, shock="Capital_Formation_lag1", response="GDP_Growth") -> dict:
    """
    Résume l'effet IRF (VAR) : max abs, signe dominant, horizon du max
    Attend colonnes : horizon,response,shock,irf
    """
    if irf_df.empty:
        return {"irf_max_abs": np.nan, "irf_sign": "NA", "irf_best_h": np.nan}

    cols = {c.lower(): c for c in irf_df.columns}
    needed = {"horizon", "response", "shock", "irf"}
    if not needed.issubset(set(cols.keys())):
        return {"irf_max_abs": np.nan, "irf_sign": "NA", "irf_best_h": np.nan}

    hcol = cols["horizon"]
    rcol = cols["response"]
    scol = cols["shock"]
    icol = cols["irf"]

    sub = irf_df[(irf_df[scol] == shock) & (irf_df[rcol] == response)].copy()
    if sub.empty:
        return {"irf_max_abs": np.nan, "irf_sign": "NA", "irf_best_h": np.nan}

    arr = sub[icol].astype(float).values
    harr = sub[hcol].astype(int).values
    idx = int(np.argmax(np.abs(arr)))
    best_val = float(arr[idx])
    best_h = int(harr[idx])

    sign = "Positive" if best_val > 0 else "Négative" if best_val < 0 else "Neutre"
    return {"irf_max_abs": float(abs(best_val)), "irf_sign": sign, "irf_best_h": best_h}


def has_direct_edge(edges_df: pd.DataFrame, src="Capital_Formation_lag1", tgt="GDP_Growth") -> dict:
    """
    Vérifie si edge direct existe + support si dispo.
    """
    if edges_df.empty:
        return {"has_edge": False, "support": np.nan, "methods": ""}

    # normaliser colonnes
    cols = [c.lower() for c in edges_df.columns]
    if "source" not in cols or "target" not in cols:
        return {"has_edge": False, "support": np.nan, "methods": ""}

    s = edges_df.columns[cols.index("source")]
    t = edges_df.columns[cols.index("target")]

    sub = edges_df[(edges_df[s] == src) & (edges_df[t] == tgt)].copy()
    if sub.empty:
        return {"has_edge": False, "support": np.nan, "methods": ""}

    support = np.nan
    methods = ""
    if "support" in cols:
        support = float(sub[edges_df.columns[cols.index("support")]].iloc[0])
    if "methods" in cols:
        methods = str(sub[edges_df.columns[cols.index("methods")]].iloc[0])

    return {"has_edge": True, "support": support, "methods": methods}


def node_degree(metrics_df: pd.DataFrame, node="Capital_Formation_lag1") -> dict:
    """
    Lit dag_nodes_metrics.csv attendu: node,in_degree,out_degree,degree
    """
    if metrics_df.empty:
        return {"in_degree": np.nan, "out_degree": np.nan, "degree": np.nan}

    cols = [c.lower() for c in metrics_df.columns]
    if "node" not in cols:
        return {"in_degree": np.nan, "out_degree": np.nan, "degree": np.nan}

    ncol = metrics_df.columns[cols.index("node")]
    sub = metrics_df[metrics_df[ncol] == node]
    if sub.empty:
        return {"in_degree": 0, "out_degree": 0, "degree": 0}

    def get(colname, default=np.nan):
        if colname in cols:
            return float(sub[metrics_df.columns[cols.index(colname)]].iloc[0])
        return default

    return {"in_degree": get("in_degree", 0), "out_degree": get("out_degree", 0), "degree": get("degree", 0)}


# -------------------------
# Main
# -------------------------
def main():
    # Liste des pays (mets exactement les noms du dataset)
    countries = [
        "France",
        "Allemagne",
        "Congo, Dem. Rep.",
        "Nigeria",
        "Angola",
        "Maroc",
        "Ghana",
    ]

    results = []

    for country in countries:
        slug = slugify(country)

        # DAG
        dag_support_path = Path("../results/phase3E_dag") / slug / "dag_edges_support.csv"
        dag_final_path = Path("../results/phase3E_dag") / slug / "dag_edges_final.csv"
        dag_nodes_path = Path("../results/phase3E_dag") / slug / "dag_nodes_metrics.csv"

        dag_support = safe_read_csv(dag_support_path)
        dag_final = safe_read_csv(dag_final_path)
        dag_nodes = safe_read_csv(dag_nodes_path)

        # VAR IRF
        irf_path = Path("../results/phase3B_var") / slug / "var_irf_long.csv"
        irf_df = safe_read_csv(irf_path)

        # PCMCI
        pcmci_path = Path("../results/phase3D_pcmci") / slug / "pcmci_links.csv"
        pcmci_df = safe_read_csv(pcmci_path)

        # --- Metrics ---
        # Centralité Capital Formation (dans le DAG final si dispo, sinon support)
        deg = node_degree(dag_nodes, node="Capital_Formation_lag1")
        central = bucket_centrality(deg["degree"]) if not np.isnan(deg["degree"]) else "NA"

        # Lien direct Capital -> GDP (dans support si possible, sinon final)
        edge_info = has_direct_edge(dag_final if not dag_final.empty else dag_support,
                            src="Capital_Formation_lag1", tgt="GDP_Growth")

        # Support moyen (sur tout le graphe support)
        support_mean = np.nan
        support_max = np.nan
        if not dag_support.empty and "support" in [c.lower() for c in dag_support.columns]:
            supcol = dag_support.columns[[c.lower() for c in dag_support.columns].index("support")]
            support_mean = float(dag_support[supcol].mean())
            support_max = float(dag_support[supcol].max())

        # IRF summary
        irf_sum = summarize_irf_effect(irf_df, shock="Capital_Formation_lag1", response="GDP_Growth")

        # PCMCI: nombre de liens lag>=1
        pcmci_links_n = len(pcmci_df) if not pcmci_df.empty else 0

        # Structure globale
        n_edges = len(dag_final) if not dag_final.empty else (len(dag_support) if not dag_support.empty else 0)
        n_nodes = len(dag_nodes) if not dag_nodes.empty else 7
        structure = classify_structure(n_edges=n_edges, n_nodes=n_nodes)

        results.append({
            "Country": country,
            "Capital_Centrality": central,
            "Capital_in_degree": deg["in_degree"],
            "Capital_out_degree": deg["out_degree"],
            "Capital_degree": deg["degree"],
            "Direct_Capital_to_GDP": "Oui" if edge_info["has_edge"] else "Non",
            "Support(Capital->GDP)": edge_info["support"],
            "Methods(Capital->GDP)": edge_info["methods"],
            "Support_mean": support_mean,
            "Support_max": support_max,
            "VAR_IRF_max_abs": irf_sum["irf_max_abs"],
            "VAR_IRF_sign": irf_sum["irf_sign"],
            "VAR_IRF_best_horizon": irf_sum["irf_best_h"],
            "PCMCI_links_count": pcmci_links_n,
            "DAG_structure": structure,
        })

    out_df = pd.DataFrame(results)

    # Export
    reports_dir = Path("../reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_path = reports_dir / "phase3F_comparative_table.csv"
    md_path = reports_dir / "phase3F_comparative_table.md"

    out_df.to_csv(csv_path, index=False)

    # Markdown (pour mail / rapport rapide)
    md = []
    md.append("# Phase 3 — Tableau comparatif multi-pays (Granger + VAR + VECM + PCMCI + DAG)\n")
    md.append(out_df.to_markdown(index=False))
    md.append("\n\n**Lecture rapide :**")
    md.append("- `Direct_Capital_to_GDP`: le lien direct Capital_Formation → GDP_Growth apparaît (support multi-méthodes)")
    md.append("- `Support_mean/max`: cohérence globale entre méthodes")
    md.append("- `VAR_IRF_*`: réponse de GDP_Growth à un choc sur Capital_Formation")
    md.append("- `DAG_structure`: densité/complexité de la structure causale\n")

    md_path.write_text("\n".join(md), encoding="utf-8")

    print("=" * 95)
    print("PHASE 3F — TABLEAU COMPARATIF — OK")
    print(f"✅ CSV : {csv_path}")
    print(f"✅ MD  : {md_path}")
    print("=" * 95)


if __name__ == "__main__":
    main()