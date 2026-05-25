#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PHASE 3D — PCMCI 

Input :
  ../data/processed/pcmci_pays.csv

Outputs :
  ../results/phase3D_pcmci/pays/
      - pcmci_links.csv
      - pcmci_summary.txt
  ../reports/figures/phase3D_pcmci/pays/
      - pcmci_graph_tigramite.png
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from tigramite.data_processing import DataFrame as TigDataFrame
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.pcmci import PCMCI


def slugify(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\w\s\-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s)
    s = s.replace("__", "_")
    return s


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def build_links_table(var_names, p_matrix, val_matrix, alpha: float, tau_min: int, tau_max: int):
    """
    Convention Tigramite :
    p_matrix[j, i, tau] = pvalue pour i(t-tau) -> j(t)
    """
    rows = []
    n = len(var_names)

    for j in range(n):          # target
        for i in range(n):      # source
            for tau in range(tau_min, tau_max + 1):
                pval = p_matrix[j, i, tau]
                val = val_matrix[j, i, tau]

                if np.isnan(pval) or np.isnan(val):
                    continue

                if pval <= alpha:
                    source = var_names[i]
                    target = var_names[j]

                    # supprimer les auto-boucles
                    if source == target:
                        continue

                    rows.append({
                        "source": source,
                        "target": target,
                        "lag": int(tau),
                        "p_value": float(pval),
                        "val": float(val),
                        "sign": "positive" if val > 0 else "negative"
                    })

    if len(rows) == 0:
        return pd.DataFrame(columns=["source", "target", "lag", "p_value", "val", "sign"])

    return (
        pd.DataFrame(rows)
        .sort_values(["target", "p_value", "lag"], ascending=[True, True, True])
        .reset_index(drop=True)
    )


def pretty_label_t(name: str, time_suffix: str) -> str:
    base = "GDP Growth" if name == "GDP_Growth" else name.replace("_", " ")
    return f"{base} ({time_suffix})"


def node_color_time(node_name: str) -> str:
    if "GDP_Growth__t" in node_name:
        return "#f4d03f"   # jaune
    if "__t" in node_name:
        return "#85c1e9"   # bleu clair
    return "#5dade2"       # bleu pour les lags


def edge_color(sign: str) -> str:
    if sign == "positive":
        return "#1e8449"   # vert
    if sign == "negative":
        return "#c0392b"   # rouge
    return "#7f8c8d"


def build_temporal_graph(links_df: pd.DataFrame):
    """
    Crée un graphe temporel explicite :
    source(t-lag) -> target(t)
    """
    G = nx.DiGraph()

    if links_df.empty:
        return G

    for _, row in links_df.iterrows():
        src = row["source"]
        tgt = row["target"]
        lag = int(row["lag"])

        src_node = f"{src}__t-{lag}"
        tgt_node = f"{tgt}__t"

        G.add_node(src_node, base=src, time=f"t-{lag}")
        G.add_node(tgt_node, base=tgt, time="t")

        G.add_edge(
            src_node,
            tgt_node,
            lag=lag,
            p_value=float(row["p_value"]),
            val=float(row["val"]),
            sign=row["sign"]
        )

    return G


def build_temporal_positions(G: nx.DiGraph):
    """
    Place les nœuds par colonnes temporelles :
    t-3/t-2 à gauche, t-1 au milieu, t à droite
    avec plus d'espace vertical pour éviter les superpositions.
    """
    pos = {}

    # Détecter automatiquement les couches temporelles présentes
    time_layers = {}
    for node, data in G.nodes(data=True):
        time_key = data["time"]
        if time_key not in time_layers:
            time_layers[time_key] = []
        time_layers[time_key].append(node)

    # Ordre temporel
    def time_order_key(t):
        if t == "t":
            return 0
        m = re.match(r"t-(\d+)", t)
        if m:
            return -int(m.group(1))
        return -999

    ordered_times = sorted(time_layers.keys(), key=time_order_key)

    # Colonnes espacées
    if len(ordered_times) == 1:
        x_positions = {ordered_times[0]: 0.0}
    else:
        xs = np.linspace(0.0, 9.0, len(ordered_times))
        x_positions = {t: float(x) for t, x in zip(ordered_times, xs)}

    for time_key in ordered_times:
        nodes = sorted(time_layers[time_key], key=lambda n: G.nodes[n]["base"])
        n = len(nodes)

        if n == 0:
            continue

        y_vals = np.linspace((n - 1) * 2.2, 0, n) if n > 1 else [0.0]

        for i, node in enumerate(nodes):
            pos[node] = (x_positions[time_key], float(y_vals[i]))

    # Ajustements manuels pour éviter les superpositions à droite
    if "t" in x_positions:
        xt = x_positions["t"]
        for node, data in G.nodes(data=True):
            if data["base"] == "GDP_Growth" and data["time"] == "t":
                pos[node] = (xt, 2.8)
        for node, data in G.nodes(data=True):
            if data["base"] == "Trade_Balance" and data["time"] == "t":
                pos[node] = (xt, 0.2)
        for node, data in G.nodes(data=True):
            if data["base"] == "Exchange_Rate" and data["time"] == "t":
                pos[node] = (xt, 5.2)

    return pos


def draw_legend(ax):
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="#1e8449", lw=3, label="Positive"),
        Line2D([0], [0], color="#c0392b", lw=3, label="Negative"),
        Line2D([0], [0], marker='o', color='w', label='Lagged predictor',
               markerfacecolor="#5dade2", markeredgecolor="black", markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Macro variable (t)',
               markerfacecolor="#85c1e9", markeredgecolor="black", markersize=10),
        Line2D([0], [0], marker='o', color='w', label='GDP Growth (t)',
               markerfacecolor="#f4d03f", markeredgecolor="black", markersize=10),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        frameon=False,
        fontsize=9
    )


def plot_pcmci_graph_temporal(links_df: pd.DataFrame, country: str, fig_path: Path):
    if links_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(
            0.5, 0.5,
            f"No significant PCMCI links for {country}",
            ha="center", va="center", fontsize=16, fontweight="bold"
        )
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        return

    G = build_temporal_graph(links_df)
    pos = build_temporal_positions(G)

    fig, ax = plt.subplots(figsize=(20, 11))
    ax.set_facecolor("white")

    # Nœuds
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=[node_color_time(n) for n in G.nodes()],
        node_size=3200,
        edgecolors="black",
        linewidths=1.4,
        ax=ax
    )

    # Labels nœuds
    labels = {}
    for n, data in G.nodes(data=True):
        labels[n] = pretty_label_t(data["base"], data["time"])

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=10,
        font_weight="bold",
        ax=ax
    )

    # Arêtes
    edges = list(G.edges(data=True))
    edge_colors = [edge_color(d["sign"]) for _, _, d in edges]
    edge_widths = [3.2 if d["p_value"] < 0.01 else 2.4 for _, _, d in edges]

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors,
        width=edge_widths,
        arrows=True,
        arrowsize=26,
        arrowstyle="-|>",
        min_source_margin=20,
        min_target_margin=30,
        connectionstyle="arc3,rad=0.06",
        ax=ax
    )

    # Labels arêtes
    edge_labels = {}
    for u, v, d in edges:
        sign_txt = "+" if d["sign"] == "positive" else "-"
        edge_labels[(u, v)] = f"{sign_txt}, p={d['p_value']:.3g}"

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=8,
        rotate=False,
        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85},
        label_pos=0.42,
        ax=ax
    )

    # Titres des colonnes temporelles
    if pos:
        max_y = max(y for _, y in pos.values())
        # Colonnes détectées
        x_to_label = {}
        for node, data in G.nodes(data=True):
            x = pos[node][0]
            x_to_label[x] = data["time"]
        for x in sorted(x_to_label.keys()):
            label = "Current time t" if x_to_label[x] == "t" else f"Lag {x_to_label[x]}"
            ax.text(x, max_y + 1.3, label, fontsize=12, fontweight="bold", ha="center")

    ax.set_title(
        f"PCMCI Temporal Causal Graph — {country}\nSignificant lagged causal links",
        fontsize=16,
        fontweight="bold",
        pad=32
    )

    draw_legend(ax)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", type=str, required=True, help='Ex: "France", "Congo, Dem. Rep."')
    parser.add_argument("--tau_max", type=int, default=3, help="Maximum lag for PCMCI")
    parser.add_argument("--pc_alpha", type=float, default=0.05, help="PC1 alpha")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold for links")
    args = parser.parse_args()

    country = args.country
    tau_max = int(args.tau_max)
    pc_alpha = float(args.pc_alpha)
    alpha = float(args.alpha)

    print("=" * 100)
    print("PHASE 3D — PCMCI")
    print("=" * 100)
    print(f"Country : {country}")

    data_processed = Path("../data/processed")
    country_slug = slugify(country)

    input_csv = data_processed / f"pcmci_{country_slug}.csv"
    if not input_csv.exists():
        input_csv = data_processed / f"pcmci_{country}.csv"

    if not input_csv.exists():
        raise FileNotFoundError(f"PCMCI input not found. Expected: {data_processed / f'pcmci_{country_slug}.csv'}")

    out_res_dir = Path("../results/phase3D_pcmci") / country_slug
    out_fig_dir = Path("../reports/figures/phase3D_pcmci") / country_slug
    ensure_dir(out_res_dir)
    ensure_dir(out_fig_dir)

    # Load
    df = pd.read_csv(input_csv)

    if "Year" in df.columns:
        df = df.drop(columns=["Year"])
    if "Country" in df.columns:
        df = df.drop(columns=["Country"])

    var_names = list(df.columns)
    X = df[var_names].astype(float).values

    print(f"Observations : {X.shape[0]}")
    print(f"Variables : {var_names}")

    # Tigramite dataframe
    tig_df = TigDataFrame(data=X, var_names=var_names)

    # Test d'indépendance plus robuste que ParCorr pour structures non linéaires
    cmiknn = CMIknn(
        significance="shuffle_test",
        knn=5
    )

    pcmci = PCMCI(dataframe=tig_df, cond_ind_test=cmiknn, verbosity=1)

    results = pcmci.run_pcmci(
        tau_min=1,
        tau_max=tau_max,
        pc_alpha=pc_alpha,
        fdr_method="fdr_bh"
    )

    if "p_matrix" not in results or "val_matrix" not in results:
        raise KeyError("PCMCI results missing p_matrix/val_matrix.")

    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]

    links_df = build_links_table(
        var_names=var_names,
        p_matrix=p_matrix,
        val_matrix=val_matrix,
        alpha=alpha,
        tau_min=1,
        tau_max=tau_max
    )

    links_path = out_res_dir / "pcmci_links.csv"
    links_df.to_csv(links_path, index=False)

    summary_path = out_res_dir / "pcmci_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("PHASE 3D — PCMCI SUMMARY\n")
        f.write(f"Country: {country}\n")
        f.write(f"Input: {input_csv}\n")
        f.write(f"Observations: {X.shape[0]}\n")
        f.write(f"Variables: {var_names}\n")
        f.write(f"tau_min: 1\n")
        f.write(f"tau_max: {tau_max}\n")
        f.write(f"pc_alpha: {pc_alpha}\n")
        f.write(f"alpha (links): {alpha}\n")
        f.write(f"independence_test: CMIknn (shuffle_test, knn=5)\n")
        f.write(f"fdr_method: fdr_bh\n")
        f.write(f"Significant links (lag>=1, without self-loops): {len(links_df)}\n\n")

        if len(links_df) == 0:
            f.write("No significant lagged links found at this alpha.\n")
        else:
            f.write(links_df.to_string(index=False))
            f.write("\n")

    print(f"✅ Significant links (lag>=1, without self-loops): {len(links_df)}")
    print(f"✅ Saved CSV/TXT to: {out_res_dir}")

    fig_path = out_fig_dir / "pcmci_graph_tigramite.png"
    plot_pcmci_graph_temporal(links_df, country, fig_path)

    print(f"✅ Figure saved: {fig_path}")
    print("=" * 100)
    print("✅ PHASE 3D — PCMCI — TERMINÉ")
    print("=" * 100)


if __name__ == "__main__":
    main()