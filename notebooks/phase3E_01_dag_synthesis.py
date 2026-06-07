#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PHASE 3E — DAG Synthesis (PCMCI + Granger + VAR-IRF + VAR-FEVD)
"""

import argparse
from pathlib import Path
import re
import shutil
import os
import numpy as np
import pandas as pd
import networkx as nx
from graphviz import Digraph, Source

os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"


def slugify(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\w\s\-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s)
    s = s.replace("__", "_")
    return s


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def find_first_existing(paths):
    for p in paths:
        if p is not None and Path(p).exists():
            return Path(p)
    return None


def dot_available() -> bool:
    return shutil.which("dot") is not None


def sign_of(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if x > 0:
        return "positive"
    if x < 0:
        return "negative"
    return "neutral"


def split_var_time(name: str):
    name = str(name).strip()
    m = re.match(r"^(.*)_lag([0-9]+)$", name)
    if m:
        base = m.group(1)
        lag = int(m.group(2))
        return base, f"t-{lag}"
    return name, "t"


def temporal_node_name(base: str, time_str: str) -> str:
    return f"{base}__{time_str}"


def temporal_label(node_name: str) -> str:
    if "__" in node_name:
        base, time_str = node_name.split("__", 1)
    else:
        base, time_str = node_name, "t"

    base_label = "GDP Growth" if base == "GDP_Growth" else base.replace("_", " ")
    return f"{base_label} ({time_str})"


def node_time(node_name: str) -> str:
    if "__" not in node_name:
        return "t"
    return node_name.split("__", 1)[1]


def is_strict_temporal_edge(source: str, target: str) -> bool:
    if "__" not in source or "__" not in target:
        return False

    s_time = node_time(source)
    t_time = node_time(target)

    return (s_time in {"t-1", "t-2", "t-3"}) and (t_time == "t")


def edge_color(sign: str) -> str:
    if sign == "positive":
        return "#16a34a"
    if sign == "negative":
        return "#dc2626"
    if sign == "mixed":
        return "#d97706"
    return "#9ca3af"


def node_style(node_name: str, degree: int, central_node: str = None):
    base, time_str = node_name.split("__", 1)

    if base == "GDP_Growth" and time_str == "t":
        fill = "#facc15"
        font = "black"
    elif node_name == central_node:
        fill = "#1d4ed8"
        font = "white"
    elif time_str == "t":
        fill = "#93c5fd"
        font = "black"
    else:
        fill = "#bfdbfe"
        font = "black"

    width = max(1.4, 1.1 + 0.16 * degree)
    height = max(0.75, 0.62 + 0.06 * degree)
    return fill, font, width, height


def break_cycles(edges_df: pd.DataFrame):
    if edges_df.empty:
        return edges_df, []

    edges_remaining = edges_df.copy()
    removed_edges = []
    iteration = 0
    max_iterations = 100

    while iteration < max_iterations:
        iteration += 1

        G = nx.DiGraph()
        edge_data = {}

        for idx, r in edges_remaining.iterrows():
            src = r["source"]
            tgt = r["target"]
            G.add_edge(src, tgt)
            edge_data[(src, tgt)] = {
                "support": int(r.get("support", 1)),
                "weight_sum": float(r.get("weight_sum", 0.0)),
                "idx": idx
            }

        try:
            cycles = list(nx.simple_cycles(G))
        except Exception:
            cycles = []

        if not cycles:
            break

        cycle = cycles[0]
        cycle_edges = []
        for i in range(len(cycle)):
            src = cycle[i]
            tgt = cycle[(i + 1) % len(cycle)]
            if (src, tgt) in edge_data:
                cycle_edges.append((src, tgt, edge_data[(src, tgt)]))

        weakest = min(
            cycle_edges,
            key=lambda x: (x[2]["support"], x[2]["weight_sum"], x[0], x[1])
        )

        src_weak, tgt_weak, data_weak = weakest
        idx_to_remove = data_weak["idx"]

        removed_edges.append({
            "source": src_weak,
            "target": tgt_weak,
            "support": data_weak["support"],
            "weight_sum": data_weak["weight_sum"],
            "reason": f"Cycle resolution iteration {iteration}",
            "cycle": " -> ".join(cycle + [cycle[0]])
        })

        edges_remaining = edges_remaining.drop(idx_to_remove)

    return edges_remaining.reset_index(drop=True), removed_edges


def load_pcmci_edges(pcmci_path: Path, alpha=None):
    df = pd.read_csv(pcmci_path)
    cols = [c.lower() for c in df.columns]

    if "source" not in cols or "target" not in cols:
        raise ValueError(f"PCMCI file columns not recognized: {df.columns.tolist()}")

    src_col = df.columns[cols.index("source")]
    tgt_col = df.columns[cols.index("target")]

    lag_col = None
    for cand in ["lag", "tau"]:
        if cand in cols:
            lag_col = df.columns[cols.index(cand)]
            break

    p_col = None
    for cand in ["p_value", "pval", "p"]:
        if cand in cols:
            p_col = df.columns[cols.index(cand)]
            break

    strength_col = None
    for cand in ["strength", "val", "value", "partial_corr", "corr"]:
        if cand in cols:
            strength_col = df.columns[cols.index(cand)]
            break

    out_rows = []
    for _, r in df.iterrows():
        lag = int(r[lag_col]) if lag_col is not None and not pd.isna(r[lag_col]) else None
        if lag is None or lag < 1:
            continue

        pval = float(r[p_col]) if p_col is not None and not pd.isna(r[p_col]) else None
        if alpha is not None and pval is not None and pval > alpha:
            continue

        strength = float(r[strength_col]) if strength_col is not None and not pd.isna(r[strength_col]) else None

        source_base = str(r[src_col])
        target_base = str(r[tgt_col])

        source_node = temporal_node_name(source_base, f"t-{lag}")
        target_node = temporal_node_name(target_base, "t")

        out_rows.append({
            "source": source_node,
            "target": target_node,
            "method": "pcmci",
            "weight": abs(strength) if strength is not None else np.nan,
            "sign": sign_of(strength),
        })

    return pd.DataFrame(out_rows)


def load_granger_edges(granger_path: Path, alpha=None):
    df = pd.read_csv(granger_path)
    cols = [c.lower() for c in df.columns]

    if "cause" not in cols or "effect" not in cols:
        raise ValueError(f"Granger file columns not recognized: {df.columns.tolist()}")

    cause_col = df.columns[cols.index("cause")]
    effect_col = df.columns[cols.index("effect")]
    p_col = df.columns[cols.index("p_value_min")] if "p_value_min" in cols else None

    out_rows = []
    for _, r in df.iterrows():
        pval = float(r[p_col]) if p_col is not None and not pd.isna(r[p_col]) else None
        if alpha is not None and pval is not None and pval > alpha:
            continue

        src_base, src_time = split_var_time(str(r[cause_col]))
        tgt_base, tgt_time = split_var_time(str(r[effect_col]))

        out_rows.append({
            "source": temporal_node_name(src_base, src_time),
            "target": temporal_node_name(tgt_base, tgt_time),
            "method": "granger",
            "weight": 1.0 if pval is None else max(1e-12, (alpha if alpha else 0.05) / pval),
            "sign": None,
        })

    return pd.DataFrame(out_rows)


def load_var_irf_edges(irf_path: Path, irf_min_abs=0.05, horizon_min=1, top_k=None):
    df = pd.read_csv(irf_path)
    cols = [c.lower() for c in df.columns]
    required = {"horizon", "response", "shock", "irf"}
    if not required.issubset(set(cols)):
        raise ValueError(f"IRF columns not recognized: {df.columns.tolist()}")

    horizon_col = df.columns[cols.index("horizon")]
    resp_col = df.columns[cols.index("response")]
    shock_col = df.columns[cols.index("shock")]
    irf_col = df.columns[cols.index("irf")]

    df2 = df[df[horizon_col] >= horizon_min].copy()
    if df2.empty:
        return pd.DataFrame(columns=["source", "target", "method", "weight", "sign"])

    grouped = []
    for (resp, shock), g in df2.groupby([resp_col, shock_col]):
        arr = g[irf_col].astype(float).values
        h_arr = g[horizon_col].astype(int).values
        idx = int(np.argmax(np.abs(arr)))
        max_abs = float(np.abs(arr[idx]))
        signed_val = float(arr[idx])
        best_h = int(h_arr[idx])
        grouped.append((shock, resp, max_abs, signed_val, best_h))

    gdf = pd.DataFrame(grouped, columns=["shock", "response", "max_abs", "signed_val", "best_horizon"])
    gdf = gdf[gdf["max_abs"] >= float(irf_min_abs)].copy()

    if gdf.empty:
        return pd.DataFrame(columns=["source", "target", "method", "weight", "sign"])

    if top_k is not None and int(top_k) > 0:
        gdf = (
            gdf.sort_values(["response", "max_abs"], ascending=[True, False])
            .groupby("response", as_index=False)
            .head(int(top_k))
        )

    out_rows = []
    for _, r in gdf.iterrows():
        src_base, src_time = split_var_time(str(r["shock"]))
        tgt_base, tgt_time = split_var_time(str(r["response"]))

        out_rows.append({
            "source": temporal_node_name(src_base, src_time),
            "target": temporal_node_name(tgt_base, tgt_time),
            "method": "var_irf",
            "weight": float(r["max_abs"]),
            "sign": sign_of(float(r["signed_val"])),
        })

    return pd.DataFrame(out_rows)


def load_var_fevd_edges(fevd_path: Path, fevd_min_share=0.10, use_max_horizon=True):
    df = pd.read_csv(fevd_path)
    cols = [c.lower() for c in df.columns]
    required = {"variable", "horizon", "shock", "fevd"}
    if not required.issubset(set(cols)):
        raise ValueError(f"FEVD columns not recognized: {df.columns.tolist()}")

    var_col = df.columns[cols.index("variable")]
    horizon_col = df.columns[cols.index("horizon")]
    shock_col = df.columns[cols.index("shock")]
    fevd_col = df.columns[cols.index("fevd")]

    df2 = df.copy()
    if use_max_horizon and not df2.empty:
        hmax = int(df2[horizon_col].max())
        df2 = df2[df2[horizon_col] == hmax].copy()

    df2[fevd_col] = df2[fevd_col].astype(float)
    df2 = df2[df2[fevd_col] >= float(fevd_min_share)].copy()
    if df2.empty:
        return pd.DataFrame(columns=["source", "target", "method", "weight", "sign"])

    out_rows = []
    for _, r in df2.iterrows():
        src_base, src_time = split_var_time(str(r[shock_col]))
        tgt_base, tgt_time = split_var_time(str(r[var_col]))

        out_rows.append({
            "source": temporal_node_name(src_base, src_time),
            "target": temporal_node_name(tgt_base, tgt_time),
            "method": "var_fevd",
            "weight": float(r[fevd_col]),
            "sign": None,
        })

    return pd.DataFrame(out_rows)


def synthesize_edges(method_dfs):
    edges = pd.concat(method_dfs, ignore_index=True) if len(method_dfs) else pd.DataFrame()

    if edges.empty:
        return pd.DataFrame(
            columns=["source", "target", "support", "methods", "sign_majority", "weight_sum"]
        ), []

    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)
    edges["method"] = edges["method"].astype(str)

    filtered_edges = []
    valid_edges = []

    for _, row in edges.iterrows():
        source = row["source"]
        target = row["target"]

        if source == target:
            filtered_edges.append({
                "source": source,
                "target": target,
                "method": row["method"],
                "reason": "Self-loop removed"
            })
            continue

        if not is_strict_temporal_edge(source, target):
            filtered_edges.append({
                "source": source,
                "target": target,
                "method": row["method"],
                "reason": "Filtered: only past -> present edges allowed"
            })
            continue

        valid_edges.append(row)

    edges = pd.DataFrame(valid_edges) if valid_edges else pd.DataFrame()

    if edges.empty:
        return pd.DataFrame(
            columns=["source", "target", "support", "methods", "sign_majority", "weight_sum"]
        ), filtered_edges

    agg = (
        edges.groupby(["source", "target"])
        .agg(
            support=("method", lambda x: len(set(x))),
            methods=("method", lambda x: ",".join(sorted(set(x)))),
            weight_sum=("weight", lambda x: float(np.nansum(pd.to_numeric(x, errors="coerce").fillna(0.0)))),
        )
        .reset_index()
    )

    sign_map = (
        edges.dropna(subset=["sign"])
        .groupby(["source", "target"])["sign"]
        .apply(list)
        .reset_index(name="signs")
    )

    agg = agg.merge(sign_map, on=["source", "target"], how="left")

    def majority_sign(signs):
        if not isinstance(signs, list) or len(signs) == 0:
            return "unknown"
        pos = sum(s == "positive" for s in signs)
        neg = sum(s == "negative" for s in signs)
        neu = sum(s == "neutral" for s in signs)

        if pos > neg and pos >= neu:
            return "positive"
        if neg > pos and neg >= neu:
            return "negative"
        if neu > pos and neu > neg:
            return "neutral"
        return "mixed"

    agg["sign_majority"] = agg["signs"].apply(majority_sign)
    agg = agg.drop(columns=["signs"], errors="ignore")

    agg = agg.sort_values(
        ["support", "weight_sum", "source", "target"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    return agg, filtered_edges


def compute_node_metrics(edges_df: pd.DataFrame):
    if edges_df.empty:
        return pd.DataFrame(columns=["node", "label", "in_degree", "out_degree", "degree"])

    G = nx.DiGraph()
    for _, r in edges_df.iterrows():
        G.add_edge(r["source"], r["target"])

    rows = []
    for n in sorted(G.nodes()):
        in_d = int(G.in_degree(n))
        out_d = int(G.out_degree(n))
        rows.append({
            "node": n,
            "label": temporal_label(n),
            "in_degree": in_d,
            "out_degree": out_d,
            "degree": in_d + out_d
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["degree", "out_degree", "in_degree", "node"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return df


def graphviz_dag(
    edges_final: pd.DataFrame,
    country: str,
    out_base: Path,
    min_support: int,
    rankdir: str = "LR",
    engine: str = "dot",
):
    title = f"Final DAG — {country} (support ≥ {min_support}, acyclic)"
    dot = Digraph(name="DAG", format="png", engine=engine)

    dot.attr(
        rankdir=rankdir,
        labelloc="t",
        fontsize="22",
        label=title,
        pad="0.25",
        bgcolor="white"
    )

    dot.attr(
        "graph",
        splines="polyline",
        overlap="false",
        nodesep="0.7",
        ranksep="1.0",
        concentrate="false"
    )
    dot.attr(
        "node",
        shape="ellipse",
        style="filled",
        fontname="Helvetica",
        fontsize="12",
        color="#1f2937"
    )
    dot.attr(
        "edge",
        fontname="Helvetica",
        fontsize="9",
        arrowsize="0.8"
    )

    if edges_final.empty:
        dot.node("EMPTY", "No retained link", fillcolor="#f3f4f6", color="#9ca3af")
    else:
        G = nx.DiGraph()
        for _, r in edges_final.iterrows():
            G.add_edge(r["source"], r["target"])

        degrees = dict(G.degree())
        central_node = max(degrees, key=lambda k: degrees[k]) if degrees else None

        nodes_t3 = sorted([n for n in G.nodes if node_time(n) == "t-3"])
        nodes_t2 = sorted([n for n in G.nodes if node_time(n) == "t-2"])
        nodes_t1 = sorted([n for n in G.nodes if node_time(n) == "t-1"])
        nodes_t = sorted([n for n in G.nodes if node_time(n) == "t"])

        def add_nodes_to_subgraph(subg, nodes):
            for n in nodes:
                deg = degrees.get(n, 0)
                fill, font, width, height = node_style(n, deg, central_node)
                subg.node(
                    n,
                    temporal_label(n),
                    fillcolor=fill,
                    fontcolor=font,
                    width=str(width),
                    height=str(height),
                    fixedsize="true",
                )

        if nodes_t3:
            with dot.subgraph(name="cluster_t3") as c:
                c.attr(label="Lag t-3", color="#d1d5db", fontsize="11")
                c.attr(rank="same")
                add_nodes_to_subgraph(c, nodes_t3)

        if nodes_t2:
            with dot.subgraph(name="cluster_t2") as c:
                c.attr(label="Lag t-2", color="#d1d5db", fontsize="11")
                c.attr(rank="same")
                add_nodes_to_subgraph(c, nodes_t2)

        if nodes_t1:
            with dot.subgraph(name="cluster_t1") as c:
                c.attr(label="Lag t-1", color="#d1d5db", fontsize="11")
                c.attr(rank="same")
                add_nodes_to_subgraph(c, nodes_t1)

        if nodes_t:
            with dot.subgraph(name="cluster_t") as c:
                c.attr(label="Current time t", color="#d1d5db", fontsize="11")
                c.attr(rank="same")
                add_nodes_to_subgraph(c, nodes_t)

        for _, r in edges_final.iterrows():
            u = r["source"]
            v = r["target"]
            s = int(r.get("support", 1))
            sign = str(r.get("sign_majority", "unknown"))

            color = edge_color(sign)
            penwidth = str(1.4 + 0.7 * max(0, s - 1))
            label = f"s={s}"

            dot.edge(
                u,
                v,
                color=color,
                penwidth=penwidth,
                label=label,
                minlen="2"
            )

        with dot.subgraph(name="cluster_legend") as c:
            c.attr(label="Legend", color="#d1d5db", fontsize="11")
            c.node("Lpos", "Positive Link", shape="box", style="filled", fillcolor="white", color="#16a34a")
            c.node("Lneg", "Negative Link", shape="box", style="filled", fillcolor="white", color="#dc2626")
            c.node("Lmix", "Mixed Link", shape="box", style="filled", fillcolor="white", color="#d97706")
            c.node("Lunk", "Unknown Sign", shape="box", style="filled", fillcolor="white", color="#9ca3af")
            c.edge("Lpos", "Lneg", style="invis")
            c.edge("Lneg", "Lmix", style="invis")
            c.edge("Lmix", "Lunk", style="invis")

    dot_path = str(out_base) + ".dot"
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write(dot.source)

    png_path = None
    pdf_path = None

    if dot_available():
        png_src = Source(dot.source, filename=str(out_base), format="png", engine=engine)
        png_path = png_src.render(cleanup=True)

        pdf_src = Source(dot.source, filename=str(out_base), format="pdf", engine=engine)
        pdf_path = pdf_src.render(cleanup=True)

    return dot_path, png_path, pdf_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", type=str, required=True, help='Ex: "France"')
    parser.add_argument("--min_support", type=int, default=1, help="Min number of methods supporting an edge")
    parser.add_argument("--alpha_granger", type=float, default=0.05, help="p-value threshold for granger")
    parser.add_argument("--alpha_pcmci", type=float, default=None, help="p-value threshold for pcmci")
    parser.add_argument("--irf_min_abs", type=float, default=0.05, help="Min max|IRF| to keep edge")
    parser.add_argument("--irf_top_k", type=int, default=2, help="Keep top K shocks per response")
    parser.add_argument("--fevd_min_share", type=float, default=0.10, help="Min FEVD share")
    parser.add_argument("--rankdir", type=str, default="LR", choices=["LR", "TB", "RL", "BT"])
    parser.add_argument("--engine", type=str, default="dot", choices=["dot", "neato", "fdp", "sfdp", "twopi", "circo"])
    args = parser.parse_args()

    country = args.country
    slug = slugify(country)

    pcmci_candidates = [
        Path("../results/phase3D_pcmci") / slug / "pcmci_links.csv",
        Path("../results/phase3D_pcmci") / country / "pcmci_links.csv",
    ]
    granger_candidates = [
        Path("../results/phase3A_granger") / slug / "granger_edges_significant.csv",
        Path("../results/phase3A_granger") / country / "granger_edges_significant.csv",
    ]
    irf_candidates = [
        Path("../results/phase3B_var") / slug / "var_irf_long.csv",
        Path("../results/phase3B_var") / country / "var_irf_long.csv",
    ]
    fevd_candidates = [
        Path("../results/phase3B_var") / slug / "var_fevd_long.csv",
        Path("../results/phase3B_var") / country / "var_fevd_long.csv",
    ]

    pcmci_path = find_first_existing(pcmci_candidates)
    granger_path = find_first_existing(granger_candidates)
    irf_path = find_first_existing(irf_candidates)
    fevd_path = find_first_existing(fevd_candidates)

    out_res_dir = Path("../results/phase3E_dag") / slug
    out_fig_dir = Path("../reports/figures/phase3E_dag") / slug
    ensure_dir(out_res_dir)
    ensure_dir(out_fig_dir)

    print("=" * 100)
    print("PHASE 3E — DAG FINAL")
    print("=" * 100)
    print(f"Country     : {country}")
    print(f"Slug        : {slug}")
    print(f"min_support : {args.min_support}")

    method_dfs = []

    if pcmci_path is not None:
        method_dfs.append(load_pcmci_edges(pcmci_path, alpha=args.alpha_pcmci))
    if granger_path is not None:
        method_dfs.append(load_granger_edges(granger_path, alpha=args.alpha_granger))

    top_k = None if args.irf_top_k is None or int(args.irf_top_k) <= 0 else int(args.irf_top_k)
    if irf_path is not None:
        method_dfs.append(load_var_irf_edges(irf_path, irf_min_abs=args.irf_min_abs, horizon_min=1, top_k=top_k))
    if fevd_path is not None:
        method_dfs.append(load_var_fevd_edges(fevd_path, fevd_min_share=args.fevd_min_share, use_max_horizon=True))

    edges_support, filtered_edges = synthesize_edges(method_dfs)

    support_path = out_res_dir / "dag_edges_support.csv"
    edges_support.to_csv(support_path, index=False)

    if filtered_edges:
        filtered_path = out_res_dir / "dag_edges_filtered.txt"
        with open(filtered_path, "w", encoding="utf-8") as f:
            f.write("FILTERED EDGES\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total filtered : {len(filtered_edges)}\n\n")
            for fe in filtered_edges:
                f.write(f"{fe['source']} -> {fe['target']} ({fe['method']})\n")
                f.write(f"  Reason: {fe['reason']}\n\n")

    edges_final = edges_support[edges_support["support"] >= int(args.min_support)].copy()
    edges_final = edges_final[edges_final["source"] != edges_final["target"]].copy()

    edges_final, removed_cycles = break_cycles(edges_final)

    if removed_cycles:
        cycles_path = out_res_dir / "dag_edges_cycles_removed.txt"
        with open(cycles_path, "w", encoding="utf-8") as f:
            f.write("REMOVED EDGES FOR CYCLE RESOLUTION\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total removed : {len(removed_cycles)}\n\n")
            for rc in removed_cycles:
                f.write(
                    f"{rc['source']} -> {rc['target']} "
                    f"(support={rc['support']}, weight={rc['weight_sum']:.4f})\n"
                )
                f.write(f"  Cycle: {rc['cycle']}\n\n")

    final_path = out_res_dir / "dag_edges_final.csv"
    edges_final.to_csv(final_path, index=False)

    node_metrics = compute_node_metrics(edges_final)
    nodes_path = out_res_dir / "dag_nodes_metrics.csv"
    node_metrics.to_csv(nodes_path, index=False)

    summary_path = out_res_dir / "dag_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("PHASE 3E — DAG SUMMARY\n")
        f.write(f"Country: {country}\n")
        f.write(f"min_support: {args.min_support}\n")
        f.write(f"Filtered edges: {len(filtered_edges)}\n")
        f.write(f"Removed cycles: {len(removed_cycles)}\n")
        f.write(f"Edges support table: {len(edges_support)}\n")
        f.write(f"Edges final DAG: {len(edges_final)}\n\n")

        if not edges_final.empty:
            f.write("=== FINAL EDGES ===\n")
            f.write(edges_final.to_string(index=False))
            f.write("\n\n")

        if not node_metrics.empty:
            f.write("=== NODE METRICS ===\n")
            f.write(node_metrics.to_string(index=False))

    out_base = out_fig_dir / f"dag_final_{slug}"
    dot_path, png_path, pdf_path = graphviz_dag(
        edges_final=edges_final,
        country=country,
        out_base=out_base,
        min_support=int(args.min_support),
        rankdir=args.rankdir,
        engine=args.engine,
    )

    print(f"\n✅ Saved support edges : {support_path}")
    print(f"✅ Saved final edges   : {final_path}")
    print(f"✅ Saved node metrics  : {nodes_path}")
    print(f"✅ Saved summary       : {summary_path}")
    print(f"✅ Saved figure base   : {out_base}")
    print("=" * 100)
    print("✅ PHASE 3E — DAG FINAL — TERMINÉ")
    print("=" * 100)


if __name__ == "__main__":
    main()