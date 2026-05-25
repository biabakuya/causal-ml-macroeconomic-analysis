#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from phase4_00_dml_config import RES_DIR, FIG_DIR, Y_COL, T_COL, BASE_X_COLS

print("="*100)
print("PHASE 4 — DML — EXPORT REPORT")
print("="*100)

ate_path = RES_DIR / "dml_ate_results.csv"
by_country_path = RES_DIR / "dml_hte_by_country.csv"

if not ate_path.exists():
    raise FileNotFoundError(f"ATE global introuvable: {ate_path}")

ate = pd.read_csv(ate_path).iloc[0].to_dict()

md_path = RES_DIR / "dml_summary_phase4.md"

lines = []
lines.append("# Phase 4 — Double Machine Learning (DML)\n")
lines.append(f"**Objectif causal :** effet de **{T_COL} (T)** sur **{Y_COL} (Y)** en contrôlant les confondeurs **X**.\n")
lines.append("## Variables\n")
lines.append(f"- **Y (outcome)** : `{Y_COL}`\n")
lines.append(f"- **T (treatment)** : `{T_COL}`\n")
lines.append(f"- **X (confounders)** : `{', '.join(BASE_X_COLS)}` (+ dummies pays dans l’ATE global)\n")

lines.append("\n## Résultat principal (ATE global)\n")
lines.append(f"- **ATE** : {ate['ATE']}\n")
lines.append(f"- **IC 95%** : [{ate['ATE_LB_95']}, {ate['ATE_UB_95']}]\n")
lines.append(f"- **StdErr** : {ate['StdErr']}\n")
lines.append(f"- **p-value** : {ate['p_value']}\n")

# If per-country exists, add + plot
if by_country_path.exists():
    dfc = pd.read_csv(by_country_path)

    lines.append("\n## Hétérogénéité simple (ATE par pays)\n")
    lines.append("Table enregistrée : `dml_hte_by_country.csv`\n")

    # Plot: ATE per country with CI
    dfc = dfc.sort_values("ATE")
    fig = plt.figure(figsize=(12, 6))
    plt.errorbar(
        x=dfc["Country"],
        y=dfc["ATE"],
        yerr=[dfc["ATE"] - dfc["ATE_LB_95"], dfc["ATE_UB_95"] - dfc["ATE"]],
        fmt="o",
        capsize=4
    )
    plt.axhline(0, linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"ATE par pays: {T_COL} → {Y_COL} (IC 95%)")
    plt.tight_layout()

    fig_path = FIG_DIR / "dml_ate_by_country.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    lines.append(f"Figure : `../reports/figures/phase4_dml/{fig_path.name}`\n")
else:
    lines.append("\n## Hétérogénéité simple (ATE par pays)\n")
    lines.append("Non exécuté (fichier `dml_ate_by_country.csv` non trouvé).\n")

md_path.write_text("\n".join(lines), encoding="utf-8")

print("✅ Saved:", md_path)
if (FIG_DIR / "dml_ate_by_country.png").exists():
    print("✅ Figure:", FIG_DIR / "dml_ate_by_country.png")

print("✅ EXPORT TERMINE")