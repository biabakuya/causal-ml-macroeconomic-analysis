import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 80)
print("PHASE 4 — VISUALIZATION")
print("=" * 80)

# =========================================================
# PATHS
# =========================================================
DATA_PATH = Path("C:/Users/ibrahim/Pictures/stage_abil_2025/data/processed/data_transformed_unscaled.csv")
RES_DIR = Path("../results/phase4_dml")
FIG_DIR = Path("../reports/figures/phase4_dml")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(DATA_PATH)
countries = df["Country"].unique()

# =========================================================
# FIGURE 1 — GDP Growth (real values)
# =========================================================
plt.figure(figsize=(12, 6))

for c in countries:
    d = df[df["Country"] == c]
    plt.plot(d["Year"], d["GDP_Growth"], label=c)

plt.xlabel("Year")
plt.ylabel("Annual GDP Growth Rate (%)")
plt.title("Annual GDP Growth by Country (2000–2024)")
plt.legend(loc="upper right")
plt.grid(True)
plt.figtext(
    0.5, -0.03,
    "Source: World Bank indicators / processed dataset. Unit: annual growth rate in percent (%).",
    ha="center",
    fontsize=9
)
plt.tight_layout()

plt.savefig(FIG_DIR / "gdp_growth_real_values.png", dpi=300, bbox_inches="tight")
plt.show()

print("Figure saved : gdp_growth_real_values.png")


# =========================================================
# FIGURE 2 — Capital Formation (real values)
# =========================================================
plt.figure(figsize=(12, 6))

for c in countries:
    d = df[df["Country"] == c]
    plt.plot(d["Year"], d["Capital_Formation"], label=c)

plt.xlabel("Year")
plt.ylabel("Gross Capital Formation (% of GDP)")
plt.title("Gross Capital Formation by Country (2000–2024)")
plt.legend(loc="upper right")
plt.grid(True)
plt.figtext(
    0.5, -0.03,
    "Source: World Bank indicators / processed dataset. Unit: percent of GDP (%).",
    ha="center",
    fontsize=9
)
plt.tight_layout()

plt.savefig(FIG_DIR / "capital_formation_real_values.png", dpi=300, bbox_inches="tight")
plt.show()

print("Figure saved : capital_formation_real_values.png")


# =========================================================
# FIGURE 3 — ATE by country (LinearDML)
# =========================================================
df_ate_country = pd.read_csv(RES_DIR / "dml_hte_by_country.csv")
df_ate_country = df_ate_country.sort_values("ATE")

plt.figure(figsize=(12, 6))

plt.errorbar(
    x=df_ate_country["Country"],
    y=df_ate_country["ATE"],
    yerr=[
        df_ate_country["ATE"] - df_ate_country["ATE_LB_95"],
        df_ate_country["ATE_UB_95"] - df_ate_country["ATE"]
    ],
    fmt="o",
    capsize=5
)

plt.axhline(0, color="black", linewidth=1)
plt.xlabel("Country")
plt.ylabel("Effect on GDP Growth (percentage points)")
plt.title("Average Treatment Effect by Country (LinearDML)")
plt.xticks(rotation=45, ha="right")
plt.grid(True, axis="y")
plt.figtext(
    0.5, -0.03,
    "Effect estimated for Capital Formation(t-1) on GDP Growth(t). Bars indicate 95% confidence intervals.",
    ha="center",
    fontsize=9
)
plt.tight_layout()

plt.savefig(FIG_DIR / "dml_ate_by_country_visual.png", dpi=300, bbox_inches="tight")
plt.show()

print("Figure saved : dml_ate_by_country_visual.png")


# =========================================================
# FIGURE 4 — ATE by group (Developed vs Developing)
# =========================================================
df_ate_group = pd.read_csv(RES_DIR / "dml_ate_by_group.csv")
df_ate_group = df_ate_group.sort_values("ATE")

plt.figure(figsize=(8, 5))

plt.errorbar(
    x=df_ate_group["Group"],
    y=df_ate_group["ATE"],
    yerr=[
        df_ate_group["ATE"] - df_ate_group["ATE_LB_95"],
        df_ate_group["ATE_UB_95"] - df_ate_group["ATE"]
    ],
    fmt="o",
    capsize=6
)

plt.axhline(0, color="black", linewidth=1)
plt.xlabel("Country Group")
plt.ylabel("Effect on GDP Growth (percentage points)")
plt.title("Average Treatment Effect by Development Group (LinearDML)")
plt.grid(True, axis="y")
plt.figtext(
    0.5, -0.05,
    "Effect estimated for Capital Formation(t-1) on GDP Growth(t). Bars indicate 95% confidence intervals.",
    ha="center",
    fontsize=9
)
plt.tight_layout()

plt.savefig(FIG_DIR / "dml_ate_by_group_visual.png", dpi=300, bbox_inches="tight")
plt.show()

print("Figure saved : dml_ate_by_group_visual.png")


# =========================================================
# FIGURE 5 — Distribution of HTE (CausalForestDML)
# =========================================================
df_hte = pd.read_csv(RES_DIR / "dml_hte_individual_effects.csv")

plt.figure(figsize=(10, 6))

plt.hist(df_hte["HTE"], bins=30)
plt.axvline(0, color="black", linewidth=1)

plt.xlabel("Individual Treatment Effect on GDP Growth (percentage points)")
plt.ylabel("Frequency")
plt.title("Distribution of Heterogeneous Treatment Effects (CausalForestDML)")
plt.grid(True)
plt.figtext(
    0.5, -0.03,
    "Distribution of estimated individual effects of Capital Formation(t-1) on GDP Growth(t).",
    ha="center",
    fontsize=9
)
plt.tight_layout()

plt.savefig(FIG_DIR / "dml_hte_distribution_visual.png", dpi=300, bbox_inches="tight")
plt.show()

print("Figure saved : dml_hte_distribution_visual.png")


# =========================================================
# FIGURE 6 — Performance comparison of nuisance models
# =========================================================
print("Generating nuisance model performance visualization...")

df_features = pd.read_csv(RES_DIR / "dml_nuisance_model_comparison.csv")
df_features = df_features.sort_values("R2_Y_mean", ascending=True)

features = df_features["Model"]
importance = df_features["R2_Y_mean"]

# Couleurs selon performance :
# rouge = faible, jaune = moyen, vert = bon
norm = plt.Normalize(importance.min(), importance.max())
colors = plt.cm.RdYlGn(norm(importance))

plt.figure(figsize=(9, 6))
bars = plt.barh(features, importance, color=colors)

plt.xlabel("Cross-validated R² for outcome model")
plt.ylabel("Machine learning model")
plt.title("Performance Comparison of Nuisance Models")
plt.grid(True, axis="x")

# Afficher les valeurs au bout des barres
for bar, val in zip(bars, importance):
    plt.text(
        bar.get_width() + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}",
        va="center",
        fontsize=9
    )

plt.figtext(
    0.5, -0.03,
    "Lower performance is shown in red and higher performance in green, based on cross-validated R².",
    ha="center",
    fontsize=9
)

plt.tight_layout()
plt.savefig(FIG_DIR / "feature_importance_hte.png", dpi=300, bbox_inches="tight")
plt.show()

print("Figure saved : feature_importance_hte.png")