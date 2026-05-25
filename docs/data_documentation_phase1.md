# Data Documentation — Phase 1 (Preparation)

**Stage ABIL 2025 — Jirince BIABA KUYA**

## Input / Output
- Input: `..\data\processed\dataset_harmonised_final.csv`
- Output (transformed, unscaled): `..\data\processed\data_transformed_unscaled.csv`
- Output (transformed + scaled): `..\data\processed\data_prepared_for_dml.csv`

## Panel
- Countries: 7
- Period: 2000–2024
- Rows: 174

## Transformations (macro-standard)
- GDP_Growth: level
- Inflation: log(1 + x)
- Capital_Formation: log(x)
- Government_Debt: log(x)
- Exchange_Rate: log(x)
- Reserves: log(x)
- Trade_Balance: level

## Cleaning
- Drop rows with NaN after logs: 1 row(s) removed

## Winsorization
- Mode: by_country
- Quantiles: 0.01 – 0.99
- Applied **before** standardization

## Standardization
- Z-score on numeric variables (μ=0, σ=1) with ddof=0

## Notes
- All plots/analyses in Phase B use the **standardized** dataset (`data_prepared_for_dml.csv`).
- For “units on axes”, prefer explicit labels like **Z-score (standardized)**, and keep original units documented in `data_dictionary.csv`.
