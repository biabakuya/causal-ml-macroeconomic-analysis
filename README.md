# Causal ML Macroeconomic Analysis

This repository contains the code, datasets, figures, and empirical results for the study of the causal effect of investment on GDP growth using Causal Discovery and Double Machine Learning, on a heterogeneous panel of seven countries (2000–2024).

It accompanies the article *"Beyond Correlation: Quantifying the Causal Effect of Investment on Growth through Causal Discovery and Double Machine Learning"* (submitted to *IEEE Access*).

---

## Project Objective

The objective of this work is to quantify the causal effect of total investment (gross capital formation) on GDP growth, while rigorously controlling for macroeconomic confounders, using modern causal inference and machine learning techniques.

The project combines:

- Causal Discovery methods (Granger, VAR, VECM, PCMCI)
- A multi-method consensus causal graph (DAG) with support scores
- Double Machine Learning (LinearDML, CausalForestDML)
- Heterogeneous Treatment Effect estimation

The analysis focuses on seven countries over the period 2000–2024.

---

## Countries Studied

- France
- Germany
- Morocco
- Ghana
- Nigeria
- Angola
- DRC (Democratic Republic of the Congo)

Two development groups are distinguished: developed (France, Germany) and developing (Angola, Ghana, Nigeria, Morocco, DRC).

---

## Methodological Pipeline

The analysis pipeline follows these main stages:

1. Multi-source data collection and harmonization (IMF WEO, IMF IFS, World Bank)
2. Data preprocessing (log transforms, winsorization, standardization)
3. Lagged temporal structure construction (treatment at t-1, outcome at t)
4. Causal Discovery (Granger, VAR/IRF/FEVD, VECM/Johansen, PCMCI)
5. Multi-method DAG synthesis with support scores
6. Double Machine Learning estimation (global, by country, by group)
7. Heterogeneous effect analysis and visualization

---

## Main Methods Used

### Causal Discovery

- Granger Causality
- VAR (Vector Autoregression) with IRF and FEVD
- VECM (Vector Error Correction Model) with Johansen test
- PCMCI (with CMIknn conditional independence test)
- Multi-method DAG synthesis

### Double Machine Learning

- LinearDML
- CausalForestDML

### Nuisance Models (compared)

- Random Forest (retained for stability)
- XGBoost
- Gradient Boosting
- Ridge Regression
- Linear Regression
- Lasso

---

## Main Variables

- GDP Growth (outcome)
- Capital Formation / total investment (treatment)
- Inflation (confounder)
- Government Debt (confounder)
- Trade Balance / current account balance (confounder)
- Exchange Rate (confounder)
- Reserves (confounder)

---

## Repository Structure

```text
causal-ml-macroeconomic-analysis/
│
├── data/                  # Raw and processed datasets
│   ├── raw/               # Source files (IMF WEO, IMF IFS)
│   └── processed/         # Harmonized and prepared datasets
├── docs/                  # Documentation files
├── notebooks/             # Analysis scripts
├── reports/               # Figures and summary tables
├── results/               # Per-country statistical outputs
├── requirements.txt
└── README.md
```

---

## Key Findings

After the full corrected pipeline, the main results are:

- The net effect of investment on growth is **not statistically significant** at the global level (ATE = -0.344, p = 0.165), nor at the country level, nor between development groups.
- The two development groups yield effects of the same (mildly negative) sign, with no evidence of two opposing causal regimes.
- The value of the multi-method approach lies in its **discriminating power**: the conditional PCMCI test discards the Granger-detected links in Germany and Angola as false positives, while retaining a robust direct link (Capital_Formation → GDP_Growth) in **Nigeria only** (support score of 3).
- These results illustrate the importance of cautious causal inference on short macroeconomic time series.

---

## Repository Content

The repository includes scripts and outputs for:

- preprocessing and harmonization
- descriptive and stationarity analysis
- Granger causality analysis
- VAR estimation (IRF, FEVD)
- cointegration and VECM estimation
- PCMCI causal discovery
- multi-method DAG synthesis
- Double Machine Learning estimation
- heterogeneous treatment effect analysis
- graphical visualizations and exported CSV reports

---

## Figures

The repository includes visualizations for each methodological phase, available in:

```text
reports/figures/
```

Some figures are not directly integrated into the article due to formatting and readability limitations, but are provided here for completeness.

---

## Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- EconML
- Statsmodels
- Tigramite
- NetworkX

---

## Installation

Clone the repository:

```bash
git clone https://github.com/biabakuya/causal-ml-macroeconomic-analysis.git
```

Move into the project directory:

```bash
cd causal-ml-macroeconomic-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Reproducibility

The repository allows reproduction of the empirical analysis. All scripts required for preprocessing, econometric estimation, causal discovery, DML estimation, figure generation, and statistical export are included. The analysis scripts are located in `notebooks/`.

---

## Academic Context

This project was developed as part of a Master's internship research project at the ABIL Laboratory, University of Kinshasa (UNIKIN), in collaboration with the International School, Vietnam National University in Hanoi (VNU), focusing on causal inference, macroeconomic analysis, machine learning, and applied econometrics.

---

## Author

Jirince K. Biaba — Master's Internship Project, 2025
