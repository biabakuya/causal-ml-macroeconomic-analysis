# Causal ML Macroeconomic Analysis

This repository contains the code, datasets, figures, and empirical results developed during my internship project on causal analysis and machine learning applied to macroeconomic data.

---

# Project Objective

The objective of this work is to study the causal impact of macroeconomic variables on GDP growth using modern causal inference and machine learning techniques.

The project combines:

- Causal Discovery methods
- Econometric approaches
- Double Machine Learning (DML)
- Heterogeneous Treatment Effect estimation

The analysis focuses on 7 countries over the period 2000–2024.

---

# Countries Studied

- Germany
- Angola
- Congo, Dem. Rep.
- France
- Ghana
- Morocco
- Nigeria

---

# Methodological Pipeline

The analysis pipeline follows these main stages:

1. Data collection and harmonization
2. Data preprocessing
3. Lagged temporal structure construction
4. Causal Discovery
5. Double Machine Learning estimation
6. Heterogeneous effect analysis
7. Interpretation and visualization

---

# Main Methods Used

## Causal Discovery

- Granger Causality
- VAR (Vector Autoregression)
- VECM (Vector Error Correction Model)
- PCMCI
- DAG synthesis

## Double Machine Learning

- LinearDML
- CausalForestDML

## Machine Learning Models

- Random Forest
- XGBoost
- Gradient Boosting
- Ridge Regression
- Linear Regression
- Lasso

---

# Main Variables

- GDP Growth
- Gross Capital Formation
- Exchange Rate
- Inflation
- Government Debt
- Trade Balance
- Reserves

---

# Repository Structure

```text
causal-ml-macroeconomic-analysis/
│
├── data/                  # Raw and processed datasets
├── docs/                  # Documentation files
├── notebooks/             # Analysis scripts and notebooks
├── reports/               # Figures and tables
├── results/               # Statistical outputs
├── requirements.txt
└── README.md