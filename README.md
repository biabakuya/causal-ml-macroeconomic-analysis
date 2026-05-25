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

# Repository Content

The repository contains:

- preprocessing scripts
- descriptive analysis
- stationarity analysis
- Granger causality analysis
- VAR estimation
- cointegration analysis
- VECM estimation
- PCMCI causal discovery
- DAG synthesis
- Double Machine Learning estimation
- heterogeneous treatment effect analysis
- graphical visualizations
- exported CSV reports

---

# Key Results

The project provides:

- Temporal causal structures by country
- Directed Acyclic Graphs (DAGs)
- Average Treatment Effects (ATE)
- Heterogeneous Treatment Effects (HTE)
- Country-level causal comparisons
- Developed vs developing country comparisons
- Machine learning nuisance model evaluation
- Comparative macroeconomic interpretations

---

# Example Figures Included

The repository includes multiple visualizations such as:

- GDP Growth evolution
- Gross Capital Formation trends
- Correlation matrices
- Scatter plots
- Granger causality heatmaps
- VAR impulse response functions
- FEVD visualizations
- PCMCI causal graphs
- Final DAGs by country
- DML Average Treatment Effects
- Heterogeneous Treatment Effect distributions
- Feature importance visualizations
- Nuisance model performance comparison

---

# Technologies Used

The project was implemented using:

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- EconML
- Statsmodels
- Tigramite
- NetworkX

---

# Installation

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

# Reproducibility

The repository allows full reproducibility of the empirical analysis.

All scripts necessary for:

- preprocessing
- econometric estimation
- causal discovery
- DML estimation
- figure generation
- statistical export

are included.

---

# Figures and Additional Visualizations

Some figures and visualizations included in this repository are not directly integrated into the report/article due to formatting and readability limitations.

Additional graphical outputs are available in:

```text
reports/figures/
```

Each figure corresponds to a specific methodological phase.

---

# Academic Context

This project was developed as part of a Master's internship research project focused on:

- causal inference
- macroeconomic analysis
- machine learning
- applied econometrics

---

# Author

Jirince Biaba  Kuya
Master Internship Project — 2025
