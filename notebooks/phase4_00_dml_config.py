#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PHASE 4 — CONFIGURATION DML (VERSION LAGGED)

Structure temporelle utilisée :

Capital_Formation(t-1)  →  GDP_Growth(t)

Toutes les variables explicatives sont considérées en t-1
afin d'éviter la causalité inverse et les problèmes de simultanéité.
"""

from pathlib import Path

# =========================================================
# Input dataset (VERSION LAGGED)
# =========================================================

DATA_PATH = Path("../data/processed/data_prepared_for_dml_lagged.csv")

# =========================================================
# Output directories
# =========================================================

RES_DIR = Path("../results/phase4_dml")
FIG_DIR = Path("../reports/figures/phase4_dml")

RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# DML specification
# =========================================================

# Outcome variable
Y_COL = "GDP_Growth"

# Treatment variable (Capital Formation t-1)
T_COL = "Capital_Formation"

# Confounders (t-1)
BASE_X_COLS = [
    "Inflation",
    "Government_Debt",
    "Trade_Balance",
    "Exchange_Rate",
    "Reserves",
]

# Panel structure
COUNTRY_COL = "Country"
YEAR_COL = "Year"

# Random seed
RANDOM_STATE = 42

# Cross-fitting folds
N_SPLITS = 5