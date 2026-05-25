# Phase 4 — Double Machine Learning (DML)

**Objectif causal :** effet de **Capital_Formation (T)** sur **GDP_Growth (Y)** en contrôlant les confondeurs **X**.

## Variables

- **Y (outcome)** : `GDP_Growth`

- **T (treatment)** : `Capital_Formation`

- **X (confounders)** : `Inflation, Government_Debt, Trade_Balance, Exchange_Rate, Reserves` (+ dummies pays dans l’ATE global)


## Résultat principal (ATE global)

- **ATE** : -0.5280976960761746

- **IC 95%** : [-1.041839587845164, -0.0143558043071853]

- **StdErr** : 0.2621180265664675

- **p-value** : 0.043932689310933


## Hétérogénéité simple (ATE par pays)

Table enregistrée : `dml_hte_by_country.csv`

Figure : `../reports/figures/phase4_dml/dml_ate_by_country.png`
