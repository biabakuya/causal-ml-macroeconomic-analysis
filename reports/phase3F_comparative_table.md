# Phase 3 — Tableau comparatif multi-pays (Granger + VAR + VECM + PCMCI + DAG)

| Country          | Capital_Centrality   |   Capital_in_degree |   Capital_out_degree |   Capital_degree | Direct_Capital_to_GDP   |   Support(Capital->GDP) | Methods(Capital->GDP)   |   Support_mean |   Support_max |   VAR_IRF_max_abs | VAR_IRF_sign   |   VAR_IRF_best_horizon |   PCMCI_links_count | DAG_structure   |
|:-----------------|:---------------------|--------------------:|---------------------:|-----------------:|:------------------------|------------------------:|:------------------------|---------------:|--------------:|------------------:|:---------------|-----------------------:|--------------------:|:----------------|
| France           | Faible               |                   0 |                    0 |                0 | Non                     |                     nan |                         |        1.16667 |             2 |           72.3297 | Positive       |                      1 |                   3 | Intermédiaire   |
| Allemagne        | Faible               |                   0 |                    0 |                0 | Non                     |                     nan |                         |        1       |             1 |           20.1595 | Négative       |                      6 |                   0 | Intégrée        |
| Congo, Dem. Rep. | Faible               |                   0 |                    0 |                0 | Non                     |                     nan |                         |        1       |             1 |            1.3802 | Positive       |                      1 |                   3 | Fragmentée      |
| Nigeria          | Faible               |                   0 |                    0 |                0 | Non                     |                     nan |                         |        1.2     |             2 |           28.4384 | Positive       |                      2 |                   8 | Fragmentée      |
| Angola           | Faible               |                   0 |                    0 |                0 | Non                     |                     nan |                         |        1.33333 |             2 |           14.107  | Positive       |                      1 |                   0 | Intermédiaire   |
| Maroc            | Faible               |                   0 |                    0 |                0 | Non                     |                     nan |                         |        1.33333 |             2 |           33.4409 | Négative       |                      4 |                   0 | Intégrée        |
| Ghana            | Faible               |                   0 |                    0 |                0 | Non                     |                     nan |                         |        1.05128 |             3 |           10.6236 | Positive       |                      2 |                  34 | Fragmentée      |


**Lecture rapide :**
- `Direct_Capital_to_GDP`: le lien direct Capital_Formation → GDP_Growth apparaît (support multi-méthodes)
- `Support_mean/max`: cohérence globale entre méthodes
- `VAR_IRF_*`: réponse de GDP_Growth à un choc sur Capital_Formation
- `DAG_structure`: densité/complexité de la structure causale
