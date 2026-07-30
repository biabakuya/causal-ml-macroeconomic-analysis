# Phase 3 — Tableau comparatif multi-pays (Granger + VAR + VECM + PCMCI + DAG)

| Country   | Capital_Centrality   |   Capital_in_degree |   Capital_out_degree |   Capital_degree | Direct_Capital_to_GDP   |   Support(Capital->GDP) | Methods(Capital->GDP)   |   Support_mean |   Support_max |   VAR_IRF_max_abs | VAR_IRF_sign   |   VAR_IRF_best_horizon |   PCMCI_links_count | DAG_structure   |
|:----------|:---------------------|--------------------:|---------------------:|-----------------:|:------------------------|------------------------:|:------------------------|---------------:|--------------:|------------------:|:---------------|-----------------------:|--------------------:|:----------------|
| France    | Faible               |                   0 |                    0 |                0 | Non                     |                     nan |                         |        1.11111 |             2 |           84.5845 | Positive       |                      1 |                   6 | Intermédiaire   |
| Germany   | Faible               |                   0 |                    0 |                0 | Non                     |                     nan |                         |        1.2     |             2 |           15.4272 | Négative       |                      1 |                   0 | Intermédiaire   |
| DRC       | NA                   |                 nan |                  nan |              nan | Non                     |                     nan |                         |      nan       |           nan |          nan      | NA             |                    nan |                   0 | Fragmentée      |
| Nigeria   | Faible               |                   0 |                    0 |                0 | Non                     |                     nan |                         |        1.2     |             3 |           29.2093 | Positive       |                      2 |                  14 | Fragmentée      |
| Angola    | Faible               |                   0 |                    0 |                0 | Non                     |                     nan |                         |        1.13636 |             2 |            8.9882 | Positive       |                      1 |                  17 | Fragmentée      |
| Morocco   | Faible               |                   0 |                    0 |                0 | Non                     |                     nan |                         |        1.125   |             2 |           39.7266 | Négative       |                      4 |                   5 | Fragmentée      |
| Ghana     | Faible               |                   0 |                    0 |                0 | Non                     |                     nan |                         |        1.02564 |             2 |           12.0051 | Positive       |                      2 |                  34 | Fragmentée      |


**Lecture rapide :**
- `Direct_Capital_to_GDP`: le lien direct Capital_Formation → GDP_Growth apparaît (support multi-méthodes)
- `Support_mean/max`: cohérence globale entre méthodes
- `VAR_IRF_*`: réponse de GDP_Growth à un choc sur Capital_Formation
- `DAG_structure`: densité/complexité de la structure causale
