# 📊 PHASE 1 : PRÉPARATION DES DONNÉES

**Stage ABIL 2025 - Causal Machine Learning en Macroéconomie**  
**Auteur:** Jirince K. Biaba  
**Durée estimée:** 3-5 jours

---

## 🎯 OBJECTIFS DE LA PHASE 1

Cette phase prépare les données macroéconomiques pour l'analyse causale :

1. ✅ **Import et nettoyage** des données brutes
2. ✅ **Gestion des valeurs manquantes** (imputation)
3. ✅ **Analyse exploratoire** (statistiques descriptives)
4. ✅ **Visualisations** comparatives France vs RDC
5. ✅ **Tests de stationnarité** et différenciation

---

## 📁 FICHIERS DE LA PHASE 1

### **Scripts Python**

1. `phase1_01_import_clean.py` - Import et nettoyage initial
2. `phase1_02_missing_data.py` - Gestion des données manquantes
3. `phase1_03_eda_statistics.py` - Statistiques descriptives
4. `phase1_04_visualizations.py` - Graphiques et visualisations
5. `phase1_05_stationarity.py` - Tests de stationnarité

### **Données d'entrée (à placer dans `data/raw/`)**

- `Data.csv` - Fichier principal de la Banque Mondiale
- `Series_-_Metadata.csv` - Métadonnées (optionnel)

---

## 🚀 EXÉCUTION ÉTAPE PAR ÉTAPE

### **Prérequis**

1. Environnement virtuel activé :
   ```bash
   venv_stage\Scripts\activate
   ```

2. Données placées dans `data/raw/` :
   - Votre fichier `Data.csv` téléchargé de la Banque Mondiale

---

### **SCRIPT 1 : Import et Nettoyage**

```bash
python phase1_01_import_clean.py
```

**Ce que fait ce script :**
- Importe le CSV de la Banque Mondiale
- Transforme du format Wide (années en colonnes) au format Long (années en lignes)
- Remplace les ".." par NaN
- Crée des noms courts pour les indicateurs
- Génère un rapport des valeurs manquantes

**Fichiers créés :**
- `data/processed/data_cleaned.csv`
- `data/processed/missing_data_report.csv`

**Temps d'exécution :** ~10 secondes

---

### **SCRIPT 2 : Gestion des Données Manquantes**

```bash
python phase1_02_missing_data.py
```

**Ce que fait ce script :**
- Analyse détaillée des patterns de données manquantes
- Applique interpolation linéaire pour les gaps internes
- Utilise forward/backward fill pour les extrémités
- Crée une heatmap des données manquantes
- Génère un rapport d'imputation

**Fichiers créés :**
- `data/processed/data_imputed.csv`
- `data/processed/imputation_report.csv`
- `outputs/figures/missing_data_heatmap.png`

**Temps d'exécution :** ~15 secondes

---

### **SCRIPT 3 : Statistiques Descriptives**

```bash
python phase1_03_eda_statistics.py
```

**Ce que fait ce script :**
- Calcule les statistiques descriptives par pays et indicateur
- Compare France vs RDC (moyennes, ratios)
- Analyse les tendances (CAGR - taux de croissance annuel moyen)
- Évalue la volatilité (coefficient de variation)
- Identifie les valeurs extrêmes (outliers)

**Fichiers créés :**
- `outputs/reports/descriptive_statistics.csv`
- `outputs/reports/trends_analysis.csv`

**Temps d'exécution :** ~20 secondes

---

### **SCRIPT 4 : Visualisations**

```bash
python phase1_04_visualizations.py
```

**Ce que fait ce script :**
- Graphiques de séries temporelles (1 par indicateur)
- Graphique comparatif global (tous indicateurs)
- Matrices de corrélation (France et RDC)
- Boxplots des distributions
- Évolutions indexées (base 100)
- Scatter plots (relations entre indicateurs)

**Fichiers créés :**
- `outputs/figures/timeseries_*.png` (8 graphiques)
- `outputs/figures/all_indicators_comparison.png`
- `outputs/figures/correlation_matrices.png`
- `outputs/figures/boxplots_comparison.png`
- `outputs/figures/indexed_evolution.png`
- `outputs/figures/scatter_plots.png`

**Temps d'exécution :** ~30 secondes

---

### **SCRIPT 5 : Tests de Stationnarité**

```bash
python phase1_05_stationarity.py
```

**Ce que fait ce script :**
- Effectue les tests ADF (Augmented Dickey-Fuller)
- Effectue les tests KPSS (Kwiatkowski-Phillips-Schmidt-Shin)
- Identifie les séries non-stationnaires
- Applique la différenciation (d=1) si nécessaire
- Crée un graphique des résultats de stationnarité

**Fichiers créés :**
- `data/processed/data_stationary.csv`
- `outputs/reports/stationarity_tests.csv`
- `outputs/reports/differencing_log.csv`
- `outputs/figures/stationarity_tests.png`

**Temps d'exécution :** ~25 secondes

---

## 📊 RÉSULTATS ATTENDUS

### **Données préparées**

✅ **data_stationary.csv** - Données prêtes pour la modélisation :
- 2 pays × 8 indicateurs × 15 années = 240 observations
- Valeurs manquantes imputées
- Séries stationnaires (différenciées si nécessaire)

### **Rapports statistiques**

✅ **descriptive_statistics.csv** :
- Moyenne, écart-type, min, max, quartiles
- Coefficient de variation
- Par pays et indicateur

✅ **trends_analysis.csv** :
- Taux de croissance annuel moyen (CAGR)
- Tendances de croissance/décroissance

✅ **stationarity_tests.csv** :
- Résultats des tests ADF et KPSS
- Détermination de la stationnarité
- P-values et statistiques de test

### **Visualisations**

✅ **~15 graphiques haute qualité** (PNG, 300 DPI) :
- Séries temporelles individuelles et comparatives
- Matrices de corrélation
- Distributions (boxplots)
- Relations entre variables (scatter plots)
- Heatmap des données manquantes
- Résultats des tests de stationnarité

---

## 🔍 VÉRIFICATION DES RÉSULTATS

Après l'exécution de tous les scripts, vérifiez que vous avez :

### **Dans `data/processed/`**
```
✓ data_cleaned.csv
✓ data_imputed.csv
✓ data_stationary.csv
✓ missing_data_report.csv
✓ imputation_report.csv
```

### **Dans `outputs/reports/`**
```
✓ descriptive_statistics.csv
✓ trends_analysis.csv
✓ stationarity_tests.csv
✓ differencing_log.csv (si différenciation appliquée)
```

### **Dans `outputs/figures/`**
```
✓ missing_data_heatmap.png
✓ timeseries_*.png (8 fichiers)
✓ all_indicators_comparison.png
✓ correlation_matrices.png
✓ boxplots_comparison.png
✓ indexed_evolution.png
✓ scatter_plots.png
✓ stationarity_tests.png
```

---

## ⚠️ PROBLÈMES COURANTS ET SOLUTIONS

### **Problème 1 : "FileNotFoundError: Data.csv not found"**

**Solution :**
```bash
# Vérifiez que votre fichier CSV est dans le bon dossier
ls data/raw/Data.csv

# Si absent, placez-le dans data/raw/
```

### **Problème 2 : "Module not found: pandas/numpy/etc."**

**Solution :**
```bash
# Réactivez l'environnement virtuel
venv_stage\Scripts\activate

# Vérifiez l'installation
python test_installation.py

# Réinstallez si nécessaire
pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

### **Problème 3 : "Données encore manquantes après imputation"**

**C'est normal !** Certaines séries sont complètement vides (ex: dette publique France).

**Ce qui se passe :**
- L'imputation ne peut PAS créer des données à partir de rien
- Si une série entière est vide (tous les ".." pour toutes les années), elle reste vide
- Ces séries seront exclues des analyses qui les nécessitent

**Séries problématiques identifiées :**
- France : Taux d'intérêt réel (complètement vide)
- France : Dette publique (complètement vide)
- RDC : Plusieurs valeurs manquantes pour inflation et dette

### **Problème 4 : "Graphiques ne s'affichent pas"**

**C'est normal !** Les graphiques sont sauvegardés directement en fichiers PNG.

**Pour les voir :**
```bash
# Ouvrez le dossier
explorer outputs\figures\

# Ou utilisez un visualiseur d'images
```

### **Problème 5 : "Tests de stationnarité échouent"**

**Causes possibles :**
- Pas assez de données (< 3 observations)
- Série complètement vide

**Solution :** Le script gère automatiquement ces cas et continue avec les autres séries.

---

## 📈 INSIGHTS CLÉS À OBSERVER

### **Après SCRIPT 1 (Import)**
- Combien de valeurs manquantes par indicateur ?
- Quels indicateurs sont complets ?

### **Après SCRIPT 2 (Imputation)**
- Combien de valeurs ont été imputées ?
- Quelles séries restent problématiques ?

### **Après SCRIPT 3 (Statistiques)**
- Quelles sont les différences France vs RDC ?
- Quels indicateurs sont les plus volatiles ?
- Quelles sont les tendances de croissance ?

### **Après SCRIPT 4 (Visualisations)**
- Y a-t-il des corrélations fortes entre indicateurs ?
- Les tendances sont-elles claires visuellement ?
- Y a-t-il des patterns différents France vs RDC ?

### **Après SCRIPT 5 (Stationnarité)**
- Combien de séries sont stationnaires ?
- Quelles séries nécessitent différenciation ?
- Les résultats ADF et KPSS concordent-ils ?

---

## 🚀 PROCHAINES ÉTAPES (PHASE 2)

Une fois la Phase 1 terminée, vous êtes prêt pour la **Phase 2** :

**Phase 2 : Cas France - Validation méthodologique**
- Modèles économétriques traditionnels (VAR, VECM, ARIMA, Granger)
- Application complète du CML (PCMCI, DoWhy, EconML)
- Comparaison CML vs modèles traditionnels

**Durée estimée Phase 2 :** 7-10 jours

---

## 💡 CONSEILS

1. **Exécutez les scripts dans l'ordre** (01 → 05)
2. **Vérifiez les résultats** après chaque script
3. **Consultez les rapports CSV** pour des détails
4. **Examinez les graphiques** pour valider visuellement
5. **Prenez des notes** sur les insights découverts

---

## 📚 RÉFÉRENCES

- **Banque Mondiale - World Development Indicators**
- **Pandas Documentation** : https://pandas.pydata.org/
- **Statsmodels Documentation** : https://www.statsmodels.org/
- **Tests de stationnarité** :
  - ADF : Dickey & Fuller (1979)
  - KPSS : Kwiatkowski et al. (1992)

---

**Document créé le 12 décembre 2024**  
**Stage ABIL 2025 - Université de Kinshasa**
