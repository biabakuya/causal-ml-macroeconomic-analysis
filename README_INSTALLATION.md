# 🚀 GUIDE D'INSTALLATION - STAGE ABIL 2025

**Causal Machine Learning appliqué à l'analyse macroéconomique**

---

## 📋 TABLE DES MATIÈRES

1. [Prérequis](#prérequis)
2. [Installation automatique](#installation-automatique)
3. [Installation manuelle](#installation-manuelle)
4. [Vérification](#vérification)
5. [Structure du projet](#structure-du-projet)
6. [Dépannage](#dépannage)
7. [Utilisation quotidienne](#utilisation-quotidienne)

---

## ✅ PRÉREQUIS

### **Logiciels requis :**

1. **Python 3.8+** (vous avez 3.12.10 ✅)
   - Vérification : `python --version`
   
2. **pip** (vous avez 25.0.1 ✅)
   - Vérification : `pip --version`

3. **Git** (optionnel, pour versioning)
   - Installation : https://git-scm.com/downloads

4. **Graphviz** (optionnel, pour visualisation DAG)
   - Windows : https://graphviz.org/download/
   - Après installation, ajouter au PATH : `C:\Program Files\Graphviz\bin`

---

## 🤖 INSTALLATION AUTOMATIQUE (RECOMMANDÉ)

### **Étape 1 : Télécharger les fichiers**

Assurez-vous d'avoir ces fichiers dans votre dossier de travail :
- `setup_environment.py`
- `requirements.txt`
- `test_installation.py`

### **Étape 2 : Exécuter le script d'installation**

```bash
python setup_environment.py
```

Ce script va :
- ✅ Créer l'environnement virtuel `venv_stage`
- ✅ Installer toutes les dépendances (5-10 minutes)
- ✅ Créer la structure de dossiers
- ✅ Configurer le .gitignore

**⏳ Patience :** L'installation prend 5-10 minutes selon votre connexion.

### **Étape 3 : Activer l'environnement**

**Windows :**
```bash
venv_stage\Scripts\activate
```

**Linux/Mac :**
```bash
source venv_stage/bin/activate
```

✅ Vous devriez voir `(venv_stage)` au début de votre ligne de commande.

### **Étape 4 : Tester l'installation**

```bash
python test_installation.py
```

Ce script va vérifier que tout est bien installé.

---

## 🔧 INSTALLATION MANUELLE

Si le script automatique échoue, voici les étapes manuelles :

### **1. Créer l'environnement virtuel**

```bash
python -m venv venv_stage
```

### **2. Activer l'environnement**

**Windows :**
```bash
venv_stage\Scripts\activate
```

**Linux/Mac :**
```bash
source venv_stage/bin/activate
```

### **3. Mettre à jour pip**

```bash
python -m pip install --upgrade pip
```

### **4. Installer les dépendances**

```bash
pip install -r requirements.txt
```

**Si une installation échoue**, essayez package par package :

```bash
# Essentiels d'abord
pip install pandas numpy matplotlib seaborn scipy statsmodels

# Machine Learning
pip install scikit-learn xgboost lightgbm

# Causal ML
pip install dowhy
pip install econml
pip install causalml
pip install tigramite

# Séries temporelles
pip install pmdarima arch

# Graphes
pip install networkx pydot

# Jupyter
pip install jupyter jupyterlab ipykernel

# Utilitaires
pip install tqdm requests
```

### **5. Créer la structure de dossiers**

```bash
mkdir -p data/raw data/processed data/results
mkdir -p scripts/phase1 scripts/phase2 scripts/phase3 scripts/phase4 scripts/phase5
mkdir -p outputs/figures outputs/tables outputs/reports
mkdir -p notebooks models docs
```

---

## ✅ VÉRIFICATION

### **Test complet :**

```bash
python test_installation.py
```

### **Test rapide dans Python :**

```python
python
>>> import pandas as pd
>>> import numpy as np
>>> import dowhy
>>> import econml
>>> import tigramite
>>> print("✅ Tout fonctionne !")
>>> exit()
```

---

## 📁 STRUCTURE DU PROJET

Après installation, votre projet aura cette structure :

```
stage_abil_2025/
│
├── venv_stage/              # Environnement virtuel (NE PAS VERSIONNER)
│
├── data/
│   ├── raw/                 # Données brutes (CSV de la Banque Mondiale)
│   ├── processed/           # Données nettoyées
│   └── results/             # Résultats des analyses
│
├── scripts/
│   ├── phase1/              # Scripts Phase 1 (préparation données)
│   ├── phase2/              # Scripts Phase 2 (cas France)
│   ├── phase3/              # Scripts Phase 3 (cas RDC)
│   ├── phase4/              # Scripts Phase 4 (synthèse)
│   └── phase5/              # Scripts Phase 5 (article)
│
├── notebooks/               # Jupyter notebooks
│
├── outputs/
│   ├── figures/             # Graphiques générés
│   ├── tables/              # Tableaux de résultats
│   └── reports/             # Rapports PDF/HTML
│
├── models/                  # Modèles sauvegardés
│
├── docs/                    # Documentation
│
├── requirements.txt         # Liste des dépendances
├── setup_environment.py     # Script d'installation
├── test_installation.py     # Script de test
└── README_INSTALLATION.md   # Ce fichier
```

---

## 🔧 DÉPANNAGE

### **Problème 1 : `pip` introuvable après activation**

**Solution :**
```bash
python -m pip install --upgrade pip
```

### **Problème 2 : Erreur avec `pygraphviz`**

**C'est normal !** Ce package est optionnel.

**Solution :** Commentez cette ligne dans `requirements.txt` :
```
# pygraphviz>=1.11
```

Puis réinstallez :
```bash
pip install -r requirements.txt
```

### **Problème 3 : Erreur "Microsoft Visual C++ required"**

**Windows uniquement.** Certains packages (comme `econml`) nécessitent les outils de compilation.

**Solution :**
1. Téléchargez "Build Tools for Visual Studio" : https://visualstudio.microsoft.com/downloads/
2. Installez "Desktop development with C++"
3. Réessayez l'installation

**Alternative :** Utilisez les wheels précompilés :
```bash
pip install --only-binary :all: econml
```

### **Problème 4 : Installation lente**

**C'est normal !** Certains packages sont volumineux :
- `scipy` : ~50 MB
- `pandas` : ~40 MB
- `xgboost` : ~100 MB
- `econml` : ~20 MB + dépendances

**Total : ~500 MB de téléchargement**

⏳ **Patience : 5-10 minutes selon votre connexion**

### **Problème 5 : "Cannot import name 'X' from 'Y'"**

Conflit de versions.

**Solution :**
```bash
pip install --upgrade [package_name]
```

Ou réinstallez tout :
```bash
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

### **Problème 6 : L'environnement ne s'active pas**

**Windows PowerShell :** Vous devez peut-être autoriser l'exécution de scripts.

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Puis réessayez l'activation.

---

## 💼 UTILISATION QUOTIDIENNE

### **Démarrer une session de travail**

1. **Ouvrir un terminal** dans le dossier du projet

2. **Activer l'environnement :**
   ```bash
   # Windows
   venv_stage\Scripts\activate
   
   # Linux/Mac
   source venv_stage/bin/activate
   ```

3. **Vérifier l'activation :** Vous devez voir `(venv_stage)` au début de la ligne de commande

4. **Lancer Jupyter (optionnel) :**
   ```bash
   jupyter lab
   ```

5. **Ou exécuter vos scripts :**
   ```bash
   python scripts/phase1/01_import_data.py
   ```

### **Terminer une session**

```bash
deactivate
```

### **Ajouter un nouveau package**

```bash
# Activer l'environnement d'abord
pip install nom_du_package

# Mettre à jour requirements.txt
pip freeze > requirements.txt
```

---

## 📚 RESSOURCES

### **Documentation des outils principaux :**

- **DoWhy :** https://www.pywhy.org/dowhy/
- **EconML :** https://econml.azurewebsites.net/
- **CausalML :** https://causalml.readthedocs.io/
- **TIGRAMITE :** https://github.com/jakobrunge/tigramite
- **Statsmodels :** https://www.statsmodels.org/
- **Pandas :** https://pandas.pydata.org/
- **Scikit-learn :** https://scikit-learn.org/

### **Tutoriels et guides :**

- **Causal Inference Book :** https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/
- **DoWhy Tutorials :** https://microsoft.github.io/dowhy/
- **EconML Examples :** https://github.com/microsoft/EconML/tree/main/notebooks

---

## ✅ CHECKLIST FINALE

Avant de commencer la Phase 1, vérifiez que :

- [ ] L'environnement virtuel est créé
- [ ] Toutes les dépendances sont installées (test_installation.py = succès)
- [ ] La structure de dossiers existe
- [ ] Vous savez activer/désactiver l'environnement
- [ ] Jupyter fonctionne (optionnel mais recommandé)

---

## 🎯 PRÊT À COMMENCER !

Une fois l'installation terminée avec succès :

1. **Placez vos données** : `data/raw/votre_fichier.csv`
2. **Consultez** : `README_PHASE1.md`
3. **Exécutez** : Les scripts de la Phase 1

---

**Bon travail ! 🚀**

*Document créé le 12 décembre 2024*  
*Stage ABIL 2025 - Université de Kinshasa*
