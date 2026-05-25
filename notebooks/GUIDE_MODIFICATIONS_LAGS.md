# GUIDE DE MODIFICATION - RÉSOLUTION Y → T
## Stage ABIL 2025 - Jirince BIABA KUYA

---

## ✅ ÉTAPE 0 : EXÉCUTER LE SCRIPT DE CRÉATION DES LAGS

```bash
cd notebooks
python phase0_create_lagged_data.py
```

**Résultat :**
- ✅ data_transformed_unscaled_lagged.csv (174 obs)
- ✅ data_prepared_for_dml_lagged.csv (174 obs)
- ✅ phase0_lagged_data_report.txt

---

## 📝 ÉTAPE 1 : MODIFIER LES SCRIPTS PHASE 3

### **Fichier 1 : phase3A_01_granger.py**

**Ligne ~45-50 (section INPUT)**

**AVANT :**
```python
# recommandé (transformé, non standardisé)
INPUT_PRIMARY = DATA_PROCESSED / "data_transformed_unscaled.csv"
INPUT_FALLBACK = DATA_PROCESSED / "data_prepared_for_dml.csv"
```

**APRÈS :**
```python
# recommandé (transformé, non standardisé, LAGGÉ)
INPUT_PRIMARY = DATA_PROCESSED / "data_transformed_unscaled_lagged.csv"
INPUT_FALLBACK = DATA_PROCESSED / "data_transformed_unscaled.csv"  # fallback si lagged absent
```

---

### **Fichier 2 : phase3D_00_prepare_pcmci_data.py**

**Ligne ~20 (INPUT_PATH)**

**AVANT :**
```python
INPUT_PATH = Path("../data/processed/data_transformed_unscaled.csv")
```

**APRÈS :**
```python
INPUT_PATH = Path("../data/processed/data_transformed_unscaled_lagged.csv")
```

---

### **Fichier 3 : phase3B_01_var.py**

**Ligne ~35 (parser.add_argument)**

**AVANT :**
```python
parser.add_argument("--input", type=str, default="../data/processed/data_transformed_unscaled.csv")
```

**APRÈS :**
```python
parser.add_argument("--input", type=str, default="../data/processed/data_transformed_unscaled_lagged.csv")
```

---

### **Fichier 4 : phase3C_01_cointegration.py**

**Ligne ~18 (INPUT_FILE)**

**AVANT :**
```python
INPUT_FILE = Path("../data/processed/data_transformed_unscaled.csv")
```

**APRÈS :**
```python
INPUT_FILE = Path("../data/processed/data_transformed_unscaled_lagged.csv")
```

---

### **Fichier 5 : phase3C_02_vecm_estimation.py**

**Ligne ~93 (INPUT_FILE)**

**AVANT :**
```python
INPUT_FILE = DATA_PROCESSED / "data_transformed_unscaled.csv"
```

**APRÈS :**
```python
INPUT_FILE = DATA_PROCESSED / "data_transformed_unscaled_lagged.csv"
```

---

## 🚀 ÉTAPE 2 : RE-EXÉCUTER PHASE 3 (pour les 7 pays)

```bash
# Granger
python phase3A_01_granger.py --country "France"
python phase3A_01_granger.py --country "Allemagne"
python phase3A_01_granger.py --country "Ghana"
python phase3A_01_granger.py --country "Maroc"
python phase3A_01_granger.py --country "Congo, Dem. Rep."
python phase3A_01_granger.py --country "Angola"
python phase3A_01_granger.py --country "Nigeria"

# PCMCI (prepare + run)
python phase3D_00_prepare_pcmci_data.py --country "France"
python phase3D_01_pcmci.py --country "France"
# ... répéter pour 7 pays

# VAR
python phase3B_01_var.py --country "France"
# ... répéter pour 7 pays

# Cointegration + VECM
python phase3C_01_cointegration.py --country "France"
python phase3C_02_vecm_estimation.py --country "France"
# ... répéter pour 7 pays

# Synthèse DAG
python phase3E_01_dag_synthesis.py
```

---

## ✅ ÉTAPE 3 : VÉRIFIER LES NOUVEAUX DAGs

**Ouvrir les DAGs générés et vérifier :**

❌ **Ne doit PLUS apparaître :** GDP_Growth → Capital_Formation  
✅ **Doit apparaître :** Capital_Formation → GDP_Growth  
✅ **Interprétation :** Capital_Formation(t-1) → GDP_Growth(t)

**Si GDP_Growth → Capital_Formation apparaît encore :**
→ PROBLÈME : Le script n'utilise pas le bon fichier
→ Vérifier que les modifications ont bien été sauvegardées

---

## 📊 ÉTAPE 4 : MODIFIER LES SCRIPTS DML (Phase 4)

**Quand tu me montreras les scripts DML, on changera :**

**AVANT :**
```python
INPUT_FILE = Path("../data/processed/data_prepared_for_dml.csv")
```

**APRÈS :**
```python
INPUT_FILE = Path("../data/processed/data_prepared_for_dml_lagged.csv")
```

---

## 📝 ÉTAPE 5 : DOCUMENTATION DANS L'ARTICLE

**Section Méthodologie - Ajouter :**

### 3.X Temporal Structure

> "To address potential simultaneity bias between capital formation and economic 
> growth, we employ a lagged specification where Capital_Formation(t-1) influences 
> GDP_Growth(t). This temporal structure is standard in macroeconomic panel analysis 
> (Granger, 1969; Koyck, 1954) and allows for unambiguous causal identification by 
> eliminating reverse causation through the temporal ordering constraint."

**Équation DML devient :**

Y_i,t = θ(T_i,t-1) + g(X_i,t-1) + ε_i,t

Où :
- Y_i,t = GDP_Growth du pays i à l'année t
- T_i,t-1 = Capital_Formation du pays i à l'année t-1
- X_i,t-1 = Confounders du pays i à l'année t-1

---

## 🎯 CHECKLIST FINALE

- [ ] Exécuté phase0_create_lagged_data.py
- [ ] Modifié phase3A_01_granger.py
- [ ] Modifié phase3D_00_prepare_pcmci_data.py
- [ ] Modifié phase3B_01_var.py
- [ ] Modifié phase3C_01_cointegration.py
- [ ] Modifié phase3C_02_vecm_estimation.py
- [ ] Re-exécuté Phase 3 pour les 7 pays
- [ ] Vérifié les nouveaux DAGs (pas de Y → T)
- [ ] Modifié scripts DML (Phase 4)
- [ ] Ajouté justification dans l'article

---

## ❓ EN CAS DE PROBLÈME

**Si les DAGs montrent encore GDP_Growth → Capital_Formation :**

1. Vérifier que les scripts utilisent bien `_lagged.csv`
2. Vérifier que les fichiers laggés existent dans data/processed/
3. Vérifier que la période est bien 2001-2024 (pas 2000-2024)
4. Contacter le superviseur si le problème persiste

---

**Bon courage ! 🚀**
