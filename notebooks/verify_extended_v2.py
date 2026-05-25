#!/usr/bin/env python3
"""
VÉRIFICATION DES FICHIERS - DATASET ÉLARGI
7 pays × 25 ans (2000-2024) = 175 observations
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*100)
print(" " * 25 + "VÉRIFICATION DATASET ÉLARGI - 7 PAYS")
print("="*100)
print()

DATA_RAW = Path('../data/raw')

# Noms de fichiers EXACTS
FILES_WEO = {
    'Allemagne_France': DATA_RAW / 'WEO_Allemagne_France_2000_2024.xls',
    '4Subsaharan': DATA_RAW / 'WEO_4Subsaharan_2000_2024.xls',
    'Maroc': DATA_RAW / 'WEO_Maroc_2000_2024.xls'
}

FILES_IFS = {
    'Reserves': DATA_RAW / 'dataset_2026-02-28T06_44_04.822830526Z_DEFAULT_INTEGRATION_IMF.STA_IL_13.0.1.csv',
    'ExchangeRate_1': DATA_RAW / 'dataset_2026-02-28T06_50_53.802810440Z_DEFAULT_INTEGRATION_IMF.STA_ER_4.0.1.csv',
    'ExchangeRate_2': DATA_RAW / 'dataset_2026-02-28T06_54_14.340132276Z_DEFAULT_INTEGRATION_IMF.STA_ER_4.0.1.csv'
}

EXPECTED_VARS_WEO = [
    'Gross domestic product, constant prices',
    'Gross national savings',
    'Inflation, average consumer prices',
    'General government gross debt',
    'Current account balance'
]

# ===========================================================================
# VÉRIFICATION WEO
# ===========================================================================

print("="*100)
print("ÉTAPE 1 : VÉRIFICATION FICHIERS WEO")
print("="*100)
print()

weo_results = {}

for name, filepath in FILES_WEO.items():
    print(f"\n{'='*100}")
    print(f"FICHIER WEO : {name}")
    print(f"{'='*100}\n")
    
    try:
        df = pd.read_csv(filepath, sep='\t', encoding='latin1')
        
        print(f"✅ Chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
        
        # Années
        year_cols = [col for col in df.columns if col.isdigit() and len(col) == 4]
        
        if year_cols:
            print(f"📅 Période : {year_cols[0]} - {year_cols[-1]} ({len(year_cols)} années)")
            
            if year_cols[0] == '2000' and year_cols[-1] == '2024' and len(year_cols) == 25:
                print(f"   ✅ Période correcte (25 années)")
            else:
                print(f"   ⚠️  Attendu: 2000-2024 (25 années)")
        
        # Pays
        if 'Country' in df.columns:
            countries = df['Country'].dropna().unique()
            print(f"\n🌍 Pays ({len(countries)}) :")
            for c in countries:
                print(f"   - {c}")
        
        # Variables
        found_count = 0
        if 'Subject Descriptor' in df.columns:
            vars_found = df['Subject Descriptor'].dropna().unique()
            print(f"\n📈 Variables ({len(vars_found)}) :")
            
            for v in EXPECTED_VARS_WEO:
                if v in vars_found:
                    print(f"   ✅ {v}")
                    found_count += 1
                else:
                    print(f"   ❌ MANQUANT : {v}")
            
            print(f"\n   Total : {found_count}/5 variables trouvées")
        
        weo_results[name] = 'OK' if year_cols and len(year_cols) == 25 and found_count == 5 else 'WARNING'
        
    except Exception as e:
        print(f"❌ ERREUR : {e}")
        weo_results[name] = 'ERROR'

# ===========================================================================
# VÉRIFICATION IFS
# ===========================================================================

print("\n" + "="*100)
print("ÉTAPE 2 : VÉRIFICATION FICHIERS IFS")
print("="*100)
print()

ifs_results = {}

for name, filepath in FILES_IFS.items():
    print(f"\n{'='*100}")
    print(f"FICHIER IFS : {name}")
    print(f"{'='*100}\n")
    
    try:
        df = pd.read_csv(filepath)
        
        print(f"✅ Chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
        
        # Années
        year_cols = [col for col in df.columns if col.isdigit() and len(col) == 4]
        
        if year_cols:
            print(f"📅 Période : {year_cols[0]} - {year_cols[-1]} ({len(year_cols)} années)")
        
        # Pays
        if 'COUNTRY' in df.columns:
            countries = df['COUNTRY'].unique()
            print(f"\n🌍 Pays/Zones ({len(countries)}) :")
            for c in countries:
                print(f"   - {c}")
        
        # Indicateur
        if 'INDICATOR' in df.columns:
            indicators = df['INDICATOR'].unique()
            print(f"\n📊 Indicateurs ({len(indicators)}) :")
            for ind in indicators:
                print(f"   - {ind}")
        
        ifs_results[name] = 'OK'
        
    except Exception as e:
        print(f"❌ ERREUR : {e}")
        ifs_results[name] = 'ERROR'

# ===========================================================================
# RAPPORT FINAL
# ===========================================================================

print("\n" + "="*100)
print(" " * 35 + "RAPPORT FINAL")
print("="*100)
print()

print("📊 STATUT DES FICHIERS :\n")

print("WEO :")
for name, status in weo_results.items():
    emoji = '✅' if status == 'OK' else '⚠️' if status == 'WARNING' else '❌'
    print(f"   {emoji} {name:<20} : {status}")

print("\nIFS :")
for name, status in ifs_results.items():
    emoji = '✅' if status == 'OK' else '⚠️' if status == 'WARNING' else '❌'
    print(f"   {emoji} {name:<20} : {status}")

print()

all_ok = all(s in ['OK', 'WARNING'] for s in list(weo_results.values()) + list(ifs_results.values()))
no_errors = all(s != 'ERROR' for s in list(weo_results.values()) + list(ifs_results.values()))

if all_ok and no_errors:
    print("="*100)
    print("✅ TOUS LES FICHIERS SONT CHARGÉS")
    print("🚀 Prêt pour HARMONISATION")
    print("="*100)
    print()
    print("📋 DATASET ATTENDU :")
    print("   Pays        : 7 (France, Allemagne, Ghana, Maroc, RDC, Angola, Nigeria)")
    print("   Période     : 2000-2024 (25 ans)")
    print("   Observations: 7 × 25 = 175")
    print("   Variables   : 7 (GDP_Growth, Capital_Formation, Inflation, Government_Debt,")
    print("                    Trade_Balance, Exchange_Rate, Reserves)")
else:
    print("="*100)
    print("⚠️  CERTAINS FICHIERS ONT DES PROBLÈMES")
    print("Vérifie les détails ci-dessus")
    print("="*100)

print()
