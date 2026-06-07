#!/usr/bin/env python3
"""
HARMONISATION COMPLÈTE DES SOURCES DE DONNÉES - DATASET ÉTENDU

Dataset étendu :
- 7 pays : France, Allemagne, Ghana, Maroc, RDC, Angola, Nigeria
- Période : 2000-2024 (25 ans)
- Résultat : 7 × 25 = 175 observations

Sources :
- 3 fichiers WEO (Allemagne_France, 4Subsaharan, Maroc)
- 3 fichiers IFS (Reserves, ExchangeRate Europe, ExchangeRate Afrique)

Output : data/processed/dataset_harmonised_final.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print(" " * 25 + "HARMONISATION DES SOURCES DE DONNÉES - DATASET ÉTENDU")
print("="*100)
print()

# CONFIGURATION DES CHEMINS


DATA_RAW = Path('../data/raw')
DATA_PROCESSED = Path('../data/processed')

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# Noms de fichiers sources
FILE_WEO_EU = DATA_RAW / 'WEO_Allemagne_France_2000_2024.xls'
FILE_WEO_SUBSAHARAN = DATA_RAW / 'WEO_4Subsaharan_2000_2024.xls'
FILE_WEO_MAROC = DATA_RAW / 'WEO_Maroc_2000_2024.xls'
FILE_IFS_RESERVES = DATA_RAW / 'dataset_2026-02-28T06_44_04.822830526Z_DEFAULT_INTEGRATION_IMF.STA_IL_13.0.1.csv'
FILE_IFS_ER_AFRICA = DATA_RAW / 'dataset_2026-02-28T06_50_53.802810440Z_DEFAULT_INTEGRATION_IMF.STA_ER_4.0.1.csv'
FILE_IFS_ER_EURO = DATA_RAW / 'dataset_2026-02-28T06_54_14.340132276Z_DEFAULT_INTEGRATION_IMF.STA_ER_4.0.1.csv'

print("Configuration des chemins :")
print(f"   Sources : {DATA_RAW}")
print(f"   Output  : {DATA_PROCESSED}")
print()

# Mapping des codes WEO vers noms de variables
WEO_CODE_MAPPING = {
    'NGDP_RPCH': 'GDP_Growth',
    'NGSD_NGDP': 'Capital_Formation',
    'PCPIPCH': 'Inflation',
    'GGXWDG_NGDP': 'Government_Debt',
    'BCA_NGDPD': 'Trade_Balance'
}

# Mapping des noms de pays (harmonisation)
COUNTRY_MAPPING = {
    'France': 'France',
    'Germany': 'Germany',
    'Angola': 'Angola',
    'Democratic Republic of the Congo': 'DRC',
    'Congo, Democratic Republic of the': 'DRC',
    'Ghana': 'Ghana',
    'Nigeria': 'Nigeria',
    'Morocco': 'Morocco',
    'Euro Area (EA)': 'Euro Area'  # Temporaire, on séparera après
}

# FONCTION POUR CHARGER ET TRANSFORMER WEO


def load_weo(filepath, file_label):
    """Charge et transforme un fichier WEO en format long"""
    
    print(f"\n{'='*100}")
    print(f"CHARGEMENT WEO : {file_label}")
    print(f"{'='*100}\n")
    
    df = pd.read_csv(filepath, sep='\t', encoding='latin1')
    
    # Identifier colonnes d'années
    year_cols = [col for col in df.columns if col.isdigit() and len(col) == 4]
    
    # Transformer en format long
    df_long = pd.melt(
        df,
        id_vars=['Country', 'WEO Subject Code'],
        value_vars=year_cols,
        var_name='Year',
        value_name='Value'
    )
    
    df_long['Year'] = df_long['Year'].astype(int)
    df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')
    
    # Mapper les codes WEO vers noms de variables
    df_long['Variable'] = df_long['WEO Subject Code'].map(WEO_CODE_MAPPING)
    
    # Filtrer seulement les variables mappées
    df_long = df_long.dropna(subset=['Variable'])
    
    # Harmoniser noms de pays
    df_long['Country'] = df_long['Country'].map(COUNTRY_MAPPING)
    
    df_clean = df_long[['Country', 'Year', 'Variable', 'Value']].copy()
    
    print(f" Chargé : {len(df_clean)} observations")
    print(f"   Pays : {df_clean['Country'].unique().tolist()}")
    print(f"   Variables : {df_clean['Variable'].unique().tolist()}")
    print()
    
    return df_clean


# ÉTAPE 1 : CHARGER WEO

print("="*100)
print("ÉTAPE 1 : CHARGEMENT FICHIERS WEO")
print("="*100)

weo_eu = load_weo(FILE_WEO_EU, "Europe (France + Allemagne)")
weo_subsaharan = load_weo(FILE_WEO_SUBSAHARAN, "4 Pays Subsahariens")
weo_maroc = load_weo(FILE_WEO_MAROC, "Maroc")

# Concaténer tous les WEO
weo_combined = pd.concat([weo_eu, weo_subsaharan, weo_maroc], ignore_index=True)

print(f" WEO combiné : {len(weo_combined)} observations")
print(f"   Pays uniques : {sorted(weo_combined['Country'].unique())}")
print()


# ÉTAPE 2 : CHARGER IFS - EXCHANGE RATE

print("="*100)
print("ÉTAPE 2 : CHARGEMENT IFS - EXCHANGE RATE")
print("="*100)
print()

# Exchange Rate Afrique
print("Loading Exchange Rate Afrique...")
ifs_er_africa = pd.read_csv(FILE_IFS_ER_AFRICA)

year_cols_ifs = [col for col in ifs_er_africa.columns if col.isdigit() and len(col) == 4]

# Filtrer Period Average
ifs_er_africa_pa = ifs_er_africa[ifs_er_africa['INDICATOR'] == 'Domestic currency per US Dollar'].copy()

ifs_er_africa_long = pd.melt(
    ifs_er_africa_pa,
    id_vars=['COUNTRY'],
    value_vars=year_cols_ifs,
    var_name='Year',
    value_name='Value'
)

ifs_er_africa_long['Year'] = ifs_er_africa_long['Year'].astype(int)
ifs_er_africa_long['Value'] = pd.to_numeric(ifs_er_africa_long['Value'], errors='coerce')
ifs_er_africa_long['Variable'] = 'Exchange_Rate'
ifs_er_africa_long['Country'] = ifs_er_africa_long['COUNTRY'].map(COUNTRY_MAPPING)

ifs_er_africa_clean = ifs_er_africa_long[['Country', 'Year', 'Variable', 'Value']].copy()

print(f" Exchange Rate Afrique : {len(ifs_er_africa_clean)} observations")
print(f"   Pays : {ifs_er_africa_clean['Country'].unique().tolist()}")
print()

# Exchange Rate Euro Area
print("Loading Exchange Rate Euro Area...")
ifs_er_euro = pd.read_csv(FILE_IFS_ER_EURO)

ifs_er_euro_pa = ifs_er_euro[ifs_er_euro['INDICATOR'] == 'Domestic currency per US Dollar'].copy()

ifs_er_euro_long = pd.melt(
    ifs_er_euro_pa,
    id_vars=['COUNTRY'],
    value_vars=year_cols_ifs,
    var_name='Year',
    value_name='Value'
)

ifs_er_euro_long['Year'] = ifs_er_euro_long['Year'].astype(int)
ifs_er_euro_long['Value'] = pd.to_numeric(ifs_er_euro_long['Value'], errors='coerce')
ifs_er_euro_long['Variable'] = 'Exchange_Rate'
ifs_er_euro_long['Country'] = ifs_er_euro_long['COUNTRY'].map(COUNTRY_MAPPING)

ifs_er_euro_clean = ifs_er_euro_long[['Country', 'Year', 'Variable', 'Value']].copy()

print(f" Exchange Rate Euro : {len(ifs_er_euro_clean)} observations")
print()

# Dupliquer Euro Area pour France ET Allemagne
ifs_er_france = ifs_er_euro_clean.copy()
ifs_er_france['Country'] = 'France'

ifs_er_allemagne = ifs_er_euro_clean.copy()
ifs_er_allemagne['Country'] = 'Germany'

print(f" Exchange Rate dupliqué pour France et Germany")
print()


# ÉTAPE 3 : CHARGER IFS - RESERVES

print("="*100)
print("ÉTAPE 3 : CHARGEMENT IFS - RESERVES")
print("="*100)
print()

ifs_reserves = pd.read_csv(FILE_IFS_RESERVES)

ifs_reserves_filtered = ifs_reserves[
    ifs_reserves['INDICATOR'] == 'Reserves excluding gold, foreign exchange'
].copy()

ifs_reserves_long = pd.melt(
    ifs_reserves_filtered,
    id_vars=['COUNTRY'],
    value_vars=year_cols_ifs,
    var_name='Year',
    value_name='Value'
)

ifs_reserves_long['Year'] = ifs_reserves_long['Year'].astype(int)
ifs_reserves_long['Value'] = pd.to_numeric(ifs_reserves_long['Value'], errors='coerce')
ifs_reserves_long['Variable'] = 'Reserves'
ifs_reserves_long['Country'] = ifs_reserves_long['COUNTRY'].map(COUNTRY_MAPPING)

ifs_reserves_clean = ifs_reserves_long[['Country', 'Year', 'Variable', 'Value']].copy()

print(f" Reserves : {len(ifs_reserves_clean)} observations")
print(f"   Pays : {sorted(ifs_reserves_clean['Country'].unique())}")
print()


# ÉTAPE 4 : FUSION DE TOUTES LES SOURCES

print("="*100)
print("ÉTAPE 4 : FUSION DE TOUTES LES SOURCES")
print("="*100)
print()

# Concaténer tous les datasets
df_combined = pd.concat([
    weo_combined,
    ifs_er_africa_clean,
    ifs_er_france,
    ifs_er_allemagne,
    ifs_reserves_clean
], ignore_index=True)

# Nettoyer les NaN dans Country
df_combined = df_combined.dropna(subset=['Country'])

print(f" Dataset combiné : {len(df_combined)} observations")
print()

# Vérifier les pays
print(" Pays présents :")
for country in sorted(df_combined['Country'].unique()):
    count = (df_combined['Country'] == country).sum()
    print(f"   - {country}: {count} observations")
print()

# Vérifier les variables
print(" Variables présentes :")
for var in sorted(df_combined['Variable'].unique()):
    count = (df_combined['Variable'] == var).sum()
    print(f"   - {var}: {count} observations")
print()

# ÉTAPE 5 : PIVOTER EN FORMAT WIDE

print("="*100)
print("ÉTAPE 5 : TRANSFORMATION EN FORMAT WIDE")
print("="*100)
print()

df_wide = df_combined.pivot_table(
    index=['Country', 'Year'],
    columns='Variable',
    values='Value',
    aggfunc='first'
).reset_index()

print(f" Format wide créé : {df_wide.shape[0]} lignes × {df_wide.shape[1]} colonnes")
print()

# Réorganiser les colonnes
column_order = ['Country', 'Year', 'GDP_Growth', 'Capital_Formation', 'Inflation', 
                'Government_Debt', 'Trade_Balance', 'Exchange_Rate', 'Reserves']
df_wide = df_wide[column_order]


# ÉTAPE 6 : VÉRIFICATION DE QUALITÉ

print("="*100)
print("ÉTAPE 6 : VÉRIFICATION DE QUALITÉ")
print("="*100)
print()

print(" DONNÉES MANQUANTES PAR VARIABLE :")
for var in df_wide.columns[2:]:
    missing = df_wide[var].isna().sum()
    total = len(df_wide)
    pct = (missing / total) * 100
    status = " " if pct < 10 else " " if pct < 30 else " "
    print(f"   {status} {var:<25} : {missing:>3}/{total} ({pct:>5.1f}%)")
print()

print(" DONNÉES MANQUANTES PAR PAYS :")
for country in sorted(df_wide['Country'].unique()):
    country_data = df_wide[df_wide['Country'] == country]
    total_values = len(country_data) * 7
    missing_values = country_data.iloc[:, 2:].isna().sum().sum()
    pct = (missing_values / total_values) * 100
    print(f"   {country:<20} : {missing_values:>3}/{total_values} ({pct:>5.1f}%)")
print()

# ÉTAPE 7 : SAUVEGARDE

print("="*100)
print("ÉTAPE 7 : SAUVEGARDE DU DATASET HARMONISÉ")
print("="*100)
print()

# Sauvegarder en CSV
output_file = DATA_PROCESSED / 'dataset_harmonised_final.csv'
df_wide.to_csv(output_file, index=False)

print(f" Dataset harmonisé sauvegardé : {output_file}")
print(f"   Dimensions : {df_wide.shape[0]} observations × {df_wide.shape[1]} colonnes")
print()

# Sauvegarder aussi en format long
output_file_long = DATA_PROCESSED / 'dataset_harmonised_long.csv'
df_combined.to_csv(output_file_long, index=False)
print(f" Dataset long sauvegardé : {output_file_long}")
print()


# RAPPORT FINAL

print("="*100)
print(" " * 30 + "RAPPORT FINAL")
print("="*100)
print()

print(" RÉSUMÉ :")
print(f"   Pays                : {df_wide['Country'].nunique()}")
print(f"   Période             : {df_wide['Year'].min()}-{df_wide['Year'].max()}")
print(f"   Variables           : {len(df_wide.columns) - 2}")
print(f"   Observations totales: {len(df_wide)}")
print(f"   Complétude globale  : {((1 - df_wide.iloc[:, 2:].isna().sum().sum() / (len(df_wide) * 7)) * 100):.1f}%")
print()

print(" PAYS INCLUS :")
for country in sorted(df_wide['Country'].unique()):
    n_obs = (df_wide['Country'] == country).sum()
    print(f"   - {country:<20} : {n_obs} observations")
print()

print(" FICHIERS CRÉÉS :")
print(f"   1. {output_file} (format wide)")
print(f"   2. {output_file_long} (format long)")
print()

print("="*100)
print(" HARMONISATION TERMINÉE AVEC SUCCÈS !")
print("="*100)
print()

print(" PROCHAINES ÉTAPES :")
print("   1. Vérifier manuellement le fichier dataset_harmonised_final.csv")
print("   2. Lancer les analyses de Phase 1 (log-differencing + standardisation)")
print("   3. Lancer les analyses de Phase 2 (causal discovery)")
print("   4. Lancer les analyses de Phase 3 (DML)")
print()