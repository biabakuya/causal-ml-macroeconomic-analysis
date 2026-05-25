"""
Script de test de l'installation
Stage ABIL 2025 - Causal Machine Learning en Macroéconomie

Ce script teste que toutes les bibliothèques nécessaires sont bien installées
et fonctionnelles.

Exécution : python test_installation.py
"""

import sys
from importlib import import_module

print("="*80)
print("TEST DE L'INSTALLATION - STAGE ABIL 2025")
print("="*80)
print()

# Liste des packages à tester par catégorie
packages_to_test = {
    "📊 Manipulation de données": [
        ("pandas", "pd"),
        ("numpy", "np"),
    ],
    
    "📈 Visualisation": [
        ("matplotlib", "plt", "matplotlib.pyplot"),
        ("seaborn", "sns"),
        ("plotly", None),
    ],
    
    "📉 Statistiques": [
        ("scipy", None),
        ("statsmodels", "sm", "statsmodels.api"),
    ],
    
    "🤖 Machine Learning": [
        ("sklearn", None, "scikit-learn"),
        ("xgboost", "xgb"),
        ("lightgbm", "lgb"),
    ],
    
    "🔗 Causal ML - DoWhy": [
        ("dowhy", None),
    ],
    
    "🔗 Causal ML - EconML": [
        ("econml", None),
    ],
    
    "🔗 Causal ML - CausalML": [
        ("causalml", None),
    ],
    
    "🔗 Causal ML - TIGRAMITE": [
        ("tigramite", None),
    ],
    
    "📊 Séries temporelles": [
        ("statsmodels", None),
        ("pmdarima", None),
        ("arch", None),
    ],
    
    "🕸️  Graphes": [
        ("networkx", "nx"),
        ("pydot", None),
    ],
    
    "📓 Jupyter": [
        ("jupyter", None),
        ("IPython", None),
    ],
    
    "🛠️  Utilitaires": [
        ("tqdm", None),
        ("requests", None),
    ]
}

# Statistiques
total_packages = 0
success_count = 0
failed_packages = []
warnings_packages = []

# Tests
for category, packages in packages_to_test.items():
    print(f"\n{category}")
    print("-" * 80)
    
    for package_info in packages:
        total_packages += 1
        
        # Parsing des informations du package
        if len(package_info) == 2:
            module_name, alias = package_info
            display_name = module_name
        elif len(package_info) == 3:
            module_name, alias, display_name = package_info
        else:
            module_name = package_info[0]
            alias = None
            display_name = module_name
        
        # Test d'import
        try:
            if alias:
                exec(f"import {module_name} as {alias}")
            else:
                import_module(module_name)
            
            # Récupérer la version si possible
            try:
                module = import_module(module_name)
                version = getattr(module, '__version__', 'N/A')
                print(f"   ✅ {display_name:<30} (version {version})")
            except:
                print(f"   ✅ {display_name:<30}")
            
            success_count += 1
            
        except ImportError as e:
            # Packages optionnels
            if module_name in ['pygraphviz', 'graphviz']:
                print(f"   ⚠️  {display_name:<30} (optionnel, non installé)")
                warnings_packages.append(display_name)
            else:
                print(f"   ❌ {display_name:<30} ERREUR: {str(e)[:50]}")
                failed_packages.append(display_name)

# Test spécial pour Graphviz (outil système)
print(f"\n🖼️  Outils système")
print("-" * 80)

try:
    import subprocess
    result = subprocess.run(["dot", "-V"], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"   ✅ Graphviz (système)          {result.stderr.strip()}")
    else:
        print(f"   ⚠️  Graphviz (système)          Non détecté (optionnel)")
        warnings_packages.append("Graphviz système")
except:
    print(f"   ⚠️  Graphviz (système)          Non détecté (optionnel)")
    warnings_packages.append("Graphviz système")

# Test Python version
print(f"\n🐍 Version Python")
print("-" * 80)
print(f"   ✅ Python {sys.version.split()[0]}")

# Résumé
print("\n" + "="*80)
print("RÉSUMÉ DU TEST")
print("="*80)
print()

success_rate = (success_count / total_packages) * 100

print(f"📊 Packages testés : {total_packages}")
print(f"✅ Succès : {success_count} ({success_rate:.1f}%)")

if warnings_packages:
    print(f"⚠️  Avertissements : {len(warnings_packages)}")
    for pkg in warnings_packages:
        print(f"   - {pkg}")

if failed_packages:
    print(f"❌ Échecs : {len(failed_packages)}")
    for pkg in failed_packages:
        print(f"   - {pkg}")
    print()
    print("💡 Pour corriger les erreurs :")
    print("   pip install -r requirements.txt")
else:
    print()
    print("🎉 TOUS LES PACKAGES ESSENTIELS SONT INSTALLÉS !")

print()

# Test rapide de fonctionnalité
print("="*80)
print("TEST RAPIDE DE FONCTIONNALITÉ")
print("="*80)
print()

try:
    import pandas as pd
    import numpy as np
    
    # Créer un petit DataFrame
    df = pd.DataFrame({
        'Year': [2010, 2011, 2012],
        'Value': [100, 110, 105]
    })
    
    print("✅ Création DataFrame pandas : OK")
    print(f"   Shape: {df.shape}")
    
    # Calcul numpy
    arr = np.array([1, 2, 3, 4, 5])
    mean = np.mean(arr)
    
    print(f"✅ Calcul numpy : OK")
    print(f"   Moyenne de {arr} = {mean}")
    
except Exception as e:
    print(f"❌ Test de fonctionnalité échoué : {e}")

print()

# Évaluation finale
print("="*80)
if success_rate >= 90 and not failed_packages:
    print("✅ INSTALLATION RÉUSSIE - VOUS ÊTES PRÊT À TRAVAILLER ! 🚀")
elif success_rate >= 75:
    print("⚠️  INSTALLATION PARTIELLE - Quelques packages manquent")
    print("   Vous pouvez commencer, mais certaines fonctionnalités seront limitées")
else:
    print("❌ INSTALLATION INCOMPLÈTE - Veuillez réinstaller")
    print("   Exécutez : pip install -r requirements.txt")

print("="*80)
print()

# Instructions suivantes
if not failed_packages:
    print("🎯 PROCHAINES ÉTAPES :")
    print("   1. Placez vos données dans : data/raw/")
    print("   2. Consultez README_PHASE1.md")
    print("   3. Exécutez les scripts de la Phase 1")
    print()
