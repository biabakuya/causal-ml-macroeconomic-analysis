"""
Script d'installation automatique de l'environnement
Stage ABIL 2025 - Causal Machine Learning en Macroéconomie

Auteur: Jirince K. Biaba
Date: Décembre 2024

Ce script va :
1. Créer un environnement virtuel Python
2. Installer toutes les dépendances nécessaires
3. Tester l'installation
4. Créer les dossiers de travail
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

print("="*80)
print("CONFIGURATION DE L'ENVIRONNEMENT - STAGE ABIL 2025")
print("Causal Machine Learning en Macroéconomie")
print("="*80)
print()

# Détection du système d'exploitation
SYSTEM = platform.system()
print(f"🖥️  Système détecté : {SYSTEM}")
print()

# Chemins
PROJECT_ROOT = Path.cwd()
VENV_NAME = "venv_stage"
VENV_PATH = PROJECT_ROOT / VENV_NAME

# Activation selon OS
if SYSTEM == "Windows":
    ACTIVATE_CMD = str(VENV_PATH / "Scripts" / "activate.bat")
    PYTHON_VENV = str(VENV_PATH / "Scripts" / "python.exe")
    PIP_VENV = str(VENV_PATH / "Scripts" / "pip.exe")
else:  # Linux/Mac
    ACTIVATE_CMD = f"source {VENV_PATH}/bin/activate"
    PYTHON_VENV = str(VENV_PATH / "bin" / "python")
    PIP_VENV = str(VENV_PATH / "bin" / "pip")

print("📁 Répertoire du projet :", PROJECT_ROOT)
print("📦 Nom de l'environnement :", VENV_NAME)
print()

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 : VÉRIFICATION PYTHON
# ─────────────────────────────────────────────────────────────────────────────

print("="*80)
print("ÉTAPE 1 : VÉRIFICATION DE PYTHON")
print("="*80)
print()

try:
    python_version = sys.version.split()[0]
    print(f"✅ Python {python_version} détecté")
    
    major, minor = map(int, python_version.split('.')[:2])
    if major < 3 or (major == 3 and minor < 8):
        print("❌ ERREUR : Python 3.8+ requis")
        print(f"   Version actuelle : {python_version}")
        print("   Téléchargez Python depuis : https://www.python.org/downloads/")
        sys.exit(1)
    print()
except Exception as e:
    print(f"❌ Erreur lors de la vérification Python : {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 : CRÉATION DE L'ENVIRONNEMENT VIRTUEL
# ─────────────────────────────────────────────────────────────────────────────

print("="*80)
print("ÉTAPE 2 : CRÉATION DE L'ENVIRONNEMENT VIRTUEL")
print("="*80)
print()

if VENV_PATH.exists():
    response = input(f"⚠️  L'environnement '{VENV_NAME}' existe déjà. Recréer ? (o/n) : ")
    if response.lower() == 'o':
        print(f"🗑️  Suppression de l'ancien environnement...")
        import shutil
        shutil.rmtree(VENV_PATH)
    else:
        print("✅ Conservation de l'environnement existant")
        print()
        skip_venv_creation = True
else:
    skip_venv_creation = False

if not skip_venv_creation:
    print(f"📦 Création de l'environnement virtuel '{VENV_NAME}'...")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(VENV_PATH)], check=True)
        print("✅ Environnement virtuel créé avec succès")
        print()
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de la création de l'environnement : {e}")
        sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 : MISE À JOUR DE PIP
# ─────────────────────────────────────────────────────────────────────────────

print("="*80)
print("ÉTAPE 3 : MISE À JOUR DE PIP")
print("="*80)
print()

print("⬆️  Mise à jour de pip dans l'environnement virtuel...")
try:
    subprocess.run([PYTHON_VENV, "-m", "pip", "install", "--upgrade", "pip"], 
                   check=True, capture_output=True)
    print("✅ pip mis à jour")
    print()
except subprocess.CalledProcessError as e:
    print(f"⚠️  Avertissement : Impossible de mettre à jour pip : {e}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 4 : INSTALLATION DES DÉPENDANCES
# ─────────────────────────────────────────────────────────────────────────────

print("="*80)
print("ÉTAPE 4 : INSTALLATION DES DÉPENDANCES")
print("="*80)
print()

requirements_file = PROJECT_ROOT / "requirements.txt"

if not requirements_file.exists():
    print(f"❌ ERREUR : Fichier 'requirements.txt' introuvable")
    print(f"   Chemin recherché : {requirements_file}")
    sys.exit(1)

print(f"📦 Installation depuis : {requirements_file}")
print()
print("⏳ Cette étape peut prendre plusieurs minutes...")
print("   (Environ 5-10 minutes selon votre connexion Internet)")
print()

try:
    # Installation avec barre de progression
    result = subprocess.run(
        [PIP_VENV, "install", "-r", str(requirements_file)],
        capture_output=False,  # Afficher la progression
        text=True
    )
    
    if result.returncode == 0:
        print()
        print("✅ Toutes les dépendances installées avec succès")
        print()
    else:
        print()
        print("⚠️  Certaines dépendances ont échoué, mais continuons...")
        print()
        
except Exception as e:
    print(f"❌ Erreur lors de l'installation : {e}")
    print()
    print("💡 Essayez d'installer manuellement :")
    print(f"   {ACTIVATE_CMD}")
    print(f"   pip install -r requirements.txt")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 5 : VÉRIFICATION SPÉCIALE - GRAPHVIZ
# ─────────────────────────────────────────────────────────────────────────────

print("="*80)
print("ÉTAPE 5 : VÉRIFICATION GRAPHVIZ (pour visualisation DAG)")
print("="*80)
print()

print("🔍 Vérification de Graphviz...")
try:
    result = subprocess.run(["dot", "-V"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Graphviz installé :", result.stderr.strip())
    else:
        print("⚠️  Graphviz non détecté")
        print()
        print("📥 INSTALLATION RECOMMANDÉE (optionnel mais utile) :")
        if SYSTEM == "Windows":
            print("   1. Téléchargez depuis : https://graphviz.org/download/")
            print("   2. Installez le .msi")
            print("   3. Ajoutez au PATH : C:\\Program Files\\Graphviz\\bin")
        elif SYSTEM == "Linux":
            print("   sudo apt-get install graphviz")
        elif SYSTEM == "Darwin":  # Mac
            print("   brew install graphviz")
except FileNotFoundError:
    print("⚠️  Graphviz non installé (optionnel)")
    print("   Les graphes causaux seront créés avec une alternative (pydot)")

print()

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 6 : CRÉATION DE LA STRUCTURE DE DOSSIERS
# ─────────────────────────────────────────────────────────────────────────────

print("="*80)
print("ÉTAPE 6 : CRÉATION DE LA STRUCTURE DE DOSSIERS")
print("="*80)
print()

folders = [
    "data/raw",           # Données brutes
    "data/processed",     # Données nettoyées
    "data/results",       # Résultats des analyses
    "notebooks",          # Jupyter notebooks
    "scripts/phase1",     # Scripts Phase 1
    "scripts/phase2",     # Scripts Phase 2
    "scripts/phase3",     # Scripts Phase 3
    "scripts/phase4",     # Scripts Phase 4
    "scripts/phase5",     # Scripts Phase 5
    "outputs/figures",    # Graphiques
    "outputs/tables",     # Tableaux
    "outputs/reports",    # Rapports
    "models",             # Modèles sauvegardés
    "docs"                # Documentation
]

print("📁 Création des dossiers de travail...")
for folder in folders:
    folder_path = PROJECT_ROOT / folder
    folder_path.mkdir(parents=True, exist_ok=True)
    print(f"   ✅ {folder}")

print()
print("✅ Structure de dossiers créée")
print()

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 7 : CRÉATION DU .GITIGNORE
# ─────────────────────────────────────────────────────────────────────────────

print("="*80)
print("ÉTAPE 7 : CRÉATION DU .gitignore")
print("="*80)
print()

gitignore_content = """# Environnement virtuel
venv_stage/
env/
ENV/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Jupyter Notebook
.ipynb_checkpoints

# Données (ne pas versionner les données sensibles)
data/raw/*.csv
data/raw/*.xlsx
data/processed/*.csv

# Résultats (fichiers volumineux)
*.pkl
*.pickle
*.joblib

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Rapports temporaires
outputs/temp/
"""

gitignore_path = PROJECT_ROOT / ".gitignore"
with open(gitignore_path, 'w', encoding='utf-8') as f:
    f.write(gitignore_content)

print("✅ .gitignore créé")
print()

# ─────────────────────────────────────────────────────────────────────────────
# RÉSUMÉ FINAL
# ─────────────────────────────────────────────────────────────────────────────

print("="*80)
print("✅ INSTALLATION TERMINÉE AVEC SUCCÈS !")
print("="*80)
print()

print("📋 RÉSUMÉ :")
print(f"   ✅ Python {python_version}")
print(f"   ✅ Environnement virtuel : {VENV_NAME}")
print(f"   ✅ Dépendances installées")
print(f"   ✅ Structure de dossiers créée")
print()

print("="*80)
print("🚀 PROCHAINES ÉTAPES")
print("="*80)
print()

print("1️⃣  ACTIVER L'ENVIRONNEMENT :")
if SYSTEM == "Windows":
    print(f"   {VENV_NAME}\\Scripts\\activate")
else:
    print(f"   source {VENV_NAME}/bin/activate")
print()

print("2️⃣  TESTER L'INSTALLATION :")
print("   python test_installation.py")
print()

print("3️⃣  LANCER JUPYTER :")
print("   jupyter lab")
print()

print("4️⃣  COMMENCER LA PHASE 1 :")
print("   Placez vos données CSV dans : data/raw/")
print("   Puis exécutez les scripts de la Phase 1")
print()

print("="*80)
print("📚 DOCUMENTATION :")
print("   Consultez README_INSTALLATION.md pour plus de détails")
print("="*80)
print()

print("🎯 Bon travail ! 🚀")
