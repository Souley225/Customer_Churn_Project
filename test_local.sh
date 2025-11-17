#!/bin/bash
# Script de test local avant déploiement
# Vérifie que l'API et l'UI fonctionnent avec USE_LOCAL_ARTIFACTS=true

echo "=========================================="
echo "Test local avec artefacts locaux"
echo "=========================================="
echo ""

# Vérifier que les artefacts existent
echo "1. Vérification des artefacts..."
if [ ! -f "data/processed/model.joblib" ]; then
    echo "❌ Erreur: data/processed/model.joblib n'existe pas"
    exit 1
fi
if [ ! -f "data/processed/preprocessor.joblib" ]; then
    echo "❌ Erreur: data/processed/preprocessor.joblib n'existe pas"
    exit 1
fi
if [ ! -f "data/processed/cleaner.joblib" ]; then
    echo "❌ Erreur: data/processed/cleaner.joblib n'existe pas"
    exit 1
fi
echo "✓ Tous les artefacts sont présents"
echo ""

# Définir la variable d'environnement
export USE_LOCAL_ARTIFACTS=true

echo "2. Test de l'API FastAPI..."
echo "Démarrage de l'API sur http://localhost:8000"
echo "Accédez à http://localhost:8000/docs pour tester"
echo ""
echo "Pour tester manuellement:"
echo "  uvicorn src.serving.api:app --reload"
echo ""

echo "3. Test de l'UI Streamlit..."
echo "Démarrage de l'UI sur http://localhost:8501"
echo ""
echo "Pour tester manuellement:"
echo "  streamlit run src/ui/app.py"
echo ""

echo "=========================================="
echo "Configuration prête pour déploiement!"
echo "=========================================="
