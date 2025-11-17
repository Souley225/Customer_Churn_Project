# Guide de déploiement sur Render

Ce guide explique comment déployer l'API FastAPI et l'interface Streamlit sur Render en utilisant les artefacts locaux (sans MLflow distant).

## Prérequis

- Compte Render: https://render.com/
- Repository GitHub avec le code poussé
- Les artefacts sont déjà committés dans `data/processed/`

## Architecture de déploiement

Le projet déploie **2 services web** :

1. **API FastAPI** (`telco-churn-api`) : Service de prédiction via API REST
2. **UI Streamlit** (`telco-churn-ui`) : Interface utilisateur interactive

Les deux services utilisent les **artefacts locaux** committés dans le repository :
- `data/processed/model.joblib` : Modèle entraîné
- `data/processed/preprocessor.joblib` : Pipeline de preprocessing
- `data/processed/cleaner.joblib` : Transformateur de nettoyage des données

## Méthode 1 : Déploiement via render.yaml (Recommandé)

### Étape 1 : Pousser le code sur GitHub

```bash
git add .
git commit -m "Configuration déploiement Render avec artefacts locaux"
git push origin main
```

### Étape 2 : Créer un Blueprint sur Render

1. Connectez-vous sur https://dashboard.render.com/
2. Cliquez sur **New** → **Blueprint**
3. Connectez votre repository GitHub
4. Render détectera automatiquement le fichier `render.yaml`
5. Cliquez sur **Apply** pour créer les deux services

### Étape 3 : Vérifier le déploiement

Une fois déployés, vous recevrez deux URLs :
- API FastAPI : `https://telco-churn-api.onrender.com`
- UI Streamlit : `https://telco-churn-ui.onrender.com`

Vérifiez que tout fonctionne :
- API : Accédez à `https://telco-churn-api.onrender.com/docs` (Swagger UI)
- UI : Accédez à l'URL de l'interface Streamlit

## Méthode 2 : Déploiement manuel

### Pour l'API FastAPI

1. Sur Render : **New** → **Web Service**
2. Connectez votre repository GitHub
3. Configuration :
   - **Name** : `telco-churn-api`
   - **Region** : Frankfurt (ou autre)
   - **Branch** : `main`
   - **Root Directory** : (laisser vide)
   - **Runtime** : `Python 3`
   - **Build Command** : `pip install -r requirements.txt`
   - **Start Command** : `uvicorn src.serving.api:app --host 0.0.0.0 --port $PORT`
4. Variables d'environnement :
   - `USE_LOCAL_ARTIFACTS` = `true`
   - `PYTHON_VERSION` = `3.11.0`
5. Cliquez sur **Create Web Service**

### Pour l'UI Streamlit

1. Sur Render : **New** → **Web Service**
2. Connectez votre repository GitHub
3. Configuration :
   - **Name** : `telco-churn-ui`
   - **Region** : Frankfurt (ou autre)
   - **Branch** : `main`
   - **Root Directory** : (laisser vide)
   - **Runtime** : `Python 3`
   - **Build Command** : `pip install -r requirements.txt`
   - **Start Command** : `streamlit run src/ui/app.py --server.port=$PORT --server.address=0.0.0.0`
4. Variables d'environnement :
   - `USE_LOCAL_ARTIFACTS` = `true`
   - `PYTHON_VERSION` = `3.11.0`
5. Cliquez sur **Create Web Service**

## Vérification du bon fonctionnement

### API FastAPI

1. Accédez à `https://<votre-api>.onrender.com/docs`
2. Testez l'endpoint `/predict` avec cet exemple :

```json
[
  {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 50.0,
    "TotalCharges": "600.0"
  }
]
```

Vous devriez recevoir une liste de probabilités de churn.

### UI Streamlit

1. Accédez à `https://<votre-ui>.onrender.com`
2. Vérifiez que l'interface se charge correctement
3. Dans la sidebar, vous devriez voir : **"Modèle chargé depuis artefacts locaux"**
4. Testez une prédiction en modifiant les paramètres

## Logs et monitoring

### Consulter les logs

Pour chaque service :
1. Allez dans le dashboard Render
2. Sélectionnez le service
3. Cliquez sur **Logs**

Au démarrage, vous devriez voir :
```
✓ Modèle chargé depuis artefacts locaux: data/processed/model.joblib
✓ Preprocessor et cleaner chargés depuis data/processed
```

### En cas d'erreur

Si vous voyez :
```
Modèle local non trouvé: data/processed/model.joblib
```

Vérifiez que :
1. Les fichiers `.joblib` sont bien committés dans git
2. Le `.gitignore` contient bien : `!data/processed/*.joblib`
3. Les fichiers sont bien présents : `git ls-files data/processed/`

## Plan gratuit Render

⚠️ **Important** : Le plan gratuit Render a des limitations :
- Les services s'endorment après 15 minutes d'inactivité
- Premier démarrage peut prendre 30-60 secondes
- 750 heures/mois gratuites par service

Pour des performances optimales, envisagez le plan payant.

## Configuration avancée

### Activer MLflow distant (optionnel)

Si plus tard vous voulez utiliser un registry MLflow distant :

1. Déployez un serveur MLflow (ex: sur Render, Heroku, AWS)
2. Dans les variables d'environnement Render :
   - Changez `USE_LOCAL_ARTIFACTS` = `false`
   - Ajoutez `MLFLOW_TRACKING_URI` = `https://votre-mlflow.com`
   - Ajoutez `MODEL_URI` = `models:/telco-churn-classifier/Production`
3. Redéployez les services

Le code basculera automatiquement en mode MLflow avec fallback local en cas d'échec.

## Support

Pour toute question sur le déploiement :
- Documentation Render : https://render.com/docs
- Issues GitHub du projet : https://github.com/Souley225/Customer_Churn_Project/issues
