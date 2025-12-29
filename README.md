<<<<<<< HEAD
# Score d'Attrition Client — Telco Customer Churn

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Licence-MIT-green?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" alt="MLflow"/>
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/DVC-Pipeline-945DD6?style=for-the-badge&logo=dvc&logoColor=white" alt="DVC"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
</p>

---

## Aperçu

Pipeline complet de Machine Learning pour la prédiction du **churn client** (attrition) basé sur le dataset **Telco Customer Churn** de Kaggle. Ce projet implémente les meilleures pratiques **MLOps** : versioning des données, suivi des expériences, optimisation des hyperparamètres et déploiement conteneurisé.

**Objectif** : Identifier les clients à risque de résiliation afin d'anticiper les actions de rétention.

---

## Stack Technologique

| Composant | Outil | Description |
|-----------|-------|-------------|
| Configuration | **Hydra** | Gestion centralisée et modulaire des configurations YAML |
| Versioning des données | **DVC** | Reproductibilité et traçabilité des pipelines de données |
| Suivi des expériences | **MLflow** | Logging des métriques, artefacts et registre de modèles |
| Optimisation | **Optuna** | Recherche automatique des hyperparamètres optimaux |
| API de prédiction | **FastAPI** | Service REST haute performance avec documentation Swagger |
| Interface utilisateur | **Streamlit** | Application web interactive pour les prédictions |
| Conteneurisation | **Docker Compose** | Orchestration multi-services pour le déploiement |
| CI/CD | **GitHub Actions** | Automatisation des tests et du déploiement |

---

## Informations du Projet

| Élément | Valeur |
|---------|--------|
| Dataset | `blastchar/telco-customer-churn` (Kaggle) |
| Variable cible | `Churn` (Yes/No — binaire) |
| Langage | Python 3.11+ |
| Gestionnaire de dépendances | Poetry |
| Licence | MIT |

---

## Structure du Projet

```
Customer_Churn_Project/
├── README.md                    # Documentation principale
├── DEPLOYMENT.md                # Guide de déploiement Render
├── pyproject.toml               # Configuration Poetry et outils
├── dvc.yaml                     # Définition du pipeline DVC
├── compose.yaml                 # Configuration Docker Compose
├── Makefile                     # Commandes utilitaires
├── render.yaml                  # Configuration déploiement Render
│
├── configs/                     # Configurations Hydra
│   ├── data.yaml
│   ├── model.yaml
│   └── train.yaml
│
├── src/                         # Code source principal
│   ├── data/                    # Téléchargement et préparation des données
│   ├── features/                # Feature engineering
│   ├── models/                  # Entraînement, évaluation, enregistrement
│   ├── serving/                 # API FastAPI
│   ├── ui/                      # Application Streamlit
│   └── utils/                   # Utilitaires partagés
│
├── docker/                      # Dockerfiles
│   ├── Dockerfile.train
│   └── Dockerfile.app
│
├── data/                        # Données (gérées par DVC)
│   ├── raw/                     # Données brutes
│   ├── interim/                 # Données intermédiaires
│   └── processed/               # Données et artefacts prêts à l'emploi
│
├── artifacts/                   # Artefacts d'entraînement
├── mlruns/                      # Expériences MLflow
├── tests/                       # Tests unitaires
└── .github/workflows/           # Pipelines CI/CD
```

---

## Installation

### Prérequis

Avant de commencer, assurez-vous d'avoir installé :

- **Python 3.11** ou supérieur (via pyenv recommandé)
- **Poetry** pour la gestion des dépendances
- **Git** et **DVC** pour le versioning
- **Docker** et **Docker Compose** pour le déploiement local
- Un **compte Kaggle** avec le fichier d'authentification `~/.kaggle/kaggle.json`

### Étapes d'installation

**1. Cloner le repository**

```bash
git clone https://github.com/Souley225/Customer_Churn_Project.git
cd Customer_Churn_Project
```

**2. Installer les dépendances**

```bash
poetry install
```

**3. Configurer les hooks pre-commit**

```bash
poetry run pre-commit install
```

**4. Configurer l'API Kaggle**

Créez le fichier d'authentification Kaggle et sécurisez-le :

```bash
# Linux/macOS
chmod 600 ~/.kaggle/kaggle.json

# Windows (PowerShell)
# Créez le fichier manuellement dans %USERPROFILE%\.kaggle\kaggle.json
```

---

## Exécution du Pipeline

Le pipeline complet est orchestré par **DVC** et configuré via **Hydra**.

### Exécution complète

```bash
dvc repro
```

### Étapes du pipeline

| Étape | Description |
|-------|-------------|
| `download` | Téléchargement automatique des données depuis Kaggle |
| `split` | Découpage en ensembles train/validation/test (70/10/20) |
| `features` | Feature engineering et transformation des variables |
| `train` | Entraînement avec optimisation Optuna et logging MLflow |
| `evaluate` | Évaluation des métriques sur le jeu de test |
| `register` | Enregistrement du meilleur modèle dans le registre MLflow |

### Nouvelles variables créées

- **tenure_bucket** : Segmentation de l'ancienneté client
- **num_services** : Nombre total de services souscrits
- **total_spend_proxy** : Estimation des dépenses cumulées

---

## Modèles Utilisés

Le pipeline teste automatiquement plusieurs algorithmes et sélectionne le meilleur :

| Modèle | Description |
|--------|-------------|
| LightGBM | Gradient boosting optimisé pour la vitesse |
| XGBoost | Gradient boosting avec régularisation avancée |
| CatBoost | Gradient boosting optimisé pour les variables catégorielles |
| Régression Logistique | Modèle de référence interprétable |

L'optimisation des hyperparamètres est réalisée par **Optuna** avec validation croisée.

---

## Déploiement Local avec Docker

### Lancer les services

```bash
# Construire les images
docker compose build

# Démarrer l'infrastructure
make up
```

### Services disponibles

| Service | URL | Description |
|---------|-----|-------------|
| MLflow UI | http://localhost:5000 | Suivi des expériences et registre |
| API FastAPI | http://localhost:8000/docs | Documentation interactive Swagger |
| Streamlit | http://localhost:8501 | Interface utilisateur graphique |

### Arrêter les services

```bash
make down
```

---

## Utilisation de l'API

### Endpoint de prédiction

`POST /predict`

### Exemple de requête

```json
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
```

### Réponse

```json
{
  "churn_probability": 0.72,
  "prediction": "Yes"
}
```

---

## Interface Streamlit

L'application Streamlit permet de :

- Saisir manuellement les caractéristiques d'un client
- Obtenir instantanément la probabilité de churn
- Visualiser les facteurs de risque principaux
- Explorer l'interprétabilité du modèle via SHAP

---

## Tests et Qualité du Code

### Exécuter les tests

```bash
poetry run pytest -q
```

### Vérifier la conformité du code

```bash
# Linting et formatage
poetry run ruff .
poetry run black --check .
poetry run isort --check .

# Vérification des types
poetry run mypy src
```

---

## Intégration Continue

Les workflows **GitHub Actions** automatisent :

| Workflow | Description |
|----------|-------------|
| Linting | Analyse statique avec Black, Ruff, isort, MyPy |
| Tests | Exécution des tests unitaires avec Pytest |
| Build | Construction et publication des images Docker |
| Validation | Vérification de la reproductibilité du pipeline |

---

## Déploiement Cloud

Pour un déploiement sur **Render**, consultez le guide détaillé : [DEPLOYMENT.md](DEPLOYMENT.md)

Le projet est pré-configuré avec `render.yaml` pour un déploiement en un clic via Blueprint.

### Configuration MLflow distant (optionnel)

Pour connecter MLflow à un backend cloud, définissez les variables dans `.env` :

```env
MLFLOW_TRACKING_URI=https://votre-mlflow-server.com
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

---

## Commandes Utiles

| Commande | Description |
|----------|-------------|
| `dvc repro` | Exécuter le pipeline complet |
| `make up` | Démarrer les services Docker |
| `make down` | Arrêter les services Docker |
| `make lint` | Vérifier la qualité du code |
| `make test` | Exécuter les tests unitaires |
| `poetry run mlflow ui` | Lancer l'interface MLflow locale |

---

## Contribution

Les contributions sont bienvenues. Pour contribuer :

1. Forkez le repository
2. Créez une branche (`git checkout -b feature/NouvelleFeature`)
3. Committez vos modifications (`git commit -m 'Ajout de NouvelleFeature'`)
4. Poussez la branche (`git push origin feature/NouvelleFeature`)
5. Ouvrez une Pull Request

Assurez-vous que les tests passent et que le code respecte les standards de formatage avant de soumettre.

---

## Contact

**Auteur** : [Souley225](https://github.com/Souley225)

**Repository** : [Customer_Churn_Project](https://github.com/Souley225/Customer_Churn_Project)

---

## Licence

Ce projet est distribué sous licence **MIT**. Voir le fichier `LICENSE` pour plus de détails.

---

<p align="center">
  <sub>Projet MLOps de prédiction du churn client</sub>
</p>
=======

>>>>>>> 28b1bc9a8733cb411d535cd713bed8dce962e673
