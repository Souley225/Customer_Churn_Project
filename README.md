<<<<<<< HEAD
=======

>>>>>>> d88c2932073292078c069f7f886aca3db1e28fbc
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

## Apercu

Pipeline complet de Machine Learning pour la prediction du **churn client** (attrition) base sur le dataset **Telco Customer Churn** de Kaggle. Ce projet implemente les meilleures pratiques **MLOps** : versioning des donnees, suivi des experiences, optimisation des hyperparametres et deploiement conteneurise.

**Objectif** : Identifier les clients a risque de resiliation afin d'anticiper les actions de retention.

---

## Stack Technologique

| Composant | Outil | Description |
|-----------|-------|-------------|
| Configuration | **Hydra** | Gestion centralisee et modulaire des configurations YAML |
| Versioning des donnees | **DVC** | Reproductibilite et tracabilite des pipelines de donnees |
| Suivi des experiences | **MLflow** | Logging des metriques, artefacts et registre de modeles |
| Optimisation | **Optuna** | Recherche automatique des hyperparametres optimaux |
| API de prediction | **FastAPI** | Service REST haute performance avec documentation Swagger |
| Interface utilisateur | **Streamlit** | Application web interactive pour les predictions |
| Conteneurisation | **Docker Compose** | Orchestration multi-services pour le deploiement |
| CI/CD | **GitHub Actions** | Automatisation des tests et du deploiement |

---

## Informations du Projet

| Element | Valeur |
|---------|--------|
| Dataset | `blastchar/telco-customer-churn` (Kaggle) |
| Variable cible | `Churn` (Yes/No - binaire) |
| Langage | Python 3.11+ |
| Gestionnaire de dependances | Poetry |
| Licence | MIT |

---

## Structure du Projet

```
Customer_Churn_Project/
├── README.md                    # Documentation principale
├── DEPLOYMENT.md                # Guide de deploiement Render
├── pyproject.toml               # Configuration Poetry et outils
├── dvc.yaml                     # Definition du pipeline DVC
├── compose.yaml                 # Configuration Docker Compose
├── Makefile                     # Commandes utilitaires
├── render.yaml                  # Configuration deploiement Render
│
├── configs/                     # Configurations Hydra
│   ├── data.yaml
│   ├── model.yaml
│   └── train.yaml
│
├── src/                         # Code source principal
│   ├── data/                    # Telechargement et preparation des donnees
│   ├── features/                # Feature engineering
│   ├── models/                  # Entrainement, evaluation, enregistrement
│   ├── serving/                 # API FastAPI
│   ├── ui/                      # Application Streamlit
│   └── utils/                   # Utilitaires partages
│
├── docker/                      # Dockerfiles
│   ├── Dockerfile.train
│   └── Dockerfile.app
│
├── data/                        # Donnees (gerees par DVC)
│   ├── raw/                     # Donnees brutes
│   ├── interim/                 # Donnees intermediaires
│   └── processed/               # Donnees et artefacts prets a l'emploi
│
├── artifacts/                   # Artefacts d'entrainement
├── mlruns/                      # Experiences MLflow
├── tests/                       # Tests unitaires
└── .github/workflows/           # Pipelines CI/CD
```

---

## Installation

### Prerequis

Avant de commencer, assurez-vous d'avoir installe :

- **Python 3.11** ou superieur (via pyenv recommande)
- **Poetry** pour la gestion des dependances
- **Git** et **DVC** pour le versioning
- **Docker** et **Docker Compose** pour le deploiement local
- Un **compte Kaggle** avec le fichier d'authentification `~/.kaggle/kaggle.json`

### Etapes d'installation

**1. Cloner le repository**

```bash
git clone https://github.com/Souley225/Customer_Churn_Project.git
cd Customer_Churn_Project
```

**2. Installer les dependances**

```bash
poetry install
```

**3. Configurer les hooks pre-commit**

```bash
poetry run pre-commit install
```

**4. Configurer l'API Kaggle**

Creez le fichier d'authentification Kaggle et securisez-le :

```bash
# Linux/macOS
chmod 600 ~/.kaggle/kaggle.json

# Windows (PowerShell)
# Creez le fichier manuellement dans %USERPROFILE%\.kaggle\kaggle.json
```

---

## Execution du Pipeline

Le pipeline complet est orchestre par **DVC** et configure via **Hydra**.

### Execution complete

```bash
dvc repro
```

### Etapes du pipeline

| Etape | Description |
|-------|-------------|
| `download` | Telechargement automatique des donnees depuis Kaggle |
| `split` | Decoupage en ensembles train/validation/test (70/10/20) |
| `features` | Feature engineering et transformation des variables |
| `train` | Entrainement avec optimisation Optuna et logging MLflow |
| `evaluate` | Evaluation des metriques sur le jeu de test |
| `register` | Enregistrement du meilleur modele dans le registre MLflow |

### Nouvelles variables creees

- **tenure_bucket** : Segmentation de l'anciennete client
- **num_services** : Nombre total de services souscrits
- **total_spend_proxy** : Estimation des depenses cumulees

---

## Modeles Utilises

Le pipeline teste automatiquement plusieurs algorithmes et selectionne le meilleur :

| Modele | Description |
|--------|-------------|
| LightGBM | Gradient boosting optimise pour la vitesse |
| XGBoost | Gradient boosting avec regularisation avancee |
| CatBoost | Gradient boosting optimise pour les variables categorielles |
| Regression Logistique | Modele de reference interpretable |

L'optimisation des hyperparametres est realisee par **Optuna** avec validation croisee.

---

## Deploiement Local avec Docker

### Lancer les services

```bash
# Construire les images
docker compose build

# Demarrer l'infrastructure
make up
```

### Services disponibles

| Service | URL | Description |
|---------|-----|-------------|
| MLflow UI | http://localhost:5000 | Suivi des experiences et registre |
| API FastAPI | http://localhost:8000/docs | Documentation interactive Swagger |
| Streamlit | http://localhost:8501 | Interface utilisateur graphique |

### Arreter les services

```bash
make down
```

---

## Utilisation de l'API

### Endpoint de prediction

`POST /predict`

### Exemple de requete

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

### Reponse

```json
{
  "churn_probability": 0.72,
  "prediction": "Yes"
}
```

---

## Interface Streamlit

L'application Streamlit permet de :

- Saisir manuellement les caracteristiques d'un client
- Obtenir instantanement la probabilite de churn
- Visualiser les facteurs de risque principaux
- Explorer l'interpretabilite du modele via SHAP

---

## Tests et Qualite du Code

### Executer les tests

```bash
poetry run pytest -q
```

### Verifier la conformite du code

```bash
# Linting et formatage
poetry run ruff .
poetry run black --check .
poetry run isort --check .

# Verification des types
poetry run mypy src
```

---

## Integration Continue

Les workflows **GitHub Actions** automatisent :

| Workflow | Description |
|----------|-------------|
| Linting | Analyse statique avec Black, Ruff, isort, MyPy |
| Tests | Execution des tests unitaires avec Pytest |
| Build | Construction et publication des images Docker |
| Validation | Verification de la reproductibilite du pipeline |

---

## Deploiement Cloud

Pour un deploiement sur **Render**, consultez le guide detaille : [DEPLOYMENT.md](DEPLOYMENT.md)

Le projet est pre-configure avec `render.yaml` pour un deploiement en un clic via Blueprint.

### Configuration MLflow distant (optionnel)

Pour connecter MLflow a un backend cloud, definissez les variables dans `.env` :

```env
MLFLOW_TRACKING_URI=https://votre-mlflow-server.com
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

---

## Commandes Utiles

| Commande | Description |
|----------|-------------|
| `dvc repro` | Executer le pipeline complet |
| `make up` | Demarrer les services Docker |
| `make down` | Arreter les services Docker |
| `make lint` | Verifier la qualite du code |
| `make test` | Executer les tests unitaires |
| `poetry run mlflow ui` | Lancer l'interface MLflow locale |

---

## Contribution

Les contributions sont bienvenues. Pour contribuer :

1. Forkez le repository
2. Creez une branche (`git checkout -b feature/NouvelleFeature`)
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

Ce projet est distribue sous licence **MIT**. Voir le fichier `LICENSE` pour plus de details.

---

<p align="center">
  <sub>Projet MLOps de prediction du churn client</sub>
</p>
<<<<<<< HEAD
=======
=======


>>>>>>> d88c2932073292078c069f7f886aca3db1e28fbc
