# Construction d'un score d'attrition: Telco Customer Churn

![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![MLflow](https://img.shields.io/badge/MLflow-tracking-orange.svg)
![Docker](https://img.shields.io/badge/docker-compose-2496ED.svg)
![DVC](https://img.shields.io/badge/DVC-pipeline-945DD6.svg)

> **Ce projet implémente un pipeline complet de Machine Learning pour la prédiction du churn client sur le dataset Telco Customer Churn avec une architecture MLOps moderne.**

---

## 📊 Présentation générale

Ce projet utilise le dataset public **Telco Customer Churn** de Kaggle pour construire, suivre, versionner et déployer un modèle de classification de manière reproductible.

### 🛠️ Stack Technologique

| Outil | Usage |
|-------|-------|
| 🔧 **Hydra** | Gestion des configurations |
| 📦 **DVC** | Versioning des données et reproductibilité |
| 📈 **MLflow** | Suivi des expériences et registre de modèles |
| 🎯 **Optuna** | Optimisation des hyperparamètres |
| 🚀 **FastAPI** | API de prédiction |
| 🎨 **Streamlit** | Interface utilisateur |
| 🐳 **Docker Compose** | Déploiement conteneurisé |
| ⚙️ **GitHub Actions** | CI/CD |

---

## 📋 Informations principales

?> **Dataset**: `blastchar/telco-customer-churn` (Kaggle)

?> **Variable cible**: `Churn` (Yes/No)

- **Langage**: Python 3.11
- **Gestionnaire de paquets**: Poetry
- **Suivi d'expériences**: MLflow (local ou distant)
- **Version des données**: DVC
- **Optimisation**: Optuna

---

## 📁 Structure du projet

```
mlops-classification-project/
├── 📄 README.md
├── 📄 pyproject.toml
├── 📄 dvc.yaml
├── 📄 params.yaml
├── 📄 compose.yaml
├── 🐳 docker/
│   ├── Dockerfile.train
│   └── Dockerfile.app
├── 📂 src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── serving/
│   ├── ui/
│   └── utils/
├── ⚙️ configs/
│   ├── data.yaml
│   ├── model.yaml
│   └── train.yaml
├── 🧪 tests/
└── 🔄 .github/workflows/
```

---

## 💻 Installation locale

### Prérequis

!> **Attention**: Assurez-vous d'avoir tous les outils suivants installés avant de commencer.

- ✅ Python 3.11 ou plus (recommandé: pyenv)
- ✅ Poetry
- ✅ Git
- ✅ DVC
- ✅ Docker et Docker Compose
- ✅ Un compte Kaggle avec le fichier `~/.kaggle/kaggle.json`

### 🚀 Étapes d'installation

#### 1️⃣ Installation de Python

```bash
# Installation de Python
pyenv install 3.11.9
pyenv local 3.11.9
```

#### 2️⃣ Installation de Poetry

```bash
# Installation de Poetry
curl -sSL https://install.python-poetry.org | python3 -
poetry --version
```

#### 3️⃣ Clonage du projet

```bash
# Clonage du projet
git clone https://github.com/Souley225/Customer_Churn_Project
cd Customer_Churn_Project

# Installation des dépendances
poetry install
```

#### 4️⃣ Initialisation de DVC

```bash
# Initialisation de DVC
dvc init

# Vérification des hooks
poetry run pre-commit install
```

### 🔑 Configuration de l'API Kaggle

Créez un fichier `~/.kaggle/kaggle.json` contenant vos identifiants Kaggle:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

---

## 🔄 Exécution du pipeline complet

Le pipeline est entièrement automatisé via **DVC** et **Hydra**.

```bash
# Exécution complète du pipeline
dvc repro
```

### 📝 Étapes du pipeline

1. **📥 Téléchargement** des données depuis Kaggle
2. **✂️ Découpage** en ensembles d'entraînement, validation et test
3. **🔨 Feature Engineering** (tenure_bucket, num_services, total_spend_proxy, etc.)
4. **🎯 Entraînement** avec Optuna et suivi MLflow
5. **📊 Évaluation** du modèle sur le jeu de test
6. **💾 Enregistrement** du meilleur modèle dans le registre MLflow

---

## 🐳 Lancement des services avec Docker Compose

```bash
# Construction des images Docker
docker compose build

# Lancement de l'infrastructure
make up
```

### 🌐 Services accessibles

| Service | URL | Description |
|---------|-----|-------------|
| 📈 **MLflow UI** | [http://localhost:5000](http://localhost:5000) | Interface de suivi des expériences |
| 🚀 **API FastAPI** | [http://localhost:8000/docs](http://localhost:8000/docs) | Documentation interactive de l'API |
| 🎨 **Streamlit** | [http://localhost:8501](http://localhost:8501) | Interface utilisateur graphique |

```bash
# Pour arrêter les services
make down
```

---

## 🔍 Description fonctionnelle du pipeline

### 📥 Téléchargement des données
Via l'API Kaggle, récupération automatique du dataset.

### 🧹 Prétraitement
- Nettoyage des valeurs manquantes
- Encodage catégoriel
- Création de nouvelles variables:
  - `tenure_bucket`: Segmentation de l'ancienneté
  - `num_services`: Nombre de services souscrits
  - `total_spend_proxy`: Estimation des dépenses totales

### 🎯 Entraînement
Sélection automatique du modèle optimal parmi:
- 🌳 LightGBM
- 🚀 XGBoost
- 🐈 CatBoost
- 📉 Régression Logistique

Optimisation via **Optuna** pour trouver les meilleurs hyperparamètres.

### 📊 Suivi
Tous les paramètres, métriques et artefacts sont enregistrés dans **MLflow**.

### ✅ Évaluation
Test du modèle sur un échantillon indépendant avec métriques détaillées.

### 💾 Enregistrement
Le meilleur modèle est promu dans le registre **MLflow**.

---

## 🧪 Tests et validation

```bash
# Lancer les tests unitaires
poetry run pytest -q

# Vérifier la conformité du code
poetry run ruff .
poetry run black --check .
poetry run mypy src
```

---

## ☁️ Déploiement cloud (optionnel)

Pour connecter **MLflow** à un backend distant (S3, GCS ou Azure):

Définissez les variables d'environnement dans un fichier `.env`:

```env
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_S3_ENDPOINT_URL=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

Relancez le projet:

```bash
docker compose up
```

---

## 🖥️ Utilisation des interfaces

### 🚀 API FastAPI

**Endpoint principal**: `/predict`

Permet de soumettre un ou plusieurs enregistrements JSON pour obtenir une probabilité de churn.

**Exemple de requête**:

```json
{
  "SeniorCitizen": 0,
  "tenure": 12,
  "MonthlyCharges": 50.5,
  "TotalCharges": 606.0,
  "Contract": "Month-to-month",
  "PaymentMethod": "Electronic check"
}
```

### 🎨 Application Streamlit

Interface graphique intuitive pour:
- ✏️ Tester des entrées manuelles
- 📊 Observer la probabilité de churn prédite
- 📈 Visualiser les facteurs de risque

---

## 🔄 Intégration continue (CI/CD)

Les workflows **GitHub Actions** assurent:

- ✅ Analyse statique du code (black, ruff, mypy, isort)
- ✅ Exécution automatique des tests unitaires
- ✅ Construction et publication des images Docker
- ✅ Validation de la reproductibilité du pipeline

---

## 📝 Commandes utiles

| Commande | Description |
|----------|-------------|
| `dvc repro` | Lancer le pipeline complet |
| `make up` | Démarrer les services Docker |
| `make down` | Arrêter les services Docker |
| `make lint` | Vérifier la qualité du code |
| `make test` | Exécuter les tests unitaires |
| `poetry run pytest` | Tests avec coverage |
| `poetry run mlflow ui` | Lancer l'interface MLflow |

---

## 📄 Licence

Le projet est distribué sous licence **MIT**.

---

## 🤝 Contribution

Les contributions sont les bienvenues! N'hésitez pas à:

1. 🍴 Fork le projet
2. 🌿 Créer une branche (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit vos changements (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push vers la branche (`git push origin feature/AmazingFeature`)
5. 🔃 Ouvrir une Pull Request

---

## 📧 Contact

**Projet maintenu par**: [Souley225](https://github.com/Souley225)

**Repository**: [Customer_Churn_Project](https://github.com/Souley225/Customer_Churn_Project)

---

<div align="center">
  <sub>Construit avec ❤️ en utilisant MLOps best practices</sub>
</div>
