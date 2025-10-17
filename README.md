# Projet MLOps de Classification : Telco Customer Churn

## 1. Présentation générale

Ce projet implémente un pipeline complet de Machine Learning pour la prédiction du **churn client** sur le dataset public **Telco Customer Churn** (Kaggle). L’objectif est de construire, suivre, versionner et déployer un modèle de classification de manière reproductible grâce à une architecture MLOps moderne.

Le projet repose sur les outils suivants :

* **Hydra** pour la gestion des configurations.
* **DVC** pour la version des données et la reproductibilité du pipeline.
* **MLflow** pour le suivi des expériences et la gestion du registre de modèles.
* **Optuna** pour l’optimisation automatique des hyperparamètres.
* **FastAPI** pour le service d’API de prédiction.
* **Streamlit** pour l’interface utilisateur.
* **Docker Compose** pour le déploiement.
* **GitHub Actions** pour l’intégration et le déploiement continus (CI/CD).

## 2. Informations principales

* **Dataset Kaggle** : `blastchar/telco-customer-churn`
* **Variable cible** : `Churn` (Yes/No)
* **Langage** : Python 3.11
* **Gestionnaire de paquets** : Poetry
* **Suivi d’expériences** : MLflow (local ou distant)
* **Version des données** : DVC
* **Optimisation** : Optuna

## 3. Structure du projet

```
mlops-classification-project/
├── README.md
├── pyproject.toml
├── dvc.yaml
├── params.yaml
├── compose.yaml
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.app
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── serving/
│   ├── ui/
│   └── utils/
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   └── train.yaml
├── tests/
└── .github/workflows/
```

## 4. Installation locale

### Prérequis

* Python 3.11 ou plus (recommandé : pyenv)
* Poetry
* Git
* DVC
* Docker et Docker Compose
* Un compte Kaggle avec le fichier `~/.kaggle/kaggle.json`

### Étapes d’installation

```bash
# Installation de Python
pyenv install 3.11.9
pyenv local 3.11.9

# Installation de Poetry
curl -sSL https://install.python-poetry.org | python3 -
poetry --version

# Clonage du projet
git clone <votre_repo>
cd mlops-classification-project

# Installation des dépendances
poetry install

# Initialisation de DVC
dvc init

# Vérification des hooks
poetry run pre-commit install
```

### Configuration de l’API Kaggle

Créez un fichier `~/.kaggle/kaggle.json` contenant vos identifiants Kaggle. Modifiez ses permissions :

```bash
chmod 600 ~/.kaggle/kaggle.json
```

## 5. Exécution du pipeline complet

Le pipeline est entièrement automatisé via DVC et Hydra.

```bash
# Exécution complète du pipeline
dvc repro
```

Ce pipeline exécute les étapes suivantes :

1. Téléchargement des données depuis Kaggle.
2. Découpage en ensembles d’entraînement, validation et test.
3. Ingénierie des variables (feature engineering spécifique au churn).
4. Entraînement avec Optuna et suivi MLflow.
5. Évaluation du modèle sur le jeu de test.
6. Enregistrement du meilleur modèle dans le registre MLflow.

## 6. Lancement des services avec Docker Compose

```bash
# Construction des images Docker
docker compose build

# Lancement de l’infrastructure
make up
```

Les services suivants seront accessibles :

* **MLflow UI** : [http://localhost:5000](http://localhost:5000)
* **API FastAPI** : [http://localhost:8000/docs](http://localhost:8000/docs)
* **Interface Streamlit** : [http://localhost:8501](http://localhost:8501)

Pour arrêter les services :

```bash
make down
```

## 7. Description fonctionnelle du pipeline

* **Téléchargement des données** : via l’API Kaggle.
* **Prétraitement** : nettoyage des valeurs manquantes, encodage catégoriel, création de nouvelles variables (`tenure_bucket`, `num_services`, `total_spend_proxy`, etc.).
* **Entraînement** : sélection automatique du modèle optimal (LightGBM, XGBoost, CatBoost, LogReg) avec recherche Optuna.
* **Suivi** : les paramètres, métriques et artefacts sont enregistrés dans MLflow.
* **Évaluation** : le modèle est testé sur un échantillon indépendant.
* **Enregistrement** : le meilleur modèle est promu dans le registre MLflow.

## 8. Tests et validation

```bash
# Lancer les tests unitaires
poetry run pytest -q

# Vérifier la conformité du code
poetry run ruff .
poetry run black --check .
poetry run mypy src
```

## 9. Déploiement cloud (optionnel)

Pour connecter MLflow à un backend distant (S3, GCS ou Azure) :

* Définissez les variables d’environnement dans un fichier `.env` :

```
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_S3_ENDPOINT_URL=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

* Relancez le projet avec `docker compose up`.

## 10. Utilisation des interfaces

### API FastAPI

Endpoint principal : `/predict` permet de soumettre un ou plusieurs enregistrements JSON pour obtenir une probabilité de churn.

### Application Streamlit

Interface graphique pour tester des entrées manuelles et observer la probabilité de churn prédite par le modèle.

## 11. Intégration continue (CI/CD)

Les workflows GitHub Actions assurent :

* L’analyse statique du code (black, ruff, mypy, isort).
* L’exécution automatique des tests unitaires.
* La construction et publication des images Docker.

## 12. Commandes utiles

```bash
# Lancer le pipeline complet
dvc repro

# Démarrer les services Docker
make up

# Vérifier la qualité du code
make lint

# Exécuter les tests unitaires
make test
```

## 13. Licence

Le projet est distribué sous licence MIT.
