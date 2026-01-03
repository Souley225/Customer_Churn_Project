
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

<p align="center">
  <img src="https://img.shields.io/badge/Secteur-Telecom-764ABC?style=flat-square" alt="Secteur"/>
  <img src="https://img.shields.io/badge/Cas_d'usage-Retention_Client-FF6B6B?style=flat-square" alt="Cas d'usage"/>
  <img src="https://img.shields.io/badge/Impact-Reduction_du_Churn-28A745?style=flat-square" alt="Impact"/>
</p>

### Problematique Metier

> Les entreprises telecom perdent en moyenne **25% de leurs revenus** a cause de l'attrition client. Identifier les clients a risque **avant** leur depart permet de declencher des actions de retention ciblees.

### Ce que ce projet demontre

| Competence | Application Concrete |
|------------|---------------------|
| **Analyse de donnees** | Exploitation d'un dataset de 7 000+ clients pour identifier les facteurs de depart |
| **Modelisation predictive** | Creation d'un score de risque avec **precision de 80%+** |
| **Mise en production** | Application deployee et accessible en ligne, prete a etre utilisee |
| **Communication des resultats** | Interface intuitive permettant aux equipes metier d'exploiter les predictions |

### Valeur Apportee

<table>
<tr>
<td width="50%">

**Avant ce projet**
- Perte de clients non anticipee
- Actions de retention generiques et couteuses
- Decisions basees sur l'intuition

</td>
<td width="50%">

**Avec ce projet**
- Identification proactive des clients a risque
- Campagnes de retention personnalisees
- Decisions eclairees par les donnees

</td>
</tr>
</table>

### Competences Transversales Demontrees

<p align="center">
  <img src="https://img.shields.io/badge/Gestion_de_Projet-Livraison_complete-4A90A4?style=flat-square" alt="Gestion de projet"/>
  <img src="https://img.shields.io/badge/Autonomie-De_l'idee_au_deploiement-E67E22?style=flat-square" alt="Autonomie"/>
  <img src="https://img.shields.io/badge/Rigueur-Code_teste_et_documente-9B59B6?style=flat-square" alt="Rigueur"/>
  <img src="https://img.shields.io/badge/Pedagogie-Interface_utilisateur_intuitive-3498DB?style=flat-square" alt="Pedagogie"/>
</p>

---

## Demo en Ligne

<p align="center">
  <a href="https://customer-churn-project-2-zlzb.onrender.com/" target="_blank">
    <img src="https://img.shields.io/badge/Tester_la_Demo-En_Ligne-667eea?style=for-the-badge&logo=streamlit&logoColor=white" alt="Demo"/>
  </a>
</p>

| Element | Lien |
|---------|------|
| Application Streamlit | [https://customer-churn-project-2-zlzb.onrender.com/](https://customer-churn-project-2-zlzb.onrender.com/) |

> **Note** : L'application est hebergee sur Render (plan gratuit). Le premier chargement peut prendre quelques secondes si le service est en veille.

**Pour tester la demo :**
1. Cliquez sur le lien ci-dessus
2. Renseignez les caracteristiques d'un client (anciennete, charges, type de contrat)
3. Cliquez sur "Calculer le risque de churn" pour obtenir la prediction
4. Explorez l'onglet "Scoring par lot" pour tester avec un fichier CSV

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
