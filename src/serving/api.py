## src/serving/api.py (FastAPI)
"""API FastAPI de scoring.

- Charge le modèle MLflow et le preprocessor
- Applique TelcoCleaner + preprocessing avant prédiction
- Expose /predict pour scoring unitaire ou batch
"""
from __future__ import annotations

import os

import joblib
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import nécessaire pour le dépickling de cleaner.joblib
from src.features.build_features import TelcoCleaner  # noqa: F401
from src.utils.paths import PROCESSED_DIR

app = FastAPI(title="Telco Churn API")

# Utiliser version Production par défaut, ou dernière version disponible
MODEL_URI = os.getenv(
    "MODEL_URI", os.getenv("MLFLOW_MODEL_URI", "models:/telco-churn-classifier/Production")
)
model = None
preprocessor = None
cleaner = None


class Record(BaseModel):
    """Schéma complet des features d'entrée (données brutes client)."""

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str | None = None


@app.on_event("startup")
def load_artifacts() -> None:
    """Charge le modèle, le preprocessor et le cleaner.

    Stratégie:
    - Si USE_LOCAL_ARTIFACTS=true : charge directement depuis PROCESSED_DIR
    - Sinon: essaie MLflow puis fallback vers PROCESSED_DIR
    """
    global model, preprocessor, cleaner

    use_local = os.getenv("USE_LOCAL_ARTIFACTS", "false").lower() == "true"

    # Chargement du modèle
    if use_local:
        # Mode local direct (pour déploiement sans MLflow)
        model_path = PROCESSED_DIR / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle local non trouvé: {model_path}")
        model = joblib.load(model_path)
        print(f"✓ Modèle chargé depuis artefacts locaux: {model_path}")
    else:
        # Mode MLflow avec fallback
        try:
            model = mlflow.sklearn.load_model(MODEL_URI)
            print(f"✓ Modèle chargé depuis MLflow: {MODEL_URI}")
        except Exception as e:
            print(f"⚠ Échec chargement MLflow ({MODEL_URI}): {e}")
            # Fallback: charger le modèle depuis PROCESSED_DIR
            model_path = PROCESSED_DIR / "model.joblib"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Modèle non trouvé ni dans MLflow ni dans {model_path}"
                ) from e
            model = joblib.load(model_path)
            print(f"✓ Modèle chargé depuis fallback: {model_path}")

    # Chargement du preprocessor et cleaner
    try:
        preprocessor_path = PROCESSED_DIR / "preprocessor.joblib"
        cleaner_path = PROCESSED_DIR / "cleaner.joblib"

        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor non trouvé: {preprocessor_path}")
        if not cleaner_path.exists():
            raise FileNotFoundError(f"Cleaner non trouvé: {cleaner_path}")

        preprocessor = joblib.load(preprocessor_path)
        cleaner = joblib.load(cleaner_path)
        print(f"✓ Preprocessor et cleaner chargés depuis {PROCESSED_DIR}")
    except Exception as e:
        raise RuntimeError(f"Erreur chargement artefacts: {e}") from e


@app.post("/predict")
def predict(items: list[Record]) -> list[float]:
    """Prédiction du risque de churn pour une liste de clients.

    Applique le pipeline complet: TelcoCleaner -> Preprocessor -> Modèle
    """
    if model is None or preprocessor is None or cleaner is None:
        raise HTTPException(status_code=500, detail="Artefacts non chargés")

    try:
        # Conversion en DataFrame
        df = pd.DataFrame([item.dict() for item in items])

        # Application du nettoyage
        df_clean = cleaner.transform(df)

        # Application du preprocessing
        x_transformed = preprocessor.transform(df_clean)

        # Prédiction
        proba = model.predict_proba(x_transformed)[:, 1].tolist()
        return proba
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur prédiction: {str(e)}") from e
