"""API FastAPI de scoring.

- Charge le modele MLflow et le preprocessor
- Applique TelcoCleaner + preprocessing avant prediction
- Expose /predict pour scoring unitaire ou batch
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import joblib
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import necessaire pour le depickling de cleaner.joblib
from src.features.build_features import TelcoCleaner  # noqa: F401
from src.utils.paths import PROCESSED_DIR

# Utiliser version Production par defaut, ou derniere version disponible
MODEL_URI = os.getenv(
    "MODEL_URI", os.getenv("MLFLOW_MODEL_URI", "models:/telco-churn-classifier/Production")
)

# Variables globales pour les artefacts
model = None
preprocessor = None
cleaner = None


class Record(BaseModel):
    """Schema complet des features d'entree (donnees brutes client)."""

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


def _load_artifacts() -> None:
    """Charge le modele, le preprocessor et le cleaner.

    Strategie:
    - Si USE_LOCAL_ARTIFACTS=true : charge directement depuis PROCESSED_DIR
    - Sinon: essaie MLflow puis fallback vers PROCESSED_DIR
    """
    global model, preprocessor, cleaner

    use_local = os.getenv("USE_LOCAL_ARTIFACTS", "false").lower() == "true"

    # Chargement du modele
    if use_local:
        # Mode local direct (pour deploiement sans MLflow)
        model_path = PROCESSED_DIR / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Modele local non trouve: {model_path}")
        model = joblib.load(model_path)
        print(f"[OK] Modele charge depuis artefacts locaux: {model_path}")
    else:
        # Mode MLflow avec fallback
        try:
            model = mlflow.sklearn.load_model(MODEL_URI)
            print(f"[OK] Modele charge depuis MLflow: {MODEL_URI}")
        except Exception as e:
            print(f"[WARN] Echec chargement MLflow ({MODEL_URI}): {e}")
            # Fallback: charger le modele depuis PROCESSED_DIR
            model_path = PROCESSED_DIR / "model.joblib"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Modele non trouve ni dans MLflow ni dans {model_path}"
                ) from e
            model = joblib.load(model_path)
            print(f"[OK] Modele charge depuis fallback: {model_path}")

    # Chargement du preprocessor et cleaner
    try:
        preprocessor_path = PROCESSED_DIR / "preprocessor.joblib"
        cleaner_path = PROCESSED_DIR / "cleaner.joblib"

        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor non trouve: {preprocessor_path}")
        if not cleaner_path.exists():
            raise FileNotFoundError(f"Cleaner non trouve: {cleaner_path}")

        preprocessor = joblib.load(preprocessor_path)
        cleaner = joblib.load(cleaner_path)
        print(f"[OK] Preprocessor et cleaner charges depuis {PROCESSED_DIR}")
    except Exception as e:
        raise RuntimeError(f"Erreur chargement artefacts: {e}") from e


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Gestionnaire de cycle de vie de l'application."""
    _load_artifacts()
    yield


app = FastAPI(title="Telco Churn API", lifespan=lifespan)


@app.post("/predict")
def predict(items: list[Record]) -> list[float]:
    """Prediction du risque de churn pour une liste de clients.

    Applique le pipeline complet: TelcoCleaner -> Preprocessor -> Modele
    """
    if model is None or preprocessor is None or cleaner is None:
        raise HTTPException(status_code=500, detail="Artefacts non charges")

    try:
        # Conversion en DataFrame
        df = pd.DataFrame([item.model_dump() for item in items])

        # Application du nettoyage
        df_clean = cleaner.transform(df)

        # Application du preprocessing
        x_transformed = preprocessor.transform(df_clean)

        # Prediction
        proba = model.predict_proba(x_transformed)[:, 1].tolist()
        return proba
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur prediction: {str(e)}") from e
