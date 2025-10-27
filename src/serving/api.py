## src/serving/api.py (FastAPI)
"""API FastAPI de scoring.

- Charge le modèle MLflow et le preprocessor
- Applique TelcoCleaner + preprocessing avant prédiction
- Expose /predict pour scoring unitaire ou batch
"""
from __future__ import annotations
import os
import joblib
from pathlib import Path
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.features.build_features import TelcoCleaner
from src.utils.paths import PROCESSED_DIR

app = FastAPI(title="Telco Churn API")

# Utiliser version Production par défaut, ou dernière version disponible
MODEL_URI = os.getenv("MODEL_URI",
                       os.getenv("MLFLOW_MODEL_URI", "models:/telco-churn-classifier/Production"))
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
    """Charge le modèle MLflow, le preprocessor et le cleaner."""
    global model, preprocessor, cleaner
    try:
        model = mlflow.sklearn.load_model(MODEL_URI)
        preprocessor_path = PROCESSED_DIR / "preprocessor.joblib"
        cleaner_path = PROCESSED_DIR / "cleaner.joblib"

        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor non trouvé: {preprocessor_path}")
        if not cleaner_path.exists():
            raise FileNotFoundError(f"Cleaner non trouvé: {cleaner_path}")

        preprocessor = joblib.load(preprocessor_path)
        cleaner = joblib.load(cleaner_path)
    except Exception as e:
        raise RuntimeError(f"Erreur chargement artefacts: {e}")


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
        X = preprocessor.transform(df_clean)

        # Prédiction
        proba = model.predict_proba(X)[:, 1].tolist()
        return proba
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur prédiction: {str(e)}")
