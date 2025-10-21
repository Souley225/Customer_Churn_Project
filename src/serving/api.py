## src/serving/api.py (FastAPI)
"""API FastAPI de scoring.

- Charge le dernier modèle du registry (ou via MODEL_URI)
- Expose /predict pour scoring unitaire ou batch léger
"""
from __future__ import annotations
import os
from typing import List  
from typing import Any
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Telco Churn API")

MODEL_URI = os.getenv("MODEL_URI",
                       os.getenv("MLFLOW_MODEL_URI", "models:/telco-churn-classifier/None"))
model = None


class Record(BaseModel):
    # Définir un schéma minimal; en pratique on passerait tout l'enregistrement brut
    tenure: float
    MonthlyCharges: float
    TotalCharges: float | None = None
    Contract: str
    PaperlessBilling: int


@app.on_event("startup")
def load_model() -> None:
    global model
    model = mlflow.sklearn.load_model(MODEL_URI)


@app.post("/predict")
def predict(items: list[Record]) -> list[float]:
    import pandas as pd

    df = pd.DataFrame([i.dict() for i in items])
    proba = model.predict_proba(df)[:, 1].tolist()
    return proba
