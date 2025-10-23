## src/models/predict.py
"""Prédiction batch à partir d'un modèle MLflow.

- Utilise un modèle chargé depuis registry ou runs
"""
from __future__ import annotations
import os
import pandas as pd
import joblib
import mlflow
from src.features.build_features import TelcoCleaner
from src.utils.paths import PROCESSED_DIR
# Charge le preprocessor
preprocessor = joblib.load(PROCESSED_DIR / "preprocessor.joblib")
cleaner = TelcoCleaner()


def predict_csv(input_csv: str, model_uri: str, output_csv: str) -> None:
    model = mlflow.sklearn.load_model(model_uri)
    df = pd.read_csv(input_csv)
    # Nettoyage + features dérivées
    df = cleaner.transform(df)
    # Préprocessing
    df = preprocessor.transform(df)
    proba = model.predict_proba(df)[:, 1]
    out = df.copy()
    out["churn_proba"] = proba
    out.to_csv(output_csv, index=False)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument("--model_uri", required=True)
    p.add_argument("--output_csv", required=True)
    args = p.parse_args()
    predict_csv(args.input_csv, args.model_uri, args.output_csv)
