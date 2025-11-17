## src/models/predict.py
"""Prédiction batch à partir d'un modèle MLflow.

- Utilise un modèle chargé depuis registry ou runs
"""
from __future__ import annotations

import joblib
import mlflow
import pandas as pd

# Import nécessaire pour le dépickling de cleaner.joblib
from src.features.build_features import TelcoCleaner  # noqa: F401
from src.utils.paths import PROCESSED_DIR

# Charge le preprocessor et cleaner
preprocessor = joblib.load(PROCESSED_DIR / "preprocessor.joblib")
cleaner = joblib.load(PROCESSED_DIR / "cleaner.joblib")


def predict_csv(input_csv: str, model_uri: str, output_csv: str) -> None:
    """Prédiction batch avec support artefacts locaux ou MLflow.

    Si USE_LOCAL_ARTIFACTS=true, charge directement depuis PROCESSED_DIR.
    Sinon essaie MLflow avec fallback local.
    """
    import os

    use_local = os.getenv("USE_LOCAL_ARTIFACTS", "false").lower() == "true"

    # Chargement du modèle
    if use_local:
        # Mode local direct
        model_path = PROCESSED_DIR / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle local non trouvé: {model_path}")
        model = joblib.load(model_path)
        print(f"✓ Modèle chargé depuis artefacts locaux: {model_path}")
    else:
        # Mode MLflow avec fallback
        try:
            model = mlflow.sklearn.load_model(model_uri)
            print(f"✓ Modèle chargé depuis MLflow: {model_uri}")
        except Exception as e:
            print(f"⚠ Échec chargement MLflow ({model_uri}): {e}")
            # Fallback: charger le modèle depuis PROCESSED_DIR
            model_path = PROCESSED_DIR / "model.joblib"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Modèle non trouvé ni dans MLflow ni dans {model_path}"
                ) from e
            model = joblib.load(model_path)
            print(f"✓ Modèle chargé depuis fallback: {model_path}")

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
