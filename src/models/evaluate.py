## src/models/evaluate.py
"""Évaluation sur le test set.

- Charge meilleur modèle du dernier run (via MLflow run_id passé en env)
"""
from __future__ import annotations
import os
import numpy as np
import mlflow
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from src.utils.paths import PROCESSED_DIR
from src.utils.logging import logger
from pathlib import Path


def evaluate() -> None:
    # Si RUN_ID non fourni, prend le dernier run local
    run_id = os.getenv("RUN_ID")
    if run_id is None:
        from pathlib import Path
        meta_files = sorted(Path("mlruns/0").glob("*/meta.yaml"), reverse=True)
        if not meta_files:
            raise RuntimeError("Aucun run MLflow trouvé.")
        run_id = meta_files[0].parent.name

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    report = classification_report(y_test, preds, output_dict=False)

    logger.info(f"Test AUC: {auc:.4f} | AP: {ap:.4f}")
    print(report)
if __name__ == "__main__":
    evaluate()