## src/models/register.py
"""Enregistrement du meilleur modèle dans le Model Registry MLflow.

- Utilise RUN_ID et met à jour la description avec la meilleure AUC.
"""
from __future__ import annotations
import os
import mlflow
from src.utils.mlflow_utils import register_best


def main() -> None:
    run_id = os.getenv("RUN_ID")
    if run_id is None:
        # auto-detecte le dernier run local
        from pathlib import Path
        meta_files = sorted(Path("mlruns/0").glob("*/meta.yaml"), reverse=True)
        if not meta_files:
            raise RuntimeError("Aucun run MLflow trouvé.")
        run_id = meta_files[0].parent.name

    model_name = os.getenv("MODEL_NAME", "telco-churn-classifier")
    register_best(run_id, f"runs:/{run_id}/model", model_name, "val_auc", 0.0)


if __name__ == "__main__":
    main()
