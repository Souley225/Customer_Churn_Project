"""Utilitaires MLflow pour logguer et récupérer des modèles.


Commentaires en français.
"""
from __future__ import annotations
import os
import mlflow
from typing import Any




def setup_mlflow(experiment_name: str | None = None) -> None:
    """Configure l'expérience MLflow depuis variables d'environnement.
    Respecte MLFLOW_TRACKING_URI si défini; sinon, stockage local.
    """
    if experiment_name is None:
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "telco-churn")
        mlflow.set_experiment(experiment_name)




def log_params_dict(params: dict[str, Any]) -> None:
    """Log de paramètres sous forme de dict plat."""
    for k, v in params.items():
        mlflow.log_param(k, v)




def register_best(run_id: str, model_uri: str, model_name: str, metric: str, value: float) -> None:
    """Enregistre un modèle dans le registry avec une description.
    Args:
    run_id: identifiant de run MLflow
    model_uri: ex. f"runs:/{run_id}/model"
    model_name: nom dans le registry
    metric: métrique clé
    value: valeur de la métrique
    """
    mlflow.register_model(model_uri=model_uri, name=model_name)
    client = mlflow.tracking.MlflowClient()
    latest = client.get_latest_versions(model_name)
    for mv in latest:
        if mv.run_id == run_id:
            client.update_model_version(
            name=model_name,
            version=mv.version,
            description=f"Best {metric}={value:.5f}"
            )
            break