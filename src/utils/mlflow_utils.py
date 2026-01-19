"""Utilitaires MLflow pour logguer et récupérer des modèles.


Commentaires en français.
"""

from __future__ import annotations

import os
from typing import Any

import mlflow


def setup_mlflow(experiment_name: str | None = None) -> None:
    """Configure l'expérience MLflow avec support S3 Supabase.

    Configure MLFLOW_S3_ENDPOINT_URL si défini pour utiliser
    Supabase Storage comme stockage d'artifacts S3-compatible.
    Respecte MLFLOW_TRACKING_URI si défini; sinon, stockage local.
    """
    # Configure S3 endpoint for Supabase Storage (if set)
    s3_endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    if s3_endpoint:
        # Ensure boto3 uses the Supabase S3 endpoint
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint

    # Set tracking URI (local mlruns or remote)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    # Set artifact location if configured
    artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT")
    if artifact_root:
        os.environ["MLFLOW_ARTIFACT_ROOT"] = artifact_root

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
                name=model_name, version=mv.version, description=f"Best {metric}={value:.5f}"
            )
            break
