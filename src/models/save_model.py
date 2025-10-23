# save_model.py
import joblib, mlflow, os
from src.utils.paths import PROCESSED_DIR

# 1. Charger le modèle depuis MLflow (dernier run)
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(experiment_ids=["0"], order_by=["start_time DESC"], max_results=1)
run_id = runs[0].info.run_id
model_uri = f"runs:/{run_id}/model"
model = mlflow.sklearn.load_model(model_uri)

# 2. Sauvegarder au format joblib
os.makedirs("data/processed", exist_ok=True)
joblib.dump(model, PROCESSED_DIR / "model.joblib")
print("✅ model.joblib créé dans data/processed/")