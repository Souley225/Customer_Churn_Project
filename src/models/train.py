
## src/models/train.py


"""Entraînement avec Optuna et MLflow (ROC-AUC comme métrique principale).

- Charge X/y depuis data/processed
- Essaie plusieurs modèles: LightGBM, XGBoost, CatBoost, LogReg
- Utilise Optuna pour affiner les hyperparamètres
- Log complet dans MLflow (params, metrics, model)
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import optuna
import mlflow
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from src.utils.paths import PROCESSED_DIR
from src.utils.logging import logger
from src.utils.mlflow_utils import setup_mlflow
from src.utils.paths import PROJECT_ROOT

# Types optionnels
try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None
try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None
try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None


def load_arrays():
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    X_val = np.load(PROCESSED_DIR / "X_val.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_val = np.load(PROCESSED_DIR / "y_val.npy")
    return X_train, X_val, y_train, y_val


def objective(trial: optuna.Trial) -> float:
    X_train, X_val, y_train, y_val = load_arrays()

    # Poids de classes pour déséquilibre
    classes = np.unique(y_train)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

    model_name = trial.suggest_categorical("model", [
        "lightgbm" if lgb else None,
        "xgboost" if xgb else None,
        "catboost" if CatBoostClassifier else None,
        "logreg",
    ])
    model_name = model_name or "logreg"

    if model_name == "lightgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128, step=8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
        }
        clf = lgb.LGBMClassifier(**params, class_weight=class_weight)
    elif model_name == "xgboost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "eval_metric": "auc",
        }
        clf = xgb.XGBClassifier(**params, scale_pos_weight=class_weight.get(1, 1.0))
    elif model_name == "catboost":
        params = {
            "iterations": trial.suggest_int("iterations", 300, 1500, step=100),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            "random_seed": 42,
            "verbose": False,
        }
        cw_val = [class_weight.get(0, 1.0), class_weight.get(1, 1.0)]
        clf = CatBoostClassifier(**params, class_weights=cw_val)
    else:
        # Baseline logistique
        C = trial.suggest_float("C", 1e-3, 10.0, log=True)
        clf = LogisticRegression(C=C, max_iter=2000, n_jobs=-1, class_weight=class_weight)

    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_val)[:, 1]
    preds = (proba >= 0.5).astype(int)
    auc = float(roc_auc_score(y_val, proba))
    f1 = float(f1_score(y_val, preds))
    ap = float(average_precision_score(y_val, proba))
    # Sauvegarde des métriques comme user attributes
    trial.set_user_attr("val_auc", auc)
    trial.set_user_attr("val_f1", f1)
    trial.set_user_attr("val_ap", ap)
    return auc


def main() -> None:
    setup_mlflow("telco-churn")
    n_trials = int(os.getenv("OPTUNA_TRIALS", "30"))
    study = optuna.create_study(direction="maximize", study_name="telco-churn")
    with mlflow.start_run() as run:
        logger.info("Démarrage optimisation Optuna…")
        study.optimize(objective, n_trials=n_trials)
        best_auc = study.best_value
        mlflow.log_metric("best_auc", best_auc)

        # Réentraîner le meilleur modèle sur train+val
        best_params = study.best_trial.params
        model_choice = best_params.get("model", "logreg")

        X_train, X_val, y_train, y_val = load_arrays()

        # Calcul des class weights de manière cohérente avec objective
        classes = np.unique(y_train)
        cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}

        # Construction du modèle final avec les meilleurs paramètres
        if model_choice == "lightgbm" and lgb:
            clf = lgb.LGBMClassifier(**{k: v for k, v in best_params.items() if k != "model"},
                                    class_weight=class_weight_dict)
        elif model_choice == "xgboost" and xgb:
            params = {k: v for k, v in best_params.items() if k != "model"}
            params.setdefault("eval_metric", "auc")
            clf = xgb.XGBClassifier(**params, scale_pos_weight=class_weight_dict.get(1, 1.0))
        elif model_choice == "catboost" and CatBoostClassifier:
            params = {k: v for k, v in best_params.items() if k != "model"}
            cw_val = [class_weight_dict.get(0, 1.0), class_weight_dict.get(1, 1.0)]
            clf = CatBoostClassifier(**params, class_weights=cw_val, verbose=False)
        else:
            C = best_params.get("C", 1.0)
            clf = LogisticRegression(C=C, max_iter=2000, n_jobs=-1, class_weight=class_weight_dict)

        # Entraînement final sur train+val combinés
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.hstack([y_train, y_val])
        clf.fit(X_combined, y_combined)

        mlflow.sklearn.log_model(clf, artifact_path="model")

        artifacts_dir = PROJECT_ROOT / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        # Log des métriques finales (best trial sur val)
        mlflow.log_metric("val_auc", study.best_trial.user_attrs["val_auc"])
        mlflow.log_metric("val_f1", study.best_trial.user_attrs["val_f1"])
        mlflow.log_metric("val_ap", study.best_trial.user_attrs["val_ap"])

        logger.info(f"Run MLflow: {run.info.run_id}")


if __name__ == "__main__":
    main()
