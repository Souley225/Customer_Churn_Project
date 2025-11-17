## src/ui/app.py (Streamlit)
"""UI Streamlit pour scorer interactivement."""
from __future__ import annotations

import os
import sys

import joblib
import mlflow
import pandas as pd
import streamlit as st

# Ajoute la racine au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.paths import PROCESSED_DIR

# Chargement du preprocessor et cleaner depuis PROCESSED_DIR
preprocessor = joblib.load(PROCESSED_DIR / "preprocessor.joblib")
cleaner = joblib.load(PROCESSED_DIR / "cleaner.joblib")

# Colonnes RAW attendues (avant transformation par TelcoCleaner)
RAW_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

# Valeurs par défaut pour les features non saisies
DEFAULTS = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "PaymentMethod": "Electronic check",
}
st.set_page_config(page_title="Telco Churn UI", layout="wide")
st.title("Telco Customer Churn — Demo")

# Chargement du modèle avec fallback MLflow -> PROCESSED_DIR
MODEL_URI = os.getenv(
    "MODEL_URI", os.getenv("MLFLOW_MODEL_URI", "models:/telco-churn-classifier/Production")
)

try:
    model = mlflow.sklearn.load_model(MODEL_URI)
    st.sidebar.success(f"Modèle chargé depuis MLflow: {MODEL_URI}")
except Exception as e:
    st.sidebar.warning(f"Échec chargement MLflow ({MODEL_URI}): {e}")
    # Fallback: charger le modèle depuis PROCESSED_DIR
    model_path = PROCESSED_DIR / "model.joblib"
    if not model_path.exists():
        st.error(f"Modèle non trouvé ni dans MLflow ni dans {model_path}")
        st.stop()
    model = joblib.load(model_path)
    st.sidebar.info(f"Modèle chargé depuis fallback: {model_path}")

st.sidebar.header("Caractéristiques")
contracts = ["Month-to-month", "One year", "Two year"]
col1, col2, col3 = st.columns(3)
with col1:
    tenure = st.number_input("tenure", 0, 72, 5)
with col2:
    monthly = st.number_input("MonthlyCharges", 0.0, 200.0, 70.0)
with col3:
    total = st.number_input("TotalCharges", 0.0, 10000.0, 0.0)

contract = st.selectbox("Contract", contracts)
paperless = st.selectbox("PaperlessBilling", ["Yes", "No"])

if st.button("Prédire le risque de churn"):
    # Création du dictionnaire avec features RAW
    raw = DEFAULTS.copy()
    raw.update(
        {
            "tenure": int(tenure),
            "MonthlyCharges": float(monthly),
            "TotalCharges": str(total) if total > 0 else " ",
            "Contract": contract,
            "PaperlessBilling": paperless,
        }
    )
    # DataFrame avec colonnes RAW uniquement
    sample_raw = pd.DataFrame([{c: raw[c] for c in RAW_COLS}])

    # Application du pipeline complet: cleaner -> preprocessor -> modèle
    sample_clean = cleaner.transform(sample_raw)
    sample_proc = preprocessor.transform(sample_clean)
    proba = model.predict_proba(sample_proc)[:, 1][0]
    st.metric("Probabilité de churn", f"{proba:.2%}")

st.markdown("---")
st.subheader("Scoring batch")
file = st.file_uploader("CSV avec colonnes client complètes", type=["csv"])
if file is not None:
    df = pd.read_csv(file)
    # Application du pipeline complet
    df_clean = cleaner.transform(df)
    df_proc = preprocessor.transform(df_clean)
    proba = model.predict_proba(df_proc)[:, 1]
    df_out = df.copy()
    df_out["churn_proba"] = proba
    st.dataframe(df_out.head())
    st.download_button(
        "Télécharger résultats", df_out.to_csv(index=False).encode(), "predictions.csv"
    )
