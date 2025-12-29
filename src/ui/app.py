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

# Valeurs par defaut pour les features non saisies
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


@st.cache_resource
def load_artifacts() -> tuple:
    """Charge les artefacts de prediction (preprocessor, cleaner, model)."""
    preprocessor = joblib.load(PROCESSED_DIR / "preprocessor.joblib")
    cleaner = joblib.load(PROCESSED_DIR / "cleaner.joblib")

    use_local = os.getenv("USE_LOCAL_ARTIFACTS", "false").lower() == "true"
    model_uri = os.getenv(
        "MODEL_URI", os.getenv("MLFLOW_MODEL_URI", "models:/telco-churn-classifier/Production")
    )

    if use_local:
        model_path = PROCESSED_DIR / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Modele local non trouve: {model_path}")
        model = joblib.load(model_path)
        source = "local"
    else:
        try:
            model = mlflow.sklearn.load_model(model_uri)
            source = "mlflow"
        except Exception:
            model_path = PROCESSED_DIR / "model.joblib"
            if not model_path.exists():
                raise FileNotFoundError(f"Modèle non trouvé : {model_path}") from None
            model = joblib.load(model_path)
            source = "fallback"
    return preprocessor, cleaner, model, source


# Configuration de la page
st.set_page_config(page_title="Telco Churn UI", layout="wide")
st.title("Telco Customer Churn - Demo")

# Chargement des artefacts
try:
    preprocessor, cleaner, model, source = load_artifacts()
    if source == "local":
        st.sidebar.info("Modele charge depuis artefacts locaux")
    elif source == "mlflow":
        st.sidebar.success("Modele charge depuis MLflow")
    else:
        st.sidebar.warning("Modele charge depuis fallback local")
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.sidebar.header("Caracteristiques")
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

if st.button("Predire le risque de churn"):
    # Creation du dictionnaire avec features RAW
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

    # Application du pipeline complet: cleaner -> preprocessor -> modele
    sample_clean = cleaner.transform(sample_raw)
    sample_proc = preprocessor.transform(sample_clean)
    proba = model.predict_proba(sample_proc)[:, 1][0]
    st.metric("Probabilite de churn", f"{proba:.2%}")

st.markdown("---")
st.subheader("Scoring batch")
file = st.file_uploader("CSV avec colonnes client completes", type=["csv"])
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
        "Telecharger resultats", df_out.to_csv(index=False).encode(), "predictions.csv"
    )
