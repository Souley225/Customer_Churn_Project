
## src/ui/app.py (Streamlit)
"""UI Streamlit pour scorer interactivement."""
from __future__ import annotations
import os
import sys
import joblib                 # 
import pandas as pd
import numpy as np
import mlflow
import streamlit as st

# Ajoute la racine au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.paths import PROCESSED_DIR

from src.features.build_features import TelcoCleaner   
from src.models.predict import predict_csv  # Pour le batch predict
import requests, zipfile, io, joblib
import  requests,  pathlib

import requests, zipfile, io, joblib, streamlit as st

GOOGLE_FILE_ID = "1yGarDcI4cdS6XqfOfyXcwOeqEaagqYSd"

@st.cache_data(show_spinner=False)
def load_artifacts():
    session = requests.Session()
    URL = "https://docs.google.com/uc?export=download"

    # 1ère requête : récupérer le token de confirmation (si Google en exige un)
    response = session.get(URL, params={"id": GOOGLE_FILE_ID}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    # 2ème requête : téléchargement effectif (avec le token si nécessaire)
    params = {"id": GOOGLE_FILE_ID, "confirm": token} if token else {"id": GOOGLE_FILE_ID}
    response = session.get(URL, params=params, stream=True)

    # Vérifications rapides
    if response.headers["Content-Type"].startswith("text/html"):
        raise RuntimeError("Contenu HTML reçu au lieu du zip – le token n’a pas fonctionné.")
    zip_bytes = response.content
    if len(zip_bytes) < 1_000:
        raise RuntimeError("Fichier trop petit – téléchargement invalide.")

    # Décompression dans le dossier local
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall("artifacts")

    # Chargement des objets sérialisés
    return (joblib.load("artifacts/preprocessor.joblib"),
            joblib.load("artifacts/model.joblib"))

preprocessor, model = load_artifacts()

# Colonnes complètes utilisées à l’entraînement (dans l’ordre)
EXPECTED_COLS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges',
    'TotalCharges', 'tenure_bucket', 'num_services', 'total_spend_proxy',
    'contract_paperless'
]

# Valeurs par défaut (même traitement que le cleaner)
DEFAULTS = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'PaymentMethod': 'Electronic check',
    'tenure_bucket': '[0,6)',
    'num_services': 0,
    'total_spend_proxy': 0.0,
    'contract_paperless': 'Month-to-month_1'
}
st.set_page_config(page_title="Telco Churn UI", layout="wide")
st.title("Telco Customer Churn — Demo")

MODEL_URI = os.getenv("MODEL_URI", 
                      os.getenv("MLFLOW_MODEL_URI", "models:/telco-churn-classifier/None"))
model = mlflow.sklearn.load_model(MODEL_URI)

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
paperless = st.selectbox("PaperlessBilling (Yes=1/No=0)", [0, 1])

if  st.button("Prédire le risque de churn"):
    # 1. création du dictionnaire COMPLET
    raw = DEFAULTS.copy()
    raw.update({
        'tenure': tenure,
        'MonthlyCharges': monthly,
        'TotalCharges': total,
        'Contract': contract,
        'PaperlessBilling': int(paperless),
    })
    # 2. DataFrame dans le bon ordre
    sample_raw = pd.DataFrame([{c: raw[c] for c in EXPECTED_COLS}])
    # 3. Nettoyage
    sample_clean = cleaner.transform(sample_raw)
    # 3. transformation + prédiction
    sample_proc = preprocessor.transform(sample_clean)
    proba = model.predict_proba(sample_proc)[:, 1][0]
    st.metric("Probabilité de churn", f"{proba:.2%}")

st.markdown("---")
st.subheader("Scoring batch")
file = st.file_uploader("CSV avec colonnes minimales", type=["csv"])
if file is not None:
    df = pd.read_csv(file)
    df_clean = cleaner.transform(df)
    df_proc = preprocessor.transform(df_clean)
    proba = model.predict_proba(df_proc)[:, 1]
    df_out = df.copy()
    df_out["churn_proba"] = proba
    st.dataframe(df_out.head())
    st.download_button("Télécharger résultats", 
                       df_out.to_csv(index=False).encode(), 
                       "predictions.csv")
