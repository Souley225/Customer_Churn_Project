
## src/ui/app.py (Streamlit)

"""UI Streamlit pour scorer interactivement.

Commentaires en français.
"""
from __future__ import annotations
import os
import mlflow
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Telco Churn UI", layout="wide")
st.title("Telco Customer Churn — Demo")

MODEL_URI = os.getenv("MODEL_URI", os.getenv("MLFLOW_MODEL_URI", "models:/telco-churn-classifier/None"))
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

if st.button("Prédire le risque de churn"):
    sample = pd.DataFrame([
        {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Contract": contract,
            "PaperlessBilling": int(paperless),
        }
    ])
    proba = model.predict_proba(sample)[:, 1][0]
    st.metric("Probabilité de churn", f"{proba:.2%}")

st.markdown("---")
st.subheader("Scoring batch")
file = st.file_uploader("CSV avec colonnes minimales", type=["csv"])
if file is not None:
    df = pd.read_csv(file)
    proba = model.predict_proba(df)[:, 1]
    df_out = df.copy()
    df_out["churn_proba"] = proba
    st.dataframe(df_out.head())
    st.download_button("Télécharger résultats", df_out.to_csv(index=False).encode(), "predictions.csv")
