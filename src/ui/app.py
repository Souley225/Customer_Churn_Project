"""UI Streamlit professionnelle pour demonstration du projet MLOps."""

from __future__ import annotations

import os
import sys

import joblib
import mlflow
import pandas as pd
import streamlit as st

# Ajoute la racine au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import obligatoire pour deserialiser cleaner.joblib
from src.features.build_features import TelcoCleaner
from src.utils.paths import PROCESSED_DIR

cleaner = TelcoCleaner()

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
                raise FileNotFoundError(f"Modele non trouve : {model_path}") from None
            model = joblib.load(model_path)
            source = "fallback"
    return preprocessor, cleaner, model, source


# Configuration de la page
st.set_page_config(
    page_title="Prediction Churn Client | Projet MLOps",
    page_icon="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/python.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personnalise pour un design professionnel
st.markdown(
    """
    <style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .main-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .tech-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
    }
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1.3rem;
        font-weight: 600;
        color: #1f2937;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    .info-card {
        background: #f8fafc;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        color: #1f2937;
    }
    .info-card i {
        color: #667eea;
    }
    .result-card-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }
    .result-card-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: #1f2937;
        text-align: center;
        text-shadow: 0 1px 1px rgba(255,255,255,0.3);
    }
    .result-card-high {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }
    .proba-value {
        font-size: 3rem;
        font-weight: 700;
    }
    .proba-label {
        font-size: 1.1rem;
        font-weight: 500;
    }
    .recommendation {
        background: #fefce8;
        border-left: 4px solid #ca8a04;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-top: 1rem;
        color: #713f12;
    }
    .recommendation strong {
        color: #854d0e;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid #e5e7eb;
        color: #4b5563;
    }
    .footer a {
        color: #667eea;
        text-decoration: none;
        font-weight: 500;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    .sidebar-info {
        background: #f0f9ff;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        color: #1e40af;
    }
    .sidebar-info h3 {
        color: #1e3a8a;
        margin: 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# En-tete principal
st.markdown(
    """
    <div class="main-header">
        <div class="main-title">
            <i class="fas fa-chart-line"></i> Prediction du Churn Client
        </div>
        <div class="main-subtitle">
            Projet MLOps complet avec pipeline de Machine Learning industrialise
        </div>
        <div style="margin-top: 1rem;">
            <span class="tech-badge"><i class="fas fa-flask"></i> MLflow</span>
            <span class="tech-badge"><i class="fas fa-database"></i> DVC</span>
            <span class="tech-badge"><i class="fas fa-cube"></i> Docker</span>
            <span class="tech-badge"><i class="fas fa-bolt"></i> FastAPI</span>
            <span class="tech-badge"><i class="fab fa-python"></i> Python 3.11</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Chargement des artefacts
try:
    preprocessor, cleaner, model, source = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Erreur de chargement : {str(e)}")
    st.stop()

# Barre laterale - A propos du projet
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-info">
            <h3><i class="fas fa-info-circle"></i> A propos du projet</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        Ce projet demontre un **pipeline MLOps complet** pour la prediction
        de l'attrition client (churn) dans le secteur des telecommunications.

        **contenu du projet :**
        - Pipeline de donnees reproductible (DVC)
        - Suivi des experiences (MLflow)
        - Feature engineering avance
        - Optimisation hyperparametres (Optuna)
        - API REST (FastAPI)
        - Conteneurisation (Docker)
        - Deploiement cloud (Render)
        """
    )

    st.markdown("---")

    # Statut du modele
    if source == "local":
        st.info("")
    elif source == "mlflow":
        st.success("Modele charge depuis MLflow Registry")
    else:
        st.warning("")

    st.markdown("---")

    st.markdown(
        """
        <div style="text-align: center; color: #6b7280; font-size: 0.9rem;">
            <a href="https://github.com/Souley225/Customer_Churn_Project" target="_blank">
                <i class="fab fa-github"></i> Voir le code source
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Contenu principal avec onglets
tab1, tab2 = st.tabs(["Prediction individuelle", "Scoring par lot"])

# ===================== ONGLET 1 : Prediction individuelle =====================
with tab1:
    st.markdown(
        """
        <div class="section-header">
            <i class="fas fa-user"></i> Caracteristiques du client
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
            <i class="fas fa-lightbulb"></i>
            <strong>Instructions :</strong> Renseignez les informations du client pour obtenir
            une estimation de la probabilite de resiliation. Les champs non affiches utilisent
            des valeurs par defaut representatives.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Formulaire organise en colonnes
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Informations de facturation**")
        tenure = st.number_input(
            "Anciennete (mois)",
            min_value=0,
            max_value=72,
            value=12,
            help="Nombre de mois depuis l'inscription du client",
        )
        monthly = st.number_input(
            "Charges mensuelles (EUR)",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            help="Montant facture mensuellement",
        )
        total = st.number_input(
            "Charges totales (EUR)",
            min_value=0.0,
            max_value=10000.0,
            value=840.0,
            help="Montant total facture depuis l'inscription",
        )

    with col2:
        st.markdown("**Type de contrat**")
        contracts = ["Month-to-month", "One year", "Two year"]
        contract_labels = {
            "Month-to-month": "Mensuel (sans engagement)",
            "One year": "Annuel (1 an)",
            "Two year": "Bisannuel (2 ans)",
        }
        contract = st.selectbox(
            "Type de contrat",
            contracts,
            format_func=lambda x: contract_labels[x],
            help="Duree d'engagement du client",
        )

        paperless = st.selectbox(
            "Facturation electronique",
            ["Yes", "No"],
            format_func=lambda x: "Oui" if x == "Yes" else "Non",
            help="Le client recoit-il ses factures par email ?",
        )

    # Bouton de prediction
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button(
        "Calculer le risque de churn",
        type="primary",
        use_container_width=True,
    )

    if predict_button:
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

        st.markdown("<br>", unsafe_allow_html=True)

        # Affichage du resultat avec code couleur
        if proba < 0.3:
            risk_class = "result-card-low"
            risk_label = "Risque faible"
            risk_icon = "fas fa-check-circle"
            recommendation = (
                "Ce client presente un profil stable. Maintenez la qualite de service actuelle."
            )
        elif proba < 0.6:
            risk_class = "result-card-medium"
            risk_label = "Risque modere"
            risk_icon = "fas fa-exclamation-triangle"
            recommendation = (
                "Attention requise. Envisagez des actions de fidelisation "
                "(offre promotionnelle, upgrade de services)."
            )
        else:
            risk_class = "result-card-high"
            risk_label = "Risque eleve"
            risk_icon = "fas fa-times-circle"
            recommendation = (
                "Action urgente recommandee. Contactez le client pour comprendre "
                "ses besoins et proposer des solutions adaptees."
            )

        col_result1, col_result2, col_result3 = st.columns([1, 2, 1])

        with col_result2:
            st.markdown(
                f"""
                <div class="{risk_class}">
                    <div class="proba-label"><i class="{risk_icon}"></i> {risk_label}</div>
                    <div class="proba-value">{proba:.1%}</div>
                    <div class="proba-label">Probabilite de resiliation</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="recommendation">
                <strong><i class="fas fa-lightbulb"></i> Recommandation :</strong><br>
                {recommendation}
            </div>
            """,
            unsafe_allow_html=True,
        )

# ===================== ONGLET 2 : Scoring par lot =====================
with tab2:
    st.markdown(
        """
        <div class="section-header">
            <i class="fas fa-file-csv"></i> Scoring par lot
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
            <i class="fas fa-info-circle"></i>
            <strong>Mode batch :</strong> Chargez un fichier CSV contenant les donnees de plusieurs
            clients pour obtenir leurs probabilites de churn respectives. Le fichier doit contenir
            toutes les colonnes necessaires (gender, tenure, Contract, etc.).
        </div>
        """,
        unsafe_allow_html=True,
    )

    file = st.file_uploader(
        "Selectionnez un fichier CSV",
        type=["csv"],
        help="Format attendu : CSV avec colonnes correspondant aux caracteristiques client",
    )

    if file is not None:
        try:
            df = pd.read_csv(file)

            st.markdown("**Apercu des donnees chargees :**")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("Lancer le scoring", type="primary"):
                with st.spinner("Calcul en cours..."):
                    # Application du pipeline complet
                    df_clean = cleaner.transform(df)
                    df_proc = preprocessor.transform(df_clean)
                    proba = model.predict_proba(df_proc)[:, 1]

                    df_out = df.copy()
                    df_out["churn_proba"] = proba
                    df_out["risque"] = df_out["churn_proba"].apply(
                        lambda x: "Eleve" if x >= 0.6 else ("Modere" if x >= 0.3 else "Faible")
                    )

                st.success(f"Scoring termine pour {len(df_out)} clients")

                # Statistiques
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Clients a risque eleve", f"{(df_out['risque'] == 'Eleve').sum()}")
                with col_stat2:
                    st.metric("Clients a risque modere", f"{(df_out['risque'] == 'Modere').sum()}")
                with col_stat3:
                    st.metric("Clients a risque faible", f"{(df_out['risque'] == 'Faible').sum()}")

                st.markdown("**Resultats :**")
                st.dataframe(
                    df_out[["churn_proba", "risque"]].join(
                        df[["tenure", "MonthlyCharges", "Contract"]]
                    ),
                    use_container_width=True,
                )

                st.download_button(
                    "Telecharger les resultats (CSV)",
                    df_out.to_csv(index=False).encode("utf-8"),
                    "predictions_churn.csv",
                    "text/csv",
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"Erreur lors du traitement : {str(e)}")

# Pied de page
st.markdown(
    """
    <div class="footer">
        <p>
            <strong>Projet MLOps - Prediction du Churn Client</strong><br>
            Pipeline complet : DVC | MLflow | FastAPI | Docker | Streamlit
        </p>
        <p>
            <a href="https://www.linkedin.com/in/souleymanes-sall" target="_blank">
                <i class="fab fa-linkedin"></i> LinkedIn
            </a>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            <a href="https://github.com/Souley225/Customer_Churn_Project" target="_blank">
                <i class="fab fa-github"></i> Repository GitHub
            </a>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            <a href="https://github.com/Souley225" target="_blank">
                <i class="fas fa-user"></i> Souleymane Sall
            </a>
        </p>
        <p style="font-size: 0.8rem;">
            Licence MIT | Donnees : Telco Customer Churn (Kaggle)
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
