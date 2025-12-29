"""Module TelcoCleaner pour le deploiement.

Ce module contient uniquement la classe TelcoCleaner sans dependances
externes pour faciliter le chargement des artefacts sur Render.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TelcoCleaner(BaseEstimator, TransformerMixin):
    """Nettoyage et enrichissement specifiques au dataset Telco.

    - Convertit TotalCharges en float (gestion d'espaces vides)
    - Normalise "No internet/phone service" -> "No"
    - Cree des features derivees: tenure buckets, num_services, total_spend_proxy
    """

    def __init__(self, tenure_bins: tuple[int, ...] = (0, 6, 12, 24, 48, 72)) -> None:
        self.tenure_bins = tenure_bins
        self.service_cols_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> TelcoCleaner:  # noqa: N803
        """Identifie les colonnes de services."""
        svc_candidates = [
            c
            for c in X.columns
            if any(
                k in c.lower()
                for k in [
                    "phone",
                    "internet",
                    "security",
                    "backup",
                    "protection",
                    "support",
                    "tv",
                    "movie",
                    "lines",
                ]
            )
        ]
        blacklist = {"PhoneService", "PaperlessBilling"}
        self.service_cols_ = [c for c in svc_candidates if c not in blacklist]
        return self

    def transform(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Transforme le DataFrame avec nettoyage et feature engineering."""
        df = df_in.copy()

        # Convertir TotalCharges (espaces -> NaN -> float)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce")

        # Normaliser "No internet/phone service" -> "No"
        replace_vals = {"No internet service": "No", "No phone service": "No"}
        df = df.replace(replace_vals)

        # Binariser Yes/No -> 1/0 pour colonnes clairement binaires
        bin_cols = [c for c in df.columns if df[c].dropna().isin(["Yes", "No"]).all()]
        for c in bin_cols:
            df[c] = (df[c] == "Yes").astype(int)

        # SeniorCitizen est 0/1 deja numerique
        if "SeniorCitizen" in df.columns:
            df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

        # Creer tenure buckets
        if "tenure" in df.columns:
            bins = list(self.tenure_bins) + [np.inf]
            labels = [f"[{bins[i]},{bins[i+1]})" for i in range(len(bins) - 1)]
            df["tenure_bucket"] = pd.cut(df["tenure"], bins=bins, labels=labels, right=False)

        # Compter le nombre de services actifs (Yes)
        services = [c for c in self.service_cols_ if c in df.columns]
        if services:
            df["num_services"] = df[services].apply(lambda r: int(sum(v == 1 for v in r)), axis=1)
        else:
            df["num_services"] = 0

        # Depense cumulee proxy
        if set(["tenure", "MonthlyCharges"]).issubset(df.columns):
            df["total_spend_proxy"] = df["tenure"].fillna(0) * df["MonthlyCharges"].fillna(0)

        # Interaction Contract x PaperlessBilling
        if set(["Contract", "PaperlessBilling"]).issubset(df.columns):
            df["contract_paperless"] = (
                df["Contract"].astype(str) + "_" + df["PaperlessBilling"].astype(str)
            )

        return df
