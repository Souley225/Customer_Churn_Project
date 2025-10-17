
"""Ingénierie de caractéristiques spécifique Telco Churn.

Principales idées (synthèse des meilleures pratiques Kaggle/Blogs/Articles):
- Conversion de `TotalCharges` (souvent string avec espaces) -> float, coercition des espaces en NaN puis imputation par median. (cf. notebooks Kaggle)
- Normalisation des catégories "No internet service" / "No phone service" en "No" pour réduire la cardinalité.
- Encodage binaire Yes/No -> 1/0 pour variables binaires.
- Création de *tenure bins* (groupes quantiles et buckets business), très informatifs pour le churn.
- Comptage des services souscrits (somme de colonnes de service == Yes) -> `num_services`.
- Interactions métier: `tenure * MonthlyCharges` (approx. dépense cumulée), interaction `Contract` x `PaperlessBilling`.
- Mise à l'échelle robuste (RobustScaler) pour numériques.
- Encodage catégoriel One-Hot pour nominales; Ordinal pour `Contract` (Month-to-month < One year < Two year).
- Gestion du déséquilibre via class_weight (modèles) et option SMOTE sur train uniquement (configurable).

Toutes les étapes sont réplicables avec Hydra et DVC.
"""
from __future__ import annotations
import joblib
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils.paths import INTERIM_DIR, PROCESSED_DIR
from src.utils.io import read_csv, to_csv
from src.utils.logging import logger


class TelcoCleaner(BaseEstimator, TransformerMixin):
    """Nettoyage et enrichissement spécifiques au dataset Telco.

    - Convertit TotalCharges en float (gestion d'espaces vides)
    - Normalise "No internet/phone service" -> "No"
    - Crée des features dérivées: tenure buckets, num_services, total_spend_proxy
    """

    def __init__(self, tenure_bins: Tuple[int, ...] = (0, 6, 12, 24, 48, 72)):
        self.tenure_bins = tenure_bins
        self.service_cols_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):  # type: ignore[override]
        # Identifie colonnes de services (contiennent typiquement Yes/No/No internet service)
        svc_candidates = [
            c for c in X.columns if any(k in c.lower() for k in ["phone", "internet", "security", "backup", "protection", "support", "tv", "movie", "lines"])
        ]
        # Retire quelques colonnes non-service connues
        blacklist = {"PhoneService", "PaperlessBilling"}
        self.service_cols_ = [c for c in svc_candidates if c not in blacklist]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        df = X.copy()
        # Convertir TotalCharges (espaces -> NaN -> float)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce")

        # Normaliser "No internet/phone service" -> "No"
        replace_vals = {"No internet service": "No", "No phone service": "No"}
        df = df.replace(replace_vals)

        # Binariser Yes/No -> 1/0 pour colonnes clairement binaires
        bin_cols = [
            c for c in df.columns if df[c].dropna().isin(["Yes", "No"]).all()
        ]
        for c in bin_cols:
            df[c] = (df[c] == "Yes").astype(int)

        # SeniorCitizen est 0/1 déjà numérique; s'assurer de type int
        if "SeniorCitizen" in df.columns:
            df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

        # Créer tenure buckets
        if "tenure" in df.columns:
            bins = list(self.tenure_bins) + [np.inf]
            labels = [f"[{bins[i]},{bins[i+1]})" for i in range(len(bins)-1)]
            df["tenure_bucket"] = pd.cut(df["tenure"], bins=bins, labels=labels, right=False)

        # Compter le nombre de services actifs (Yes)
        services = [c for c in self.service_cols_ if c in df.columns]
        if services:
            df["num_services"] = df[services].apply(lambda r: int(sum(v == 1 for v in r)), axis=1)
        else:
            df["num_services"] = 0

        # Dépense cumulée proxy
        if set(["tenure", "MonthlyCharges"]).issubset(df.columns):
            df["total_spend_proxy"] = df["tenure"].fillna(0) * df["MonthlyCharges"].fillna(0)

        # Interaction Contract x PaperlessBilling
        if set(["Contract", "PaperlessBilling"]).issubset(df.columns):
            df["contract_paperless"] = (
                df["Contract"].astype(str) + "_" + df["PaperlessBilling"].astype(str)
            )

        return df


@dataclass
class FeatureConfig:
    use_smote: bool = False


def build() -> None:
    """Construit X/y transformés et sauvegarde les splits traités.
    
    - Applique TelcoCleaner
    - Prépare ColumnTransformer (num -> imputer+scaler, cat->imputer+OneHot, contract->ordinal)
    - Sauvegarde X_*.npy et y_*.npy + CSV transformés pour audit
    """
    train = read_csv(INTERIM_DIR / "train.csv")
    val = read_csv(INTERIM_DIR / "val.csv")
    test = read_csv(INTERIM_DIR / "test.csv")

    # Nettoyage & enrichissement
    cleaner = TelcoCleaner()
    train = cleaner.fit_transform(train)
    val = cleaner.transform(val)
    test = cleaner.transform(test)

    # Séparer cible
    target = "Churn"
    y_train = (train[target] == "Yes").astype(int) if train[target].dtype == object else train[target]
    y_val = (val[target] == "Yes").astype(int) if val[target].dtype == object else val[target]
    y_test = (test[target] == "Yes").astype(int) if test[target].dtype == object else test[target]

    train = train.drop(columns=[target, "customerID"], errors="ignore")
    val = val.drop(columns=[target, "customerID"], errors="ignore")
    test = test.drop(columns=[target, "customerID"], errors="ignore")

    # Définir types
    numeric_features = train.select_dtypes(include=[np.number]).columns.tolist()
    # Exclure les binaires déjà encodées de l'OHE
    categorical_features = train.select_dtypes(include=["object", "category"]).columns.tolist()

    # Gérer Contract comme ordinal (mois->1 an->2 ans)
    ordinal_cols = [c for c in ["Contract"] if c in train.columns]
    if ordinal_cols:
        categorical_features = [c for c in categorical_features if c not in ordinal_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    ordinal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ord",
                OrdinalEncoder(
                    categories=[["Month-to-month", "One year", "Two year"]],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))
    if ordinal_cols:
        transformers.append(("ord", ordinal_transformer, ordinal_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    X_train = preprocessor.fit_transform(train)
    X_val = preprocessor.transform(val)
    X_test = preprocessor.transform(test)

    # Sauvegarde
    preprocessor_path = PROCESSED_DIR / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    logger.info("Preprocessor sauvegardé dans %s", preprocessor_path)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "X_val.npy", X_val)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_train.npy", y_train.values)
    np.save(PROCESSED_DIR / "y_val.npy", y_val.values)
    np.save(PROCESSED_DIR / "y_test.npy", y_test.values)

    # Pour audit humain
    to_csv(train, PROCESSED_DIR / "train_transformed_preview.csv")
    to_csv(val, PROCESSED_DIR / "val_transformed_preview.csv")
    to_csv(test, PROCESSED_DIR / "test_transformed_preview.csv")

    logger.info("Features construites et sauvegardées dans data/processed/")


if __name__ == "__main__":
    build()

