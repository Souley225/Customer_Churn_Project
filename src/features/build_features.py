"""Module de feature engineering pour le dataset Telco Customer Churn.

- Conversion de TotalCharges (souvent string avec espaces) -> float
- Normalisation des categories "No internet service" / "No phone service" en "No"
- Encodage binaire Yes/No -> 1/0 pour variables binaires
- Creation de tenure bins (groupes quantiles et buckets business)
- Comptage des services souscrits -> num_services
- Interactions metier: tenure * MonthlyCharges, Contract x PaperlessBilling
- Mise a l'echelle robuste (RobustScaler) pour numeriques
- Encodage categoriel One-Hot pour nominales, Ordinal pour Contract

Toutes les etapes sont replicables avec Hydra et DVC.
"""

from __future__ import annotations

from dataclasses import dataclass

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
        """Applique le nettoyage et le feature engineering."""
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

        # Compter le nombre de services actifs
        services = [c for c in self.service_cols_ if c in df.columns]
        if services:
            df["num_services"] = df[services].apply(lambda r: int(sum(v == 1 for v in r)), axis=1)
        else:
            df["num_services"] = 0

        # Depense cumulee proxy
        if {"tenure", "MonthlyCharges"}.issubset(df.columns):
            df["total_spend_proxy"] = df["tenure"].fillna(0) * df["MonthlyCharges"].fillna(0)

        # Interaction Contract x PaperlessBilling
        if {"Contract", "PaperlessBilling"}.issubset(df.columns):
            df["contract_paperless"] = (
                df["Contract"].astype(str) + "_" + df["PaperlessBilling"].astype(str)
            )

        return df


@dataclass
class FeatureConfig:
    """Configuration pour le feature engineering."""

    use_smote: bool = False


def build() -> None:
    """Construit X/y transformes et sauvegarde les splits traites.

    - Applique TelcoCleaner
    - Prepare ColumnTransformer (num -> imputer+scaler, cat->imputer+OneHot)
    - Sauvegarde X_*.npy et y_*.npy + CSV transformes pour audit
    """
    # Imports locaux pour eviter les erreurs lors de l'import de TelcoCleaner
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

    from src.utils.io import read_csv, to_csv
    from src.utils.logging import logger
    from src.utils.paths import INTERIM_DIR, PROCESSED_DIR

    train = read_csv(INTERIM_DIR / "train.csv")
    val = read_csv(INTERIM_DIR / "val.csv")
    test = read_csv(INTERIM_DIR / "test.csv")

    # Nettoyage et enrichissement
    cleaner = TelcoCleaner()
    train = cleaner.fit_transform(train)
    val = cleaner.transform(val)
    test = cleaner.transform(test)

    # Separer cible
    target = "Churn"
    y_train = (
        (train[target] == "Yes").astype(int) if train[target].dtype == object else train[target]
    )
    y_val = (val[target] == "Yes").astype(int) if val[target].dtype == object else val[target]
    y_test = (test[target] == "Yes").astype(int) if test[target].dtype == object else test[target]

    train = train.drop(columns=[target, "customerID"], errors="ignore")
    val = val.drop(columns=[target, "customerID"], errors="ignore")
    test = test.drop(columns=[target, "customerID"], errors="ignore")

    # Definir types
    numeric_features = train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = train.select_dtypes(include=["object", "category"]).columns.tolist()

    # Gerer Contract comme ordinal
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

    x_train = preprocessor.fit_transform(train)
    x_val = preprocessor.transform(val)
    x_test = preprocessor.transform(test)

    # Sauvegarde du preprocessor ET du cleaner fitte
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    preprocessor_path = PROCESSED_DIR / "preprocessor.joblib"
    cleaner_path = PROCESSED_DIR / "cleaner.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(cleaner, cleaner_path)
    logger.info("Preprocessor sauvegarde dans %s", preprocessor_path)
    logger.info("Cleaner sauvegarde dans %s", cleaner_path)

    np.save(PROCESSED_DIR / "X_train.npy", x_train)
    np.save(PROCESSED_DIR / "X_val.npy", x_val)
    np.save(PROCESSED_DIR / "X_test.npy", x_test)
    np.save(PROCESSED_DIR / "y_train.npy", y_train.values)
    np.save(PROCESSED_DIR / "y_val.npy", y_val.values)
    np.save(PROCESSED_DIR / "y_test.npy", y_test.values)

    # Pour audit humain
    to_csv(train, PROCESSED_DIR / "train_transformed_preview.csv")
    to_csv(val, PROCESSED_DIR / "val_transformed_preview.csv")
    to_csv(test, PROCESSED_DIR / "test_transformed_preview.csv")

    logger.info("Features construites et sauvegardees dans data/processed/")


if __name__ == "__main__":
    build()
